import { fstat, readFileSync, writeFileSync } from 'fs';
import * as path from 'path';
import * as process from 'process';
import type { IGraphCardinalityPredictionLayers, IQueryGraphEncodingModels, ITrainResult } from '@comunica/actor-rdf-join-inner-multi-reinforcement-learning-tree';
import { InstanceModelGCN, InstanceModelLSTM, ModelTrainerOffline } from '@comunica/actor-rdf-join-inner-multi-reinforcement-learning-tree';
import { DenseOwnImplementation } from '@comunica/actor-rdf-join-inner-multi-reinforcement-learning-tree/lib/fullyConnectedLayers';
import { BindingsFactory } from '@comunica/bindings-factory';
import { materializeOperation } from '@comunica/bus-query-operation';
import type { IActionSparqlSerialize, IActorQueryResultSerializeOutput } from '@comunica/bus-query-result-serialize';
import { KeysCore, KeysInitQuery, KeysRdfResolveQuadPattern, KeysRlTrain } from '@comunica/context-entries';
import { ActionContext } from '@comunica/core';
import type { IAggregateValues, IQueryGraphViews, IResultSetRepresentation, IRunningMoments } from '@comunica/mediator-join-reinforcement-learning';
import { MediatorJoinReinforcementLearning } from '@comunica/mediator-join-reinforcement-learning';
import { ExperienceBuffer } from '@comunica/model-trainer';
import type {
  IActionContext, IPhysicalQueryPlanLogger,
  IQueryOperationResult,
  IQueryEngine, IQueryExplained,
  QueryFormatType,
  QueryType, QueryExplainMode, BindingsStream,
  QueryAlgebraContext, QueryStringContext, IQueryBindingsEnhanced,
  IQueryQuadsEnhanced, QueryEnhanced, IQueryContextCommon, FunctionArgumentsCache,
  ITrainEpisode,
  IBatchedTrainingExamples,
  ITrainingExample,
} from '@comunica/types';
import type * as RDF from '@rdfjs/types';
import type { AsyncIterator} from 'asynciterator';
import { ArrayIterator } from 'asynciterator';
import type { Algebra } from 'sparqlalgebrajs';
import type { ActorInitQueryBase } from './ActorInitQueryBase';
import { MemoryPhysicalQueryPlanLogger } from './MemoryPhysicalQueryPlanLogger';
import * as _ from 'lodash';
import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs'

/**
 * Base implementation of a Comunica query engine.
 */
export class QueryEngineBase<QueryContext extends IQueryContextCommon = IQueryContextCommon>
implements IQueryEngine<QueryContext> {
  private readonly actorInitQuery: ActorInitQueryBase;
  private readonly defaultFunctionArgumentsCache: FunctionArgumentsCache;
  public modelTrainerOffline: ModelTrainerOffline;
  public modelInstanceLSTM: InstanceModelLSTM;
  public modelInstanceGCN: InstanceModelGCN;
  public batchedTrainExamples: IBatchedTrainingExamples;

  public trainEpisode: ITrainEpisode;
  public BF: BindingsFactory;

  public constructor(actorInitQuery: ActorInitQueryBase<QueryContext>) {
    this.actorInitQuery = actorInitQuery;
    this.defaultFunctionArgumentsCache = {};
    // Initialise empty episode on creation
    this.emptyTrainEpisode();
    this.batchedTrainExamples = { trainingExamples: new Map<string, ITrainingExample>(), leafFeatures: { hiddenStates: [], memoryCell: []}};
    this.BF = new BindingsFactory();
  }

  public async queryBindings<QueryFormatTypeInner extends QueryFormatType>(
    query: QueryFormatTypeInner,
    context?: QueryContext & QueryFormatTypeInner extends string ? QueryStringContext : QueryAlgebraContext,
  ): Promise<BindingsStream> {
    return this.queryOfType<QueryFormatTypeInner, IQueryBindingsEnhanced>(query, context, 'bindings');
  }

  public async queryQuads<QueryFormatTypeInner extends QueryFormatType>(
    query: QueryFormatTypeInner,
    context?: QueryContext & QueryFormatTypeInner extends string ? QueryStringContext : QueryAlgebraContext,
  ): Promise<AsyncIterator<RDF.Quad> & RDF.ResultStream<RDF.Quad>> {
    return this.queryOfType<QueryFormatTypeInner, IQueryQuadsEnhanced>(query, context, 'quads');
  }

  public async queryBoolean<QueryFormatTypeInner extends QueryFormatType>(
    query: QueryFormatTypeInner,
    context?: QueryContext & QueryFormatTypeInner extends string ? QueryStringContext : QueryAlgebraContext,
  ): Promise<boolean> {
    return this.queryOfType<QueryFormatTypeInner, RDF.QueryBoolean>(query, context, 'boolean');
  }

  public async queryVoid<QueryFormatTypeInner extends QueryFormatType>(
    query: QueryFormatTypeInner,
    context?: QueryContext & QueryFormatTypeInner extends string ? QueryStringContext : QueryAlgebraContext,
  ): Promise<void> {
    return this.queryOfType<QueryFormatTypeInner, RDF.QueryVoid>(query, context, 'void');
  }

  protected async queryOfType<QueryFormatTypeInner extends QueryFormatType, QueryTypeOut extends QueryEnhanced>(
    query: QueryFormatTypeInner,
    context: undefined | (QueryContext & QueryFormatTypeInner extends string ?
      QueryStringContext : QueryAlgebraContext),
    expectedType: QueryTypeOut['resultType'],
    randomId?: number,
  ): Promise<ReturnType<QueryTypeOut['execute']>> {
    const result = await this.query<QueryFormatTypeInner>(query, context, randomId);
    if (result.resultType === expectedType) {
      return result.execute();
    }
    throw new Error(`Query result type '${expectedType}' was expected, while '${result.resultType}' was found.`);
  }

  /**
   * Evaluate the given query
   * @param query A query string or algebra.
   * @param context An optional query context.
   * @return {Promise<QueryType>} A promise that resolves to the query output.
   */
  public async query<QueryFormatTypeInner extends QueryFormatType>(
    query: QueryFormatTypeInner,
    context?: QueryContext & QueryFormatTypeInner extends string ? QueryStringContext : QueryAlgebraContext,
    randomId?: number,
  ): Promise<QueryType> {
    const output = await this.queryOrExplain(query, context);
    if ('explain' in output) {
      throw new Error(`Tried to explain a query when in query-only mode`);
    }
    return output;
  }

  /**
   * Explain the given query
   * @param {string | Algebra.Operation} query A query string or algebra.
   * @param context An optional query context.
   * @param explainMode The explain mode.
   * @return {Promise<QueryType | IQueryExplained>} A promise that resolves to
   *                                                               the query output or explanation.
   */
  public async explain<QueryFormatTypeInner extends QueryFormatType>(
    query: QueryFormatTypeInner,
    context: QueryContext & QueryFormatTypeInner extends string ? QueryStringContext : QueryAlgebraContext,
    explainMode: QueryExplainMode,
  ): Promise<IQueryExplained> {
    context.explain = explainMode;
    const output = await this.queryOrExplain(query, context);
    return <IQueryExplained> output;
  }

  /**
   * Evaluate or explain the given query
   * @param {string | Algebra.Operation} query A query string or algebra.
   * @param context An optional query context.
   * @return {Promise<QueryType | IQueryExplained>} A promise that resolves to
   *                                                               the query output or explanation.
   */
  public async queryOrExplain<QueryFormatTypeInner extends QueryFormatType>(
    query: QueryFormatTypeInner,
    context?: QueryContext & QueryFormatTypeInner extends string ? QueryStringContext : QueryAlgebraContext,
  ): Promise<QueryType | IQueryExplained> {
    context = context || <any>{};

    // Expand shortcuts
    for (const key in context) {
      if (this.actorInitQuery.contextKeyShortcuts[key]) {
        context[this.actorInitQuery.contextKeyShortcuts[key]] = context[key];
        delete context[key];
      }
    }

    // Prepare context
    let actionContext: IActionContext = new ActionContext(context);
    let queryFormat: RDF.QueryFormat = { language: 'sparql', version: '1.1' };
    if (actionContext.has(KeysInitQuery.queryFormat)) {
      queryFormat = actionContext.get(KeysInitQuery.queryFormat)!;
      actionContext = actionContext.delete(KeysInitQuery.queryFormat);
      if (queryFormat.language === 'graphql') {
        actionContext = actionContext.setDefault(KeysInitQuery.graphqlSingularizeVariables, {});
      }
    }
    const baseIRI: string | undefined = actionContext.get(KeysInitQuery.baseIRI);
    actionContext = actionContext
      .setDefault(KeysInitQuery.queryTimestamp, new Date())
      .setDefault(KeysRdfResolveQuadPattern.sourceIds, new Map())
      // Set the default logger if none is provided
      .setDefault(KeysCore.log, this.actorInitQuery.logger)
      .setDefault(KeysInitQuery.functionArgumentsCache, this.defaultFunctionArgumentsCache)

    // Pre-processing the context
    actionContext = (await this.actorInitQuery.mediatorContextPreprocess.mediate({ context: actionContext })).context;

    // Determine explain mode
    const explainMode: QueryExplainMode = actionContext.get(KeysInitQuery.explain)!;

    // Parse query
    let operation: Algebra.Operation;
    if (typeof query === 'string') {
      // Save the original query string in the context
      actionContext = actionContext.set(KeysInitQuery.queryString, query);

      const queryParseOutput = await this.actorInitQuery.mediatorQueryParse
        .mediate({ context: actionContext, query, queryFormat, baseIRI });
      operation = queryParseOutput.operation;
      // Update the baseIRI in the context if the query modified it.
      if (queryParseOutput.baseIRI) {
        actionContext = actionContext.set(KeysInitQuery.baseIRI, queryParseOutput.baseIRI);
      }
    } else {
      operation = query;
    }

    // Print parsed query
    if (explainMode === 'parsed') {
      return {
        explain: true,
        type: explainMode,
        data: operation,
      };
    }

    // Apply initial bindings in context
    if (actionContext.has(KeysInitQuery.initialBindings)) {
      operation = materializeOperation(operation, actionContext.get(KeysInitQuery.initialBindings)!);

      // Delete the query string from the context, since our initial query might have changed
      actionContext = actionContext.delete(KeysInitQuery.queryString);
    }

    // Optimize the query operation
    const mediatorResult = await this.actorInitQuery.mediatorOptimizeQueryOperation
      .mediate({ context: actionContext, operation });
    operation = mediatorResult.operation;
    actionContext = mediatorResult.context || actionContext;

    // Print logical query plan
    if (explainMode === 'logical') {
      return {
        explain: true,
        type: explainMode,
        data: operation,
      };
    }

    // Save original query in context
    actionContext = actionContext.set(KeysInitQuery.query, operation);

    // If we need a physical query plan, store a physical query plan logger in the context, and collect it after exec
    let physicalQueryPlanLogger: IPhysicalQueryPlanLogger | undefined;
    if (explainMode === 'physical') {
      physicalQueryPlanLogger = new MemoryPhysicalQueryPlanLogger();
      actionContext = actionContext.set(KeysInitQuery.physicalQueryPlanLogger, physicalQueryPlanLogger);
    }

    // Execute query
    const output = await this.actorInitQuery.mediatorQueryOperation.mediate({
      context: actionContext,
      operation,
    });
    output.context = actionContext;

    const finalOutput = QueryEngineBase.internalToFinalResult(output);

    // Output physical query plan after query exec if needed
    if (physicalQueryPlanLogger) {
      // Make sure the whole result is produced
      switch (finalOutput.resultType) {
        case 'bindings':
          await (await finalOutput.execute()).toArray();
          break;
        case 'quads':
          await (await finalOutput.execute()).toArray();
          break;
        case 'boolean':
          await finalOutput.execute();
          break;
        case 'void':
          await finalOutput.execute();
          break;
      }

      return {
        explain: true,
        type: explainMode,
        data: physicalQueryPlanLogger.toJson(),
      };
    }
    return finalOutput;
  }

  public async getFeaturesQuery<QueryFormatTypeInner extends QueryFormatType>(
    query: QueryFormatTypeInner,
    context?: QueryContext & QueryFormatTypeInner extends string ? QueryStringContext : QueryAlgebraContext,
  ): Promise < IQueryFeatures > {
    context = context || <any>{};

    if (context) {
      context.batchedTrainingExamples = this.batchedTrainExamples;
    }

    // Expand shortcuts
    for (const key in context) {
      if (this.actorInitQuery.contextKeyShortcuts[key]) {
        context[this.actorInitQuery.contextKeyShortcuts[key]] = context[key];
        delete context[key];
      }
    }
    // Prepare context
    let actionContext: IActionContext = new ActionContext(context);
    let queryFormat: RDF.QueryFormat = { language: 'sparql', version: '1.1' };
    if (actionContext.has(KeysInitQuery.queryFormat)) {
      queryFormat = actionContext.get(KeysInitQuery.queryFormat)!;
      actionContext = actionContext.delete(KeysInitQuery.queryFormat);
      if (queryFormat.language === 'graphql') {
        actionContext = actionContext.setDefault(KeysInitQuery.graphqlSingularizeVariables, {});
      }
    }

    const baseIRI: string | undefined = actionContext.get(KeysInitQuery.baseIRI);

    actionContext = actionContext
      .setDefault(KeysInitQuery.queryTimestamp, new Date())
      .setDefault(KeysRdfResolveQuadPattern.sourceIds, new Map())
      // Set the default logger if none is provided
      .setDefault(KeysCore.log, this.actorInitQuery.logger)
      .setDefault(KeysInitQuery.functionArgumentsCache, this.defaultFunctionArgumentsCache)
      .setDefault(KeysRlTrain.trainEpisode, this.trainEpisode)
      .setDefault(KeysRlTrain.modelInstanceLSTM, this.modelInstanceLSTM)
      .setDefault(KeysRlTrain.modelInstanceGCN, this.modelInstanceGCN)
      // We dont want to use running moments as we are creating a database before training
      // Thus we can simply use the resulting mean and standard deviation of training examples
      .setDefault(KeysRlTrain.trackRunningMoments, false);

    // Pre-processing the context
    actionContext = (await this.actorInitQuery.mediatorContextPreprocess.mediate({ context: actionContext })).context;

    // Parse query
    let operation: Algebra.Operation;
    if (typeof query === 'string') {
      // Save the original query string in the context
      actionContext = actionContext.set(KeysInitQuery.queryString, query);

      const queryParseOutput = await this.actorInitQuery.mediatorQueryParse
        .mediate({ context: actionContext, query, queryFormat, baseIRI });
      operation = queryParseOutput.operation;
      // Update the baseIRI in the context if the query modified it.
      if (queryParseOutput.baseIRI) {
        actionContext = actionContext.set(KeysInitQuery.baseIRI, queryParseOutput.baseIRI);
      }
    } else {
      operation = query;
    }

    // Apply initial bindings in context
    if (actionContext.has(KeysInitQuery.initialBindings)) {
      operation = materializeOperation(operation, actionContext.get(KeysInitQuery.initialBindings)!);

      // Delete the query string from the context, since our initial query might have changed
      actionContext = actionContext.delete(KeysInitQuery.queryString);
    }
    // Optimize the query operation
    const mediatorResult = await this.actorInitQuery.mediatorOptimizeQueryOperation
      .mediate({ context: actionContext, operation });
    operation = mediatorResult.operation;
    actionContext = mediatorResult.context || actionContext;

    // Save original query in context
    actionContext = actionContext.set(KeysInitQuery.query, operation);

    // Execute query
    const numTensorBeforeQuery = tf.memory().numTensors;
    const output = await this.actorInitQuery.mediatorQueryOperation.mediate({
      context: actionContext,
      operation,
    });


    const featuresQuery = (<ITrainEpisode> actionContext.get(KeysRlTrain.trainEpisode)!).leafFeatureTensor;
    const graphViews = (<ITrainEpisode> actionContext.get(KeysRlTrain.trainEpisode)!).graphViews;

    // Reset train episode if we are not training (Bit of a shoddy workaround, because we need some episode information
    // If we train
    if (!actionContext.get(KeysRlTrain.train) && context && !context.trainEndPoint) {
      this.emptyTrainEpisode();
    }

    return {
      leafFeatures: featuresQuery,
      graphViews,
    };
  }

  /**
   * @param context An optional context.
   * @return {Promise<{[p: string]: number}>} All available SPARQL (weighted) result media types.
   */
  public async getResultMediaTypes(context?: any): Promise<Record<string, number>> {
    context = ActionContext.ensureActionContext(context);
    return (await this.actorInitQuery.mediatorQueryResultSerializeMediaTypeCombiner
      .mediate({ context, mediaTypes: true })).mediaTypes;
  }

  /**
   * @param context An optional context.
   * @return {Promise<{[p: string]: number}>} All available SPARQL result media type formats.
   */
  public async getResultMediaTypeFormats(context?: any): Promise<Record<string, string>> {
    context = ActionContext.ensureActionContext(context);
    return (await this.actorInitQuery.mediatorQueryResultSerializeMediaTypeFormatCombiner
      .mediate({ context, mediaTypeFormats: true })).mediaTypeFormats;
  }

  /**
   * Convert a query result to a string stream based on a certain media type.
   * @param {IQueryOperationResult} queryResult A query result.
   * @param {string} mediaType A media type.
   * @param {ActionContext} context An optional context.
   * @return {Promise<IActorQueryResultSerializeOutput>} A text stream.
   */
  public async resultToString(queryResult: RDF.Query<any>, mediaType?: string, context?: any):
  Promise<IActorQueryResultSerializeOutput> {
    context = ActionContext.ensureActionContext(context);
    if (!mediaType) {
      switch (queryResult.resultType) {
        case 'bindings':
          mediaType = 'application/json';
          break;
        case 'quads':
          mediaType = 'application/trig';
          break;
        default:
          mediaType = 'simple';
          break;
      }
    }
    const handle: IActionSparqlSerialize = { ...await QueryEngineBase.finalToInternalResult(queryResult), context };
    return (await this.actorInitQuery.mediatorQueryResultSerialize
      .mediate({ context, handle, handleMediaType: mediaType })).handle;
  }

  /**
   * Invalidate all internal caches related to the given page URL.
   * If no page URL is given, then all pages will be invalidated.
   * @param {string} url The page URL to invalidate.
   * @param context An optional ActionContext to pass to the actors.
   * @return {Promise<any>} A promise resolving when the caches have been invalidated.
   */
  public invalidateHttpCache(url?: string, context?: any): Promise<any> {
    context = ActionContext.ensureActionContext(context);
    return this.actorInitQuery.mediatorHttpInvalidate.mediate({ url, context });
  }

  public scale(data: number[], trainMin?: number, trainMax?: number){
    if (!trainMin){
      trainMin = Math.min(...data);
    }
    if(!trainMax){
      trainMax = Math.max(...data);
    }
    return data.map(x => (x-trainMin!)/(trainMax!-trainMin!));
  }

  public async featurizeTrainQueries( trainQueries: string[], trainCardinalities: number[],
    queriesContext: QueryStringContext
  ): Promise<IFeaturizedTrainQueries> {
    const rawFeaturesTrain: IQueryFeatures[] = await this.prepareFeaturizedQueries(trainQueries, queriesContext);
    // Prepare train features and input to optimizer
    const tpEncodingsTrain: tf.Tensor[][] = rawFeaturesTrain.map(x => x.leafFeatures);
    const standardizedTpEncTrain: tf.Tensor[][] = this.standardizeCardinalityTriplePattern(tpEncodingsTrain);
    const graphViewsTrain: IQueryGraphViews[] = rawFeaturesTrain.map(x => x.graphViews);

    // Take log to stabilize training
    const logTrainCardinalities: number[] = trainCardinalities.map(x => Math.log(x));
    // Flush all tensors used during training to prevent memory leaks
    // Flush raw feature tensors
    rawFeaturesTrain.map(x => x.leafFeatures.map(y => y.dispose()));
    // Flush tp encodings
    tpEncodingsTrain.map(x => x.map(y => y.dispose()));

    return {
      tpEncTrain: standardizedTpEncTrain,
      cardinalitiesTrain: logTrainCardinalities,
      graphViewsTrain: graphViewsTrain,
    }
  }

  public async featurizeValQueries( valCardinalities: number[], valQueries: string[], trainQueries: string[],
    queriesContext: QueryStringContext, trainMinQueryCardinality: number, trainMaxQueryCardinality: number,

  ): Promise<IFeaturizedValQueries>{

    const rawFeaturesTrain: IQueryFeatures[] = await this.prepareFeaturizedQueries(trainQueries, queriesContext);
    // Prepare train features and input to optimizer
    const tpEncodingsTrain: tf.Tensor[][] = rawFeaturesTrain.map(x => x.leafFeatures);
    const meanCardinilityTpTrain = this.getMeanCardinality(tpEncodingsTrain);
    const stdCardinalityTpTrain = this.getStdCardinality(tpEncodingsTrain, meanCardinilityTpTrain);

    const rawFeaturesVal: IQueryFeatures[] = await this.prepareFeaturizedQueries(valQueries, queriesContext);
    const tpEncodingsVal: tf.Tensor[][] = rawFeaturesVal.map(x => x.leafFeatures);
    // Std validation data with training mean / std
    const standardizedTpEncVal: tf.Tensor[][] = this.standardizeCardinalityTriplePattern(tpEncodingsVal, 
      meanCardinilityTpTrain, stdCardinalityTpTrain);

    const graphViewsVal: IQueryGraphViews[] = rawFeaturesVal.map(x => x.graphViews);
    // Prepare val features and input to optimizer
    const tensorTpEncVal = standardizedTpEncVal.map(x => tf.stack(x));
    // Take log to stabilize training
    const logValCardinalities: number[] = valCardinalities.map(x => Math.log(x+1));

    // Flush all tensors used during training to prevent memory leaks
    // Flush raw feature tensors
    rawFeaturesVal.map(x => x.leafFeatures.map(y => y.dispose()));
    // Flush tp encodings
    tpEncodingsVal.map(x => x.map(y => y.dispose()));
    // Flush std tp encodings
    standardizedTpEncVal.map(x => x.map(y => y.dispose()));

    return {
      tpEncVal: tensorTpEncVal,
      cardinalitiesVal: logValCardinalities,
      graphViewsVal: graphViewsVal
    }
  }

  // What can still be done:
  // 1. Isomorphism removal
  // 2. Onehot encode vectors
  // 3. Concat instead of average whne uinsg rdf2vec
  // Increase width first layer
  // ??
  public async pretrainOptimizer(
    trainQueries: string[],
    trainCardinalities: number[],
    valQueries: string[],
    valCardinalities: number[],
    queriesContext: QueryStringContext,
    batchSize: number,
    epochs: number,
    optimizer: string,
    lr: number,
    modelDirectory: string,
    chkpLocations: string,
    logLocation?: string,
    logValidationPredictions: boolean = false
  ) {
    // Create model to pretrain
    this.modelInstanceGCN = new InstanceModelGCN(modelDirectory);
    // Init GCN model from config
    this.modelInstanceGCN.initModel(modelDirectory, true);
    // Init model trainer with specified optimizer and learning rate
    this.modelTrainerOffline = new ModelTrainerOffline({ optimizer, learningRate: lr });

    // Cardinality estimation layer from "Cardinality estimation over knowledge graphs with embeddings and graph neural networks" 
    // by Tim Schwabe and use mean pooling + hidden layer with relu and linear layer
    const cardinalityPredictionLayers: IGraphCardinalityPredictionLayers = {
      pooling: tf.layers.average(),
      hiddenLayer: new DenseOwnImplementation(384, 256),
      activationHiddenLayer: tf.layers.activation({ activation: 'relu' }),
      finalLayer: new DenseOwnImplementation(256, 1)
    };

    const featurizedTrainQueries = await this.featurizeTrainQueries(
      trainQueries, trainCardinalities, queriesContext
    );

    const trainMin = Math.min(...trainCardinalities);
    const trainMax = Math.max(...trainCardinalities);
    const featurizedValQueries = await this.featurizeValQueries(
      valCardinalities, valQueries, trainQueries, queriesContext , trainMin, trainMax
    );

    const pretrainResult: IPretrainLog = {
      trainLoss: [], 
      validationError: []
    };

    for (let i = 0; i < epochs; i++) {
      const shuffled = this.shuffle(
        featurizedTrainQueries.tpEncTrain, 
        featurizedTrainQueries.cardinalitiesTrain, 
        featurizedTrainQueries.graphViewsTrain
      );
      const shuffledFeaturesAsTensors: tf.Tensor[] = featurizedTrainQueries.tpEncTrain.map(x => tf.stack(x));
      const trainOutput = this.batchedPretraining(
        shuffledFeaturesAsTensors,
        featurizedTrainQueries.cardinalitiesTrain,
        featurizedTrainQueries.graphViewsTrain,
        batchSize,
        this.modelInstanceGCN.getModels(),
        cardinalityPredictionLayers,
      );
      console.log(`Epoch ${i+1}/${epochs}, 
      trainLoss: ${trainOutput.loss.toFixed(2)}, 
      mean: ${trainOutput.averagePrediction.toFixed(2)}, 
      std: ${trainOutput.stdPrediction.toFixed(2)}`
      );

      const averageAbsoluteError = this.validateModel(
        valQueries,
        featurizedValQueries.tpEncVal,
        featurizedValQueries.cardinalitiesVal,
        featurizedValQueries.graphViewsVal,
        this.modelInstanceGCN.getModels(),
        cardinalityPredictionLayers,
        logValidationPredictions,
        logLocation? path.join(logLocation, `validationOutputEpoch${i+1}`) : undefined
      );

      console.log(`Validaton absolute error: ${averageAbsoluteError}`);
      pretrainResult.trainLoss.push(trainOutput.loss);
      pretrainResult.validationError.push(averageAbsoluteError);

      console.log(`Saving model.`);
      const ckpEpochLocation = path.join(chkpLocations, `ckp${i+1}`);
      if (!fs.existsSync(ckpEpochLocation)){
        fs.mkdirSync(ckpEpochLocation);      
      }
      this.saveModel(ckpEpochLocation, cardinalityPredictionLayers);
      // Flush shuffled features
      shuffledFeaturesAsTensors.map(x => x.dispose());
    }

    featurizedValQueries.tpEncVal.map(x => x.dispose());
    featurizedTrainQueries.tpEncTrain.map(x => x.map(y => y.dispose()));
    this.modelInstanceGCN.disposeModels();

    cardinalityPredictionLayers.pooling.dispose();
    cardinalityPredictionLayers.hiddenLayer.disposeLayer();
    cardinalityPredictionLayers.finalLayer.disposeLayer();
    return pretrainResult  
  }

  public async prepareFeaturizedQueries(queries: string[], context: QueryStringContext): Promise<IQueryFeatures[]> {
    const allQueryFeatures: IQueryFeatures[] = [];
    for (const query of queries) {
      const featuresQuery = await this.getFeaturesQuery(query, context);
      allQueryFeatures.push(featuresQuery);
    }
    return allQueryFeatures;
  }

  public batchedPretraining(
    features: tf.Tensor[],
    cardinalities: number[],
    graphViews: IQueryGraphViews[],
    batchSize: number,
    modelsGCN: IQueryGraphEncodingModels,
    cardinalityPredictionLayers: IGraphCardinalityPredictionLayers,
  ): ITrainResult {
    const numBatches = Math.ceil(features.length / batchSize);
    let totalLoss = 0;
    let totalMean = 0;
    let totalStd = 0;
    for (let b = 0; b < numBatches; b++) {
      const triplePatternEncodingBatch: tf.Tensor[] = features.slice(b * batchSize, Math.min((b + 1) * batchSize, features.length));
      const graphViewsInBatch: IQueryGraphViews[] = graphViews.slice(b * batchSize, Math.min((b + 1) * batchSize, graphViews.length));
      const queryCardinalitiesBatch: number[] = cardinalities.slice(b * batchSize, Math.min((b + 1) * batchSize, cardinalities.length));

      const loss = this.modelTrainerOffline.pretrainOptimizerBatched(
        queryCardinalitiesBatch,
        triplePatternEncodingBatch,
        graphViewsInBatch,
        modelsGCN,
        cardinalityPredictionLayers,
      );
      totalLoss += loss.loss;
      totalMean += loss.averagePrediction;
      totalStd += loss.stdPrediction;
    }
    return {
      loss: totalLoss / features.length, 
      averagePrediction: totalMean / numBatches, 
      stdPrediction: totalStd / numBatches
    };
  }

  public standardizeCardinalityTriplePattern(features: tf.Tensor[][], mean?: number, std?: number) {
    return tf.tidy(() => {
      // let totalCardinality = 0;
      // let totalFeaturizedPatterns = 0;
  
      // const cardinalitiesTriplePattern: number[] = [];
      const featuresAsArray: number[][][][] = [];
      // Calculate mean cardinality of triple patterns
      if (!mean){
        mean = this.getMeanCardinality(features);
      }
      if (!std){
        std = this.getStdCardinality(features, mean);
      }

      for (const feature of features) {
        const innerArray: number[][][] = [];
        for (const element of feature) {
          const tensorAsArray: number[][] = <number[][]> element.arraySync();
          innerArray.push(tensorAsArray);
        }
        featuresAsArray.push(innerArray);
      }
      // let totalStd = 0;
      // for (const feature of features) {
      //   for (const element of feature) {
      //     const tensorAsArray: number[][] = <number[][]> element.arraySync();
      //     totalStd += (tensorAsArray[0][0] - meanCardinality) ** 2;
      //   }
      // }
      // const standardDeviation = Math.sqrt(totalStd / (totalFeaturizedPatterns - 1));
      for (const element of featuresAsArray) {
        for (const element_ of element) {
          element_[0][0] = (element_[0][0] - mean) / std;
        }
      }
      // Convert back into tensors
      const standardizedFeatures = featuresAsArray.map(x => x.map(y => tf.tensor(y)));
      return standardizedFeatures;  
    });
  }

  private getStdCardinality(features: tf.Tensor[][], meanCardinality: number){
    let totalStd = 0;
    let totalFeaturizedPatterns = 0;

    for (const feature of features) {
      for (const element of feature) {
        const tensorAsArray: number[][] = <number[][]> element.arraySync();
        totalStd += (tensorAsArray[0][0] - meanCardinality) ** 2;
        totalFeaturizedPatterns += 1;
      }
    }
    const standardDeviation = Math.sqrt(totalStd / (totalFeaturizedPatterns - 1));
    return standardDeviation;
  }

  private getMeanCardinality(features: tf.Tensor[][]){
    let totalCardinality = 0;
    let totalFeaturizedPatterns = 0;
    const cardinalitiesTriplePattern: number[] = [];
    const featuresAsArray: number[][][][] = [];

    for (const feature of features) {
      const innerArray: number[][][] = [];
      for (const element of feature) {
        const tensorAsArray: number[][] = <number[][]> element.arraySync();

        cardinalitiesTriplePattern.push(tensorAsArray[0][0]);
        innerArray.push(tensorAsArray);
        totalCardinality += tensorAsArray[0][0];
        totalFeaturizedPatterns += 1;
      }
      featuresAsArray.push(innerArray);
    }
    const mean = totalCardinality / totalFeaturizedPatterns;
    return mean;
  }

  public validateModel(
    queries: string[],
    features: tf.Tensor[],
    cardinalities: number[],
    graphViews: IQueryGraphViews[],
    modelsGCN: IQueryGraphEncodingModels,
    cardinalityPredictionLayers: IGraphCardinalityPredictionLayers,
    logValidationPredictions: boolean,
    logLocation?: string
  ) {
    return tf.tidy(()=>{
      const cardPredictions = [];
      for (let i = 0; i < features.length; i++){
        const prediction = this.modelTrainerOffline.getCardinalityPrediction(
          features[i], 
          graphViews[i], 
          modelsGCN, 
          cardinalityPredictionLayers,
          true
        );
        cardPredictions.push(prediction)
      }
      const executionTimesBatch: tf.Tensor[] = cardinalities.map(x=>tf.tensor(x));
      const valErrorTensor = this.modelTrainerOffline.meanAbsoluteError( tf.concat(cardPredictions).squeeze(), tf.concat(executionTimesBatch).squeeze());
      if (logValidationPredictions && logLocation){
        const validationOutputs: IValidationResult[] = [];
        const predArray: any = <number[]> cardPredictions.map(x => x.arraySync()).flat();
        for (let i = 0; i < queries.length; i++){
          validationOutputs.push({query: queries[i], pred: predArray[i][0], actual: cardinalities[i]})
        }
        fs.writeFileSync(logLocation, JSON.stringify(validationOutputs));
      }
      const valError = tf.sum(valErrorTensor).squeeze();
      const valErrorNumber: number = <number> valError.arraySync();
      const averageValError = valErrorNumber / features.length;
      return averageValError;  
    });
  }

  public async validateTrainedModel(
    trainQueries: string[],
    trainCardinalities: number[],
    valQueries: string[],
    valCardinalities: number[],
    queriesContext: QueryStringContext,
    modelDirectory: string,
    logLocation: string,
  ){
    this.modelInstanceGCN = new InstanceModelGCN(modelDirectory);
    // Init GCN model from config
    this.modelInstanceGCN.initModel(modelDirectory, true);

    // Cardinality estimation layer from "Cardinality estimation over knowledge graphs with embeddings and graph neural networks" 
    // by Tim Schwabe and use mean pooling + hidden layer with relu and linear layer
    const cardinalityPredictionLayers: IGraphCardinalityPredictionLayers = {
      pooling: tf.layers.average(),
      hiddenLayer: new DenseOwnImplementation(384, 256),
      activationHiddenLayer: tf.layers.activation({ activation: 'relu' }),
      finalLayer: new DenseOwnImplementation(256, 1)
    };
    // Initialize the trained cardinality prediction head
    InstanceModelGCN.initializeCardinalityPredictionHeadFromWeights(modelDirectory, cardinalityPredictionLayers);
    const trainMin = Math.min(...trainCardinalities);
    const trainMax = Math.max(...trainCardinalities);
    const featurizedValQueries = await this.featurizeValQueries(
      valCardinalities, valQueries, trainQueries, queriesContext , trainMin, trainMax
    );
    console.log(featurizedValQueries.cardinalitiesVal)
    const averageAbsoluteError = this.validateModel(
      valQueries,
      featurizedValQueries.tpEncVal,
      featurizedValQueries.cardinalitiesVal,
      featurizedValQueries.graphViewsVal,
      this.modelInstanceGCN.getModels(),
      cardinalityPredictionLayers,
      true,
      logLocation? path.join(logLocation, `validationOutputTrainedModel`) : undefined
    );
    console.log(`Finished validating trained model, average absolute error: ${averageAbsoluteError}`)
  }

  public saveModel(location: string, cardinalityPredictionLayers: IGraphCardinalityPredictionLayers){
    this.modelInstanceGCN.createDirectoryStructureCheckpoint(location);
    this.modelInstanceGCN.saveModels(path.join(
      location,
      `gcn-models`
    ));

    InstanceModelGCN.saveCardinalityPredictionHead(
      path.join(location, `gcn-models`), 
      cardinalityPredictionLayers
    );
  }

  public shuffle(
    features: tf.Tensor[][],
    cardinalities: number[],
    graphViews: IQueryGraphViews[],
  ) {
    let i,
        j,
        xE,
        xG,
        xQ;
    for (i = features.length - 1; i > 0; i--) {
      j = Math.floor(Math.random() * (i + 1));
      xE = features[i]; xQ = cardinalities[i]; xG = graphViews[i];
      features[i] = features[j]; cardinalities[i] = cardinalities[j]; graphViews[i] = graphViews[j];
      features[j] = xE; cardinalities[j] = xQ; graphViews[j] = xG;
    }
    return [ features, cardinalities, graphViews ];
  }

  public async trainModel(batchedTrainingExamples: IBatchedTrainingExamples) {
    function shuffleData(executionTimes: number[], qValues: number[], partialJoinTree: number[][][]) {
      let i,
          j,
          xE,
          xP,
          xQ;
      for (i = executionTimes.length - 1; i > 0; i--) {
        j = Math.floor(Math.random() * (i + 1));
        xE = executionTimes[i]; xQ = qValues[i]; xP = partialJoinTree[i];
        executionTimes[i] = executionTimes[j]; qValues[i] = qValues[j]; partialJoinTree[i] = partialJoinTree[j];
        executionTimes[j] = xE; qValues[j] = xQ; partialJoinTree[j] = xP;
      }
      return [ qValues, executionTimes, partialJoinTree ];
    }

    if (batchedTrainingExamples.trainingExamples.size === 0) {
      throw new Error('Empty map passed');
    }
    const actualExecutionTime: number[] = [];
    const predictedQValues: number[] = [];
    const partialJoinTree: number[][][] = [];
    for (const [ joinKey, trainingExample ] of batchedTrainingExamples.trainingExamples.entries()) {
      actualExecutionTime.push(trainingExample.actualExecutionTime);
      predictedQValues.push(trainingExample.qValue);
      partialJoinTree.push(MediatorJoinReinforcementLearning.keyToIdx(joinKey));
    }
    const shuffled = shuffleData(actualExecutionTime, predictedQValues, partialJoinTree);
    const loss = this.modelTrainerOffline.trainOfflineBatched(partialJoinTree, batchedTrainingExamples.leafFeatures, actualExecutionTime, this.modelInstanceLSTM.getModel(), 4);
    return loss;
  }

  public disposeTrainEpisode() {
    this.trainEpisode.learnedFeatureTensor.hiddenStates.map(x => x.dispose());
    this.trainEpisode.learnedFeatureTensor.memoryCell.map(x => x.dispose());
  }

  public emptyTrainEpisode() {
    this.trainEpisode = {
      joinsMade: [],
      learnedFeatureTensor: { hiddenStates: [], memoryCell: []},
      sharedVariables: [],
      leafFeatureTensor: [],
      graphViews: { subSubView: [], objObjView: [], objSubView: []},
      isEmpty: true,
    };
  }

  /**
   *
   * @returns Empty binding stream
   */
  public returnNop() {
    const nop: IQueryOperationResult = {
      bindingsStream: new ArrayIterator([ this.BF.bindings() ], { autoStart: false }),
      metadata: () => Promise.resolve({
        cardinality: { type: 'exact', value: 1 },
        canContainUndefs: false,
        variables: [],
      }),
      type: 'bindings',
    };
    return QueryEngineBase.internalToFinalResult(nop);
  }

  /**
   * Convert an internal query result to a final one.
   * @param internalResult An intermediary query result.
   */
  public static internalToFinalResult(internalResult: IQueryOperationResult): QueryType {
    switch (internalResult.type) {
      case 'bindings':
        return {
          resultType: 'bindings',
          execute: async() => internalResult.bindingsStream,
          metadata: async() => <any> await internalResult.metadata(),
          context: internalResult.context,
        };
      case 'quads':
        return {
          resultType: 'quads',
          execute: async() => internalResult.quadStream,
          metadata: async() => <any> await internalResult.metadata(),
          context: internalResult.context,
        };
      case 'boolean':
        return {
          resultType: 'boolean',
          execute: async() => internalResult.execute(),
          context: internalResult.context,
        };
      case 'void':
        return {
          resultType: 'void',
          execute: async() => internalResult.execute(),
          context: internalResult.context,
        };
    }
  }

  /**
   * Convert a final query result to an internal one.
   * @param finalResult A final query result.
   */
  public static async finalToInternalResult(finalResult: RDF.Query<any>): Promise<IQueryOperationResult> {
    switch (finalResult.resultType) {
      case 'bindings':
        return {
          type: 'bindings',
          bindingsStream: <BindingsStream> await finalResult.execute(),
          metadata: async() => <any> await finalResult.metadata(),
        };
      case 'quads':
        return {
          type: 'quads',
          quadStream: <AsyncIterator<RDF.Quad>> await finalResult.execute(),
          metadata: async() => <any> await finalResult.metadata(),
        };
      case 'boolean':
        return {
          type: 'boolean',
          execute: () => finalResult.execute(),
        };
      case 'void':
        return {
          type: 'void',
          execute: () => finalResult.execute(),
        };
    }
  }
}
export interface IExperience{
  actualExecutionTimeNorm: number;
  actualExecutionTimeRaw: number;
  joinIndexes: number[][];
  N: number;
}

export interface IQueryFeatures{
  leafFeatures: tf.Tensor[];
  graphViews: IQueryGraphViews;
}

export interface IPretrainLog{
  trainLoss: number[]
  validationError: number[]
}

export interface IValidationResult{
  query: string
  pred: number
  actual: number
}

export interface IFeaturizedQueries{
  tpEncTrain: tf.Tensor[][]
  cardinalitiesTrain: number[]
  graphViewsTrain: IQueryGraphViews[]
  tpEncVal: tf.Tensor[]
  cardinalitiesVal: number[]
  graphViewsVal: IQueryGraphViews[]
}

export interface IFeaturizedTrainQueries{
  tpEncTrain: tf.Tensor[][]
  cardinalitiesTrain: number[]
  graphViewsTrain: IQueryGraphViews[]
}

export interface IFeaturizedValQueries{
  tpEncVal: tf.Tensor[]
  cardinalitiesVal: number[]
  graphViewsVal: IQueryGraphViews[]
}