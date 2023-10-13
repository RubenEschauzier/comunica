import { materializeOperation } from '@comunica/bus-query-operation';
import type { IActionSparqlSerialize, IActorQueryResultSerializeOutput } from '@comunica/bus-query-result-serialize';
import { KeysCore, KeysInitQuery, KeysRdfResolveQuadPattern, KeysRlTrain } from '@comunica/context-entries';
import { ActionContext } from '@comunica/core';
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
  ITrainingExample
} from '@comunica/types';
import type * as RDF from '@rdfjs/types';
import { ArrayIterator, AsyncIterator, SingletonIterator } from 'asynciterator';
import type { Algebra } from 'sparqlalgebrajs';
import type { ActorInitQueryBase } from './ActorInitQueryBase';
import { MemoryPhysicalQueryPlanLogger } from './MemoryPhysicalQueryPlanLogger';
import { IAggregateValues, IQueryGraphViews, IResultSetRepresentation, IResultSetRepresentationGraph, IRunningMoments, MediatorJoinReinforcementLearning } from '@comunica/mediator-join-reinforcement-learning';
import { InstanceModelGCN, InstanceModelLSTM, ModelTrainerOffline } from '@comunica/actor-rdf-join-inner-multi-reinforcement-learning-tree';
import * as chalk from 'chalk';
import { ExperienceBuffer, IExperienceKey } from '@comunica/model-trainer';
import * as _ from 'lodash';
import * as fs from 'graceful-fs';
import * as tf from '@tensorflow/tfjs-node';
import { BindingsFactory } from '@comunica/bindings-factory';

import * as process from 'process';
import * as path from 'path'
import { readFileSync, writeFileSync } from 'fs';

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

  // Values needed for endpoint training
  public experienceBuffer: ExperienceBuffer;
  public runningMomentsFeatures: IRunningMoments;
  public runningMomentsExecution: IRunningMoments;
  public trainEpisode: ITrainEpisode;
  public nExpPerQuery: number;
  public BF: BindingsFactory;
  public breakRecursion: boolean;
  public currentQueryKey: string;
  public currentTrainingStateEngineDirectory: string;
  // Values for epoch checkpointing
  public numTrainSteps: number;
  public numQueries: number;
  public totalEpochsDone: number;



  public constructor(actorInitQuery: ActorInitQueryBase<QueryContext>) {
    this.actorInitQuery = actorInitQuery;
    this.defaultFunctionArgumentsCache = {};
    // Initialise empty episode on creation
    this.emptyTrainEpisode();
    // Instantiate used models
    this.modelInstanceLSTM = new InstanceModelLSTM(this.actorInitQuery.modelConfigLocationLstm);
    this.modelInstanceGCN = new InstanceModelGCN(this.actorInitQuery.modelConfigLocationGcn);
    this.currentTrainingStateEngineDirectory = "latestSavedState";
    // Moved this to first execution of query, because we want to initiate this from config, honestly should work more with components.js for this
    this.modelTrainerOffline = new ModelTrainerOffline({optimizer: 'adam', learningRate: 0.0001});
    this.batchedTrainExamples = {trainingExamples: new Map<string, ITrainingExample>, leafFeatures: {hiddenStates: [], memoryCell:[]}};
    // End point training
    this.BF = new BindingsFactory();
    this.breakRecursion = false;
    // Initialise empty running moments
    this.runningMomentsFeatures = {indexes: [0], runningStats: new Map<number, IAggregateValues>()};
    this.runningMomentsExecution = {indexes: [0], runningStats: new Map<number, IAggregateValues>()};
    for (const index of this.runningMomentsExecution.indexes){
      const startPoint: IAggregateValues = {N: 0, mean: 0, std: 1, M2: 1}
      this.runningMomentsExecution.runningStats.set(index, startPoint);
    }
    for (const index of this.runningMomentsFeatures.indexes){
      const startPoint: IAggregateValues = {N: 0, mean: 0, std: 1, M2: 1}
      this.runningMomentsFeatures.runningStats.set(index, startPoint);
    }

    // Some hardcoded magic numbers for training
    this.experienceBuffer = new ExperienceBuffer(1500);
    this.nExpPerQuery = 8;
    this.numTrainSteps = 0;
    this.numQueries = 120;
    this.totalEpochsDone = 0;

  }

  public async querySingleTrainStep<QueryFormatTypeInner extends QueryFormatType>(
    query: QueryFormatTypeInner,
    context: QueryContext & QueryFormatTypeInner extends string ? QueryStringContext : QueryAlgebraContext,
    randomId?: number
  ){
    let trainLoss: number[] = [];
    try{
      trainLoss = this.readTrainLoss(this.actorInitQuery.modelCheckPointLocation+'/trainLoss.txt');
    }
    catch{
      console.warn(chalk.red("No train loss array found"));
    }

    const clonedContext = _.cloneDeep(context);
    clonedContext.batchedTrainingExamples = this.batchedTrainExamples;

    const queryKey = this.experienceBuffer.getQueryKey(query.toString());
    this.currentQueryKey = queryKey;

    const trainResult = await this.executeTrainQuery(query, clonedContext, queryKey, this.experienceBuffer, randomId);
    const experiences: IExperience[] = [];
    const features = []; 
    // Sample experiences from the buffer if we have enough prior executions
    if (this.experienceBuffer.getSize()>20){
      for (let z=0;z<this.nExpPerQuery;z++){
          const experience: [IExperience, IExperienceKey] = this.experienceBuffer.getRandomExperience();
          experiences.push(experience[0]);
          const feature = this.experienceBuffer.getFeatures(experience[1].query)!;
          features.push(feature);
      }
      // const loss = await this.trainModelExperienceReplay(experiences, features);
      const featureTensors = features.map(x=>tf.stack(x.hiddenStates));
      const loss = await this.trainModelExperienceReplayTemp(experiences, featureTensors, features.map(x=>x.graphViews));

      trainLoss.push(loss);
      this.numTrainSteps += 1;
    }
    
    this.writeTrainLoss(trainLoss, this.actorInitQuery.modelCheckPointLocation+'/trainLoss.txt');
    this.cleanBatchTrainingExamples();
    // When the number of train steps is equal or larger to the number of queries, we count 1 epoch, record statistics and checkpoint the model
    if (this.numTrainSteps >= this.numQueries){
      this.totalEpochsDone += 1;
      
      const trainLossEpoch: number[] = trainLoss.slice((this.totalEpochsDone-1)*this.numQueries, this.totalEpochsDone*this.numQueries);
      const avgTrainLossEpoch: number = trainLossEpoch.reduce((a,b)=>a+b,0)/trainLossEpoch.length||0; 
      console.log(`Epoch ${this.totalEpochsDone}: Avg train loss: ${avgTrainLossEpoch}`);
      this.saveState(0, `ckp${this.totalEpochsDone}`);
      let epochTrainLoss: number[] = []
      try{
        // Read epoch loss from previous chkpoint to append to
        epochTrainLoss = this.readTrainLoss(this.actorInitQuery.modelCheckPointLocation+"/ckp"+(this.totalEpochsDone-1)+"/epochTrainLoss.txt")
      }
      catch{
        console.warn("Couldn't find epoch train loss information");
      }
      epochTrainLoss.push(avgTrainLossEpoch);
      this.writeTrainLoss(epochTrainLoss, this.actorInitQuery.modelCheckPointLocation + "/ckp" + this.totalEpochsDone + "/epochTrainLoss.txt");
      // Reset the query execution counter
      this.numTrainSteps = 0;
    }
    return trainResult[1];
  }

  public async loadState(ckpDirName: string){
    // Load running moments
    try{
      const runningMomentsFeatures = this.modelInstanceLSTM.loadRunningMoments(
        path.join(
          this.actorInitQuery.modelCheckPointLocation, 
          ckpDirName, 
          'runningMomentsFeatures.txt'));
      const runningMomentsExecution = this.modelInstanceLSTM.loadRunningMoments(
        path.join(
          this.actorInitQuery.modelCheckPointLocation, 
          ckpDirName, 
          'runningMomentsExecution.txt'));
      this.runningMomentsFeatures = runningMomentsFeatures;
      this.runningMomentsExecution = runningMomentsExecution;
    }
    catch{
      console.warn(chalk.red("INFO: No running moments found"));
    }

    // Load training statistics
    try{
      const data = JSON.parse(readFileSync(
        path.join(
          this.actorInitQuery.modelCheckPointLocation,
          ckpDirName,
          'numTrainStepsEpochs.txt'
        ), 
        'utf-8'));
      this.numTrainSteps = +data[0];
      this.totalEpochsDone = +data[1];
    }
    catch{
      console.warn(chalk.red("INFO: Failed to read number of training steps executed"));
    }
    // Load experience buffer
    try{
      this.experienceBuffer.loadBuffer(
        path.join(
          this.actorInitQuery.modelCheckPointLocation,
          ckpDirName
        ));
    }
    catch(err){
      // If we can't find a certain file, we initialise an empty buffer
      this.experienceBuffer = new ExperienceBuffer(this.experienceBuffer.maxSize);
      console.warn(chalk.red("INFO: No existing or incomplete buffer found"));
    }
    // Load LSTM model weights
    try{
      await this.modelInstanceLSTM.initModel(path.join(
        this.actorInitQuery.modelCheckPointLocation,
        ckpDirName, 
        `models`, 
        `lstm-model`));
      console.log("Successfully initialised LSTM from weight files.");
    }
    catch(err){
      // Flush any initialisation that went through to start from fresh model
      this.modelInstanceLSTM.flushModel()
      // Initialise random model
      await this.modelInstanceLSTM.initModelRandom(
        path.join(
          this.actorInitQuery.modelCheckPointLocation,
          ckpDirName, 
          `models`, 
          `lstm-model`));
      console.warn(chalk.red("INFO: Failed to load model weights, randomly initialising model weights"));
    }
    // Load GCN model weights, this shouldn't fail so no error handling
    this.modelInstanceGCN.initModel(
      path.join(
      this.actorInitQuery.modelCheckPointLocation,
      ckpDirName, 
      `models`, 
      `gcn-models`
    ));
    console.log(`Successfully loaded experience buffer. Size Buffer after Loading: ${this.experienceBuffer.getSize()}`);
  }

  public saveState(timeOutValue: number, ckpDirName: string){
    this.createDirectoryStructureCheckpoint(ckpDirName);
    try{
      this.modelInstanceLSTM.saveModel(path.join(
        this.actorInitQuery.modelCheckPointLocation,
        ckpDirName,
        `models`,
        `lstm-model`
        ));
    }
    catch{
      console.warn(chalk.red("INFO: Failed to save LSTM model"));
    }
    try{
      this.modelInstanceGCN.saveModel(path.join(
        this.actorInitQuery.modelCheckPointLocation,
        ckpDirName,
        `models`,
        `gcn-models`
        ));
    }
    catch(err:any){
      console.warn(chalk.red(`INFO: Failed to save GCN model: error: ${err.message}`))
    }
    try{
      writeFileSync(path.join(this.actorInitQuery.modelCheckPointLocation,
        ckpDirName,
        'numTrainStepsEpochs.txt'
      ), 
      JSON.stringify([this.numTrainSteps, this.totalEpochsDone]));
    }
    catch{
      console.warn(chalk.red("INFO: Failed to save number of training steps executed"))
    }

    try{
      this.modelInstanceLSTM.saveRunningMoments(this.runningMomentsFeatures, 
        path.join(this.actorInitQuery.modelCheckPointLocation,
          ckpDirName,
          '/runningMomentsFeatures.txt'));
      this.modelInstanceLSTM.saveRunningMoments(this.runningMomentsExecution, 
        path.join(this.actorInitQuery.modelCheckPointLocation,
          ckpDirName,
          '/runningMomentsExecution.txt')
        );  
    }
    catch{
      console.warn(chalk.red("INFO: Failed to save running moments"));
    }

    // First add timedout experience to buffer with current execution time
    if (this.trainEpisode.joinsMade.length==0){
      // We have no joins in our episode, so we just save buffer as is and exit
      this.experienceBuffer.saveBuffer(
        path.join(
        this.actorInitQuery.modelCheckPointLocation,
        ckpDirName
      ));
      console.warn(chalk.red("Empty trainEpisode when saving training state"));
      return;
    }
    // If we have episode information we save it to buffer and then exit
    const elapsed = timeOutValue / 1000;
    const statsY: IAggregateValues = this.runningMomentsExecution.runningStats.get(this.runningMomentsExecution.indexes[0])!;

    this.updateRunningMoments(statsY, elapsed);
    
    const standardisedElapsed = (elapsed - statsY.mean) / statsY.std;
    let partialJoinPlan: number[][] = [];

    for (const joinMade of this.trainEpisode.joinsMade){
      partialJoinPlan.push(joinMade);
      const newExperience: IExperience = {actualExecutionTimeRaw: elapsed, actualExecutionTimeNorm: standardisedElapsed, 
        joinIndexes: [...partialJoinPlan], N:1};
        this.experienceBuffer.setExperience(this.currentQueryKey, MediatorJoinReinforcementLearning.idxToKey([...partialJoinPlan]), newExperience, statsY);
    }
    this.experienceBuffer.saveBuffer(
      path.join(
      this.actorInitQuery.modelCheckPointLocation,
      ckpDirName
    ));
  }
  /**
   * Creates directory structure of chkpoint at specified save location. Structure is as follows:
   * Ckp${i}
   * |- traininginfo...
   * |- models
   *    |- gcn-models
   *      |- subj-subj...
   *      |- obj-obj...
   *      |- obj-sub...
   *    |- lstm-model
   *      |- ...
   */
  public createDirectoryStructureCheckpoint(ckpDirName: string){
    console.log("QEB Create Dir structure")
    if (!fs.existsSync(path.join(this.actorInitQuery.modelCheckPointLocation, ckpDirName))){
      fs.mkdirSync(path.join(this.actorInitQuery.modelCheckPointLocation, ckpDirName));
    }
    const ckpDir = path.join(this.actorInitQuery.modelCheckPointLocation, ckpDirName);
    if (!fs.existsSync(path.join(ckpDir, "models"))){
      fs.mkdirSync(path.join(ckpDir, "models"));
    }
    const modelDir = path.join(ckpDir, "models");
    if (!fs.existsSync(path.join(modelDir, "lstm-model"))){
      fs.mkdirSync(path.join(modelDir, "lstm-model"));
    }
    if (!fs.existsSync(path.join(modelDir, "gcn-models"))){
      fs.mkdirSync(path.join(modelDir, "gcn-models"));
    }
    if(!fs.existsSync(path.join(modelDir, "gcn-models", "gcn-model-subj-subj"))){
      fs.mkdirSync(path.join(modelDir, "gcn-models", "gcn-model-subj-subj"));
    }
    if(!fs.existsSync(path.join(modelDir, "gcn-models", "gcn-model-obj-obj"))){
      fs.mkdirSync(path.join(modelDir, "gcn-models", "gcn-model-obj-obj"));
    }
    if(!fs.existsSync(path.join(modelDir, "gcn-models", "gcn-model-obj-subj"))){
      fs.mkdirSync(path.join(modelDir, "gcn-models", "gcn-model-obj-subj"));
    }
    console.log("QEB Create Dir structure DONE")

  }

  public readTrainLoss(fileLocation: string){
    const loss: number[] = JSON.parse(fs.readFileSync(fileLocation, 'utf-8'));
    return loss;
  }

  public writeTrainLoss(loss: number[], fileLocation: string){
    fs.writeFileSync(fileLocation, JSON.stringify(loss));
  }

  public async executeTrainQuery<QueryFormatTypeInner extends QueryFormatType>(
    query: QueryFormatTypeInner, 
    context: QueryContext & QueryFormatTypeInner extends string ? QueryStringContext : QueryAlgebraContext,
    queryKey: string,
    experienceBuffer: ExperienceBuffer,
    randomId?: number
    ):Promise<[number, BindingsStream]>{
    const startT = this.getTimeSeconds();
    // Let the RL model optimize the query
    const binding = await this.queryOfType<QueryFormatTypeInner, IQueryBindingsEnhanced>(query, context, 'bindings', randomId);
    const endTSearch = this.getTimeSeconds();
    
    // If there are no joins in query we do not record the experience for the model to train
    if(this.trainEpisode.joinsMade.length==0){
      this.disposeTrainEpisode();
      this.emptyTrainEpisode();
      return [endTSearch - startT, binding];
    }

    // Get all query optimization steps made by model
    const joinOrderKeys: string[] = [];
    for (let i = 1; i <this.trainEpisode.joinsMade.length+1;i++){
        const joinIndexId = this.trainEpisode.joinsMade.slice(0, i);
        joinOrderKeys.push(MediatorJoinReinforcementLearning.idxToKey(joinIndexId));
    }

    // Consume the query plan to obtain query execution time
    const elapsed = await this.consumeStream(binding, experienceBuffer, startT, joinOrderKeys, queryKey, false, true);

    // Clear episode tensors to reset the model state
    this.disposeTrainEpisode();
    this.emptyTrainEpisode();
    return [endTSearch-startT, binding];
  }

  public consumeStream(bindingStream: BindingsStream, 
    experienceBuffer: ExperienceBuffer,
    startTime: number, joinsMadeEpisode: string[], queryKey: string, 
    validation: boolean, recordExperience: boolean){
    let numBindings = 0;
    const consumedStream: Promise<number> = new Promise((resolve, reject)=>{
      bindingStream.on('data', () =>{
        numBindings+=1;
      });
      bindingStream.on('end', () => {
        const endTime: number = this.getTimeSeconds();
        const elapsed: number = endTime-startTime;
        const statsY: IAggregateValues = this.runningMomentsExecution.runningStats.get(this.runningMomentsExecution.indexes[0])!;

        if (!validation){
          this.updateRunningMoments(statsY, elapsed);
        }
        if (recordExperience){
          const standardisedElapsed = (elapsed - statsY.mean) / statsY.std;
          for (const joinMade of joinsMadeEpisode){
            const newExperience: IExperience = {actualExecutionTimeRaw: elapsed, actualExecutionTimeNorm: standardisedElapsed, 
              joinIndexes: MediatorJoinReinforcementLearning.keyToIdx(joinMade), N:1};
              experienceBuffer.setExperience(queryKey, joinMade, newExperience, statsY);
            }
        }
        resolve(elapsed)
      })
    });
    return consumedStream;
  }

  public async executeQueryInitFeaturesBuffer<QueryFormatTypeInner extends QueryFormatType>(
     query: QueryFormatTypeInner,
     context: QueryContext & QueryFormatTypeInner extends string ? QueryStringContext : QueryAlgebraContext, 
    ){
    const bindingsStream: BindingsStream = await this.queryBindings(query, context);
    const leafFeatures: IResultSetRepresentation = {hiddenStates: this.batchedTrainExamples.leafFeatures.hiddenStates.map(x=>x.clone()),
    memoryCell: this.batchedTrainExamples.leafFeatures.memoryCell.map(x=>x.clone())};
    this.disposeTrainEpisode();
    this.emptyTrainEpisode();
    this.cleanBatchTrainingExamples();    
    return leafFeatures;
}

  public updateRunningMoments(toUpdateAggregate: IAggregateValues, newValue: number){
    toUpdateAggregate.N +=1;
    const delta = newValue - toUpdateAggregate.mean; 
    toUpdateAggregate.mean += delta / toUpdateAggregate.N;
    const newDelta = newValue - toUpdateAggregate.mean;
    toUpdateAggregate.M2 += delta * newDelta;
    toUpdateAggregate.std = Math.sqrt(toUpdateAggregate.M2 / toUpdateAggregate.N);
  }

  public cleanBatchTrainingExamples(){
    this.batchedTrainExamples.leafFeatures.hiddenStates.map(x =>x.dispose());
    this.batchedTrainExamples.leafFeatures.memoryCell.map(x=>x.dispose());
    this.batchedTrainExamples = {trainingExamples: new Map<string, ITrainingExample>, leafFeatures: {hiddenStates: [], memoryCell:[]}};
  }

  public stdArray(values: number[], mean: number){
    const std: number = Math.sqrt((values.map(x=>(x-mean)**2).reduce((a, b)=>a+b,0) / values.length));
    return std
  }

  public getTimeSeconds(){
    const hrTime: number[] = process.hrtime();
    const time: number = hrTime[0] + hrTime[1] / 1000000000;
    return time
  }

  public writeEpochFiles(fileLocations: string[], epochInformation: number[][]){
    for (let i=0;i<fileLocations.length;i++){
        fs.writeFileSync(fileLocations[i], JSON.stringify([...epochInformation[i]]));
    }
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
    randomId?: number
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
    randomId?: number
  ): Promise<QueryType> {
    const output = await this.queryOrExplain(query, context, randomId);
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
    randomId?: number
  ): Promise<QueryType | IQueryExplained> {
    context = context || <any>{};
    /**
     * If there are no entries in queryToKey map we are at first run or after time-out, thus reload the model state.
     * We do this at start query to prevent unnecessary initialisation of features
    */    
    // TEMP CODEEE REMOVE THIS
    if (!randomId){
      randomId = Math.floor(Math.random() * (Number.MAX_SAFE_INTEGER));
    }
    console.log(`Starting query with randomId ${randomId}`);
    if (context && this.experienceBuffer.queryToKey.size==0 && context.trainEndPoint){
      console.log("Here then?");
      await this.loadState(this.currentTrainingStateEngineDirectory);
    }

    // Flag to determine if we should perform an initialisation query
    const initQuery = this.experienceBuffer.queryExists(query.toString())? false : true;
    // Prepare batchedTrainExamples so we can store our leaf features
    if (context){
      context.batchedTrainingExamples = this.batchedTrainExamples;
    }
    // We should really use the query log in experiencebuffer to determine the # of triple patterns in query, 
    // so we skip training to train the model on those queries.

    if (context && context.trainEndPoint){
      context.train = true;
    }
    // If we're not doing init query, then we should check if the query should be used for training
    // Queries with <3 triple patterns are useless to train on
    let moreThanTwoTp = true;
    if (!initQuery){
      const queryFeatures = this.experienceBuffer.getFeatures(query.toString())!;
      if (queryFeatures.hiddenStates.length < 3){
        moreThanTwoTp = false;
      }
    }
    console.log(`Query: ${query}`)
    console.log(moreThanTwoTp);
    /**
     * We execute training step on endpoint if we
     * 1. Get a context
     * 2. The context says we are doing endpoint training
     * 3. If we haven't already called this.querySingleTrainStep (To prevent infinite recursion)
     * 4. If the query already is initialised in the experienceBuffer
     * 5. The query has more than two triple patterns (Nothing to train on otherwise)
     */
    if (context?.trainEndPoint && !this.breakRecursion && !initQuery && moreThanTwoTp){
      console.log("Start training query")
      // Execute query step with given query
      this.breakRecursion = true;
      await this.querySingleTrainStep(query, context, randomId);
      this.breakRecursion = false;
      context.batchedTrainingExamples = this.batchedTrainExamples;
      // Reset train episode
      this.emptyTrainEpisode();
      console.log("End training query, returning nop")
      return this.returnNop()
    }
    console.log('1')
    // Expand shortcuts
    for (const key in context) {
      if (this.actorInitQuery.contextKeyShortcuts[key]) {
        context[this.actorInitQuery.contextKeyShortcuts[key]] = context[key];
        delete context[key];
      }
    }
    console.log('2 expanded shortcuts')
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
    console.log('prepared context')
    // Initialise the running moments from file (The running moments file should only be passed on first execution)
    if(actionContext.get(KeysRlTrain.runningMomentsFeatureFile)){
      console.log('init running moments')
      if (this.modelInstanceLSTM.initMoments){
        console.warn(chalk.red("WARNING: Reloading Moments When They Are Already Initialised"));
      }

      const loadedRunningMomentsFeature: IRunningMoments = 
      this.modelInstanceLSTM.loadRunningMoments(actionContext.get(KeysRlTrain.runningMomentsFeatureFile)!);
      this.runningMomentsFeatures = loadedRunningMomentsFeature;
      // Signal we loaded the moments
      this.modelInstanceLSTM.initMoments=true;
    }

    // If no path was passed and the moments are not initialised we create mean 0 std 1 moments
    else if (!this.modelInstanceLSTM.initMoments){
      console.log('init running moments wihtout init')
      console.warn(chalk.red("INFO: Running Query Without Initialised Running Moments"));
      for (const index of this.runningMomentsFeatures.indexes){
        const startPoint: IAggregateValues = {N: 0, mean: 0, std: 1, M2: 1}
        this.runningMomentsFeatures.runningStats.set(index, startPoint);
      }
      this.modelInstanceLSTM.initMoments=true;
    }

    const baseIRI: string | undefined = actionContext.get(KeysInitQuery.baseIRI);
    // Why are we doing this?
    // await this.modelInstanceLSTM.initModelRandom();

    actionContext = actionContext
      .setDefault(KeysInitQuery.queryTimestamp, new Date())
      .setDefault(KeysRdfResolveQuadPattern.sourceIds, new Map())
      // Set the default logger if none is provided
      .setDefault(KeysCore.log, this.actorInitQuery.logger)
      .setDefault(KeysInitQuery.functionArgumentsCache, this.defaultFunctionArgumentsCache)
      .setDefault(KeysRlTrain.trainEpisode, this.trainEpisode)
      .setDefault(KeysRlTrain.modelInstanceLSTM, this.modelInstanceLSTM)
      .setDefault(KeysRlTrain.modelInstanceGCN, this.modelInstanceGCN)
      .setDefault(KeysRlTrain.runningMomentsFeatures, this.runningMomentsFeatures);

    // Pre-processing the context
    actionContext = (await this.actorInitQuery.mediatorContextPreprocess.mediate({ context: actionContext })).context;
    // Determine explain mode
    const explainMode: QueryExplainMode = actionContext.get(KeysInitQuery.explain)!;
    console.log("Parsing query")
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
    console.log("Apply intial bindings")
    // Apply initial bindings in context
    if (actionContext.has(KeysInitQuery.initialBindings)) {
      operation = materializeOperation(operation, actionContext.get(KeysInitQuery.initialBindings)!);

      // Delete the query string from the context, since our initial query might have changed
      actionContext = actionContext.delete(KeysInitQuery.queryString);
    }
    console.log("Optimize query op")
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
    console.log("Execute query")
    // Execute query
    const output = await this.actorInitQuery.mediatorQueryOperation.mediate({
      context: actionContext,
      operation,
    });
    console.log("Finish query")
    output.context = actionContext;
    // If we are in initialisation run, we initialise the query with its features and return empty bindings


    // Reset train episode if we are not training (Bit of a shoddy workaround, because we need some episode information 
    // If we train
    if (!actionContext.get(KeysRlTrain.train) && context && !context.trainEndPoint){
      this.emptyTrainEpisode();
    }
    const finalOutput = QueryEngineBase.internalToFinalResult(output);

    if (initQuery && moreThanTwoTp){
      // Get the leaf features resulting from query
      const leafFeatures: IResultSetRepresentation = {hiddenStates: this.batchedTrainExamples.leafFeatures.hiddenStates.map(x=>x.clone()),
        memoryCell: this.batchedTrainExamples.leafFeatures.memoryCell.map(x=>x.clone())};
      const leafFeaturesExpanded: IResultSetRepresentationGraph = {...leafFeatures, graphViews: this.trainEpisode.graphViews};
      // Clean the examples
      this.cleanBatchTrainingExamples();
      this.disposeTrainEpisode();
      this.emptyTrainEpisode();
      this.experienceBuffer.initQueryInformation(query.toString(), leafFeaturesExpanded);
      console.log("Finish initialization query")   
      return finalOutput;
    }
    // Output physical query plan after query exec if needed
    console.log(`Ending query, random-id: ${randomId}`);
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

  public async trainModel(batchedTrainingExamples: IBatchedTrainingExamples){
    function shuffleData(executionTimes: number[], qValues: number[], partialJoinTree: number[][][]){
      let i, j, xE, xQ, xP;
      for (i=executionTimes.length-1;i>0;i--){
        j = Math.floor(Math.random()*(i+1));
        xE = executionTimes[i]; xQ = qValues[i]; xP = partialJoinTree[i];
        executionTimes[i] = executionTimes[j]; qValues[i]=qValues[j]; partialJoinTree[i]=partialJoinTree[j];
        executionTimes[j] = xE; qValues[j]= xQ; partialJoinTree[j] = xP;
      }
      return [qValues, executionTimes, partialJoinTree]
    }

    if (batchedTrainingExamples.trainingExamples.size==0){
      throw new Error("Empty map passed");
    }
    const actualExecutionTime: number[] = [];
    const predictedQValues: number[] = [];
    const partialJoinTree: number[][][] = [];
    for (const [joinKey, trainingExample] of batchedTrainingExamples.trainingExamples.entries()){
      actualExecutionTime.push(trainingExample.actualExecutionTime);
      predictedQValues.push(trainingExample.qValue);
      partialJoinTree.push(MediatorJoinReinforcementLearning.keyToIdx(joinKey));
    }
    const shuffled = shuffleData(actualExecutionTime, predictedQValues, partialJoinTree);
    const loss = this.modelTrainerOffline.trainOfflineBatched(partialJoinTree, batchedTrainingExamples.leafFeatures, 
      actualExecutionTime, this.modelInstanceLSTM.getModel(), 4);
    return loss;

    /**
     * TODO For Results:
     * 1. Implement train loss  (done)
     * 1.5 Fix memory leaks :/ (done)
     * 2. Normalise features / y (done)
     * 3. Track Statistics (+ STD) (done)
     * 4. Create new vectors for watdiv!! (done)
     * 5. Regularization using dropout (done)
     * NOTE: We use dropout only in the forwardpass recursive, this is because we only train on the forwardpass recursive. This might lead to mismatch between 
     * training performance and actual query execution time?? Other hand it does make sense since we want the best possible model to execute joins 
     * even during training
     * 5. Query Encoding Implemented (todo use PCA with #PC equal to minimal number of joins to consider using multi join = 3)
     * (Optional for workshop):
     * 1. Experience Replay (https://paperswithcode.com/method/experience-replay) 
     * (https://towardsdatascience.com/a-technical-introduction-to-experience-replay-for-off-policy-deep-reinforcement-learning-9812bc920a96)
     * 2. Temperature scaling for bolzman softmax
     * 3. Better memory management
     * 
     * Considerations:
     * When we use features that 'mean' something in leaf result sets and then let the model generate feature representations for
     * joins, don't we lose information of what the features mean in the result set leafs
     * On one hand, the model can learn what what representations it should make to get the best results
     * On other hand, it seems counter intuitive?
     * 
     * Graph Encoding:
     * https://towardsdatascience.com/graph-representation-learning-network-embeddings-d1162625c52b
     * Hyperbolic embeddings
     * PCA on adjacency matrix
     * 
     * Triple pattern encoding:
     * Replace RDF2Vec with BERTesque model
     * This way we can learn contextual encodings of named nodes AND instantly have access to a [MASK] token embedding which can represent variables in a triple pattern
     * We can consider training the BERT model together with the optimizer or pretrain BERT only.
     * 
     * Lit review:
     * Comprehensive review of join order optimizers
     * https://www.vldb.org/pvldb/vol15/p1658-zhao.pdf 
     * Super cool paper on query performance prediction
     * https://www.vldb.org/pvldb/vol12/p1733-marcus.pdf
     */

  }

  public async trainModelExperienceReplay(experiences: IExperience[], leafFeatures: IResultSetRepresentation[]){
    if (experiences.length==0){
      throw new Error("Empty training batch passed");
    }
    const actualExecutionTimes: number[] = [];
    const partialJoinTree: number[][][] = [];
    for (const exp of experiences){
      actualExecutionTimes.push(exp.actualExecutionTimeNorm);
      partialJoinTree.push(exp.joinIndexes)
    }
    const loss = this.modelTrainerOffline.trainOfflineExperienceReplay(partialJoinTree, leafFeatures, 
      actualExecutionTimes, this.modelInstanceLSTM.getModel(), 4);
    return loss;
  }

  public async trainModelExperienceReplayTemp(experiences: IExperience[], leafFeatures: tf.Tensor[], graphViews: IQueryGraphViews[]){
    if (experiences.length==0){
      throw new Error("Empty training batch passed");
    }
    const actualExecutionTimes: number[] = [];
    const partialJoinTree: number[][][] = [];
    for (const exp of experiences){
      actualExecutionTimes.push(exp.actualExecutionTimeNorm);
      partialJoinTree.push(exp.joinIndexes)
    }
    const loss = this.modelTrainerOffline.trainOfflineExperienceReplayTemp(partialJoinTree, leafFeatures, 
      graphViews, actualExecutionTimes, this.modelInstanceLSTM.getModel(), this.modelInstanceGCN.getModels(), 4);
    return loss;
  }


  public disposeTrainEpisode(){
    this.trainEpisode.learnedFeatureTensor.hiddenStates.map(x=>x.dispose());
    this.trainEpisode.learnedFeatureTensor.memoryCell.map(x=>x.dispose()); 
  }

  public emptyTrainEpisode(){
    this.trainEpisode = {
      joinsMade: [], 
      learnedFeatureTensor: {hiddenStates:[], memoryCell:[]}, 
      sharedVariables: [], 
      leafFeatureTensor: [],
      graphViews: {subSubView: [], objObjView: [], objSubView: []},
      isEmpty:true
    };    
  }
  /**
   * 
   * @returns Empty binding stream
   */
  public returnNop(){
    const nop: IQueryOperationResult = {
      bindingsStream: new ArrayIterator([this.BF.bindings()], {autoStart: false}),
      metadata: () => Promise.resolve({
        cardinality: { type: 'exact', value: 1 },
        canContainUndefs: false,
        variables: [],
      }),
      type: 'bindings',
    };
    console.log("Finish returning nop")
    return QueryEngineBase.internalToFinalResult(nop)
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
  joinIndexes: number[][]
  N: number;
}
