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
import { AsyncIterator, SingletonIterator } from 'asynciterator';
import type { Algebra } from 'sparqlalgebrajs';
import type { ActorInitQueryBase } from './ActorInitQueryBase';
import { MemoryPhysicalQueryPlanLogger } from './MemoryPhysicalQueryPlanLogger';
import { IAggregateValues, IResultSetRepresentation, IRunningMoments, MediatorJoinReinforcementLearning } from '@comunica/mediator-join-reinforcement-learning';
import { InstanceModel, ModelTrainerOffline } from '@comunica/actor-rdf-join-inner-multi-reinforcement-learning-tree';
import * as chalk from 'chalk';
import { ExperienceBuffer, IExperienceKey } from '@comunica/model-trainer';
import * as _ from 'lodash';
import * as fs from 'graceful-fs';
import * as path from 'path';
import { BindingsFactory } from '@comunica/bindings-factory';

import * as process from 'process';

/**
 * Base implementation of a Comunica query engine.
 */
export class QueryEngineBase<QueryContext extends IQueryContextCommon = IQueryContextCommon>
implements IQueryEngine<QueryContext> {
  private readonly actorInitQuery: ActorInitQueryBase;
  private readonly defaultFunctionArgumentsCache: FunctionArgumentsCache;
  public modelTrainerOffline: ModelTrainerOffline;
  public modelInstance: InstanceModel;
  public batchedTrainExamples: IBatchedTrainingExamples;

  // Values needed for endpoint training
  public trainingStateInformationLocation: string;
  public experienceBuffer: ExperienceBuffer;
  public runningMomentsFeatures: IRunningMoments;
  public runningMomentsExecution: IRunningMoments;
  public trainEpisode: ITrainEpisode;
  public nExpPerQuery: number;
  public BF: BindingsFactory;
  public breakRecursion: boolean;
  public currentQueryKey: string;
  
  // Some test values, should be removed when it works
  public numQueryCalls: number;



  public constructor(actorInitQuery: ActorInitQueryBase<QueryContext>) {
    this.actorInitQuery = actorInitQuery;
    this.defaultFunctionArgumentsCache = {};
    this.trainEpisode = {joinsMade: [], featureTensor: {hiddenStates:[], memoryCell:[]}, isEmpty:true};
    // Hardcoded, should be config, but not sure how to incorporate
    this.modelInstance = new InstanceModel();
    this.modelTrainerOffline = new ModelTrainerOffline({optimizer: 'adam', learningRate: 0.005});
    this.batchedTrainExamples = {trainingExamples: new Map<string, ITrainingExample>, leafFeatures: {hiddenStates: [], memoryCell:[]}};
    // End point training
    this.BF = new BindingsFactory();
    this.breakRecursion = false;
    this.trainingStateInformationLocation = __dirname + "/../trainingStateEngine/";
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
    this.experienceBuffer = new ExperienceBuffer(1500);
    this.nExpPerQuery = 8;
    this.numQueryCalls = 0;

  }

  public async querySingleTrainStep<QueryFormatTypeInner extends QueryFormatType>(
    query: QueryFormatTypeInner,
    context: QueryContext & QueryFormatTypeInner extends string ? QueryStringContext : QueryAlgebraContext,
  ){
    let trainLoss: number[] = [];
    try{
      trainLoss = this.readTrainLoss(this.trainingStateInformationLocation+'trainLoss.txt');
    }
    catch{
      console.warn(chalk.red("No train loss array found"));
    }

    const clonedContext = _.cloneDeep(context);
    clonedContext.batchedTrainingExamples = this.batchedTrainExamples;

    const queryKey = this.experienceBuffer.getQueryKey(query.toString());
    this.currentQueryKey = queryKey;

    const trainResult = await this.executeTrainQuery(query, clonedContext, queryKey, this.experienceBuffer);
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
      const loss = await this.trainModelExperienceReplay(experiences, features);
      trainLoss.push(loss);
    }
    this.writeTrainLoss(trainLoss, this.trainingStateInformationLocation+'trainLoss.txt');
    return trainResult[1];
  }

  public async loadState(){
    // Load running moments
    try{
      const runningMomentsFeatures = this.modelInstance.loadRunningMoments(this.trainingStateInformationLocation+'runningMomentsFeatures.txt');
      const runningMomentsExecution = this.modelInstance.loadRunningMoments(this.trainingStateInformationLocation+'runningMomentsExecution.txt');
      this.runningMomentsFeatures = runningMomentsFeatures;
      this.runningMomentsExecution = runningMomentsExecution;
    }
    catch{
      console.warn(chalk.red("INFO: No running moments found"));
    }
    // Load experience buffer
    try{
      this.experienceBuffer.loadBuffer(this.trainingStateInformationLocation);
    }
    catch(err){
      // If we can't find a certain file, we initialise an empty buffer
      this.experienceBuffer = new ExperienceBuffer(this.experienceBuffer.maxSize);
      console.warn(chalk.red("INFO: No existing or incomplete buffer found"));
    }
    // Load model weights
    try{
      await this.modelInstance.initModel(this.trainingStateInformationLocation+'weights');
    }
    catch(err){
      // Flush any initialisation that went through to start from fresh model
      this.modelInstance.flushModel()
      // Initialise random model
      await this.modelInstance.initModelRandom();
      console.warn(chalk.red("INFO: Failed to load model weights, randomly initialising model weights"));
    }
  }

  public saveState(timeOutValue: number){
    try{
      this.modelInstance.saveModel(this.trainingStateInformationLocation+'weights');
    }
    catch{
      console.warn(chalk.red("Failed to save model"));
    }
    // First add timedout experience to buffer with current execution time
    if (this.trainEpisode.joinsMade.length==0){
      // We have no joins in our episode, so we just save buffer as is and exit
      this.experienceBuffer.saveBuffer(this.trainingStateInformationLocation);
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
    this.experienceBuffer.saveBuffer(this.trainingStateInformationLocation);
    try{
      this.modelInstance.saveRunningMoments(this.runningMomentsFeatures, 
        this.trainingStateInformationLocation+'runningMomentsFeatures.txt');
      this.modelInstance.saveRunningMoments(this.runningMomentsExecution, 
        this.trainingStateInformationLocation+'runningMomentsExecution.txt');  
    }
    catch{
      console.warn(chalk.red("Failed to save running moments"));
    }
  }
  
  public readTrainLoss(fileLocation: string){
    const loss: number[] = JSON.parse(fs.readFileSync(fileLocation, 'utf-8'));
    return loss;
  }

  public writeTrainLoss(loss: number[], fileLocation: string){
    fs.writeFileSync(fileLocation, JSON.stringify(loss));
  }

  public async queryBindingsTrain<QueryFormatTypeInner extends QueryFormatType>(
    queries: QueryFormatTypeInner[][],
    valQueries: QueryFormatTypeInner[][],
    nSimTrain: number,
    nSimVal: number,
    nEpoch: number,
    nExpPerSim: number,
    sizeBuffer: number,
    batchedTrainExamples: IBatchedTrainingExamples,
    context: QueryContext & QueryFormatTypeInner extends string ? QueryStringContext : QueryAlgebraContext,
    nextModelLocation: string,
    pathEpochInfos: string[]
  ) {
    const experienceBuffer: ExperienceBuffer = new ExperienceBuffer(sizeBuffer);
    if (!batchedTrainExamples){
      throw new Error("Undefined batchedTrainExample passed");
    }
    this.batchedTrainExamples = batchedTrainExamples!;
    // Initialise training examples
    // let batchedTrainingExamples = {trainingExamples: new Map<string, ITrainingExample>, leafFeatures: {hiddenStates: [], memoryCell:[]}};
    // let batchedValidationExamples = {trainingExamples: new Map<string, ITrainingExample>, leafFeatures: {hiddenStates: [], memoryCell:[]}};

    // Init running moments of execution time
    const runningMomentsExecutionTime = {indexes: [0], runningStats: new Map<number, IAggregateValues>()};
    for (const index of runningMomentsExecutionTime.indexes){
        const startPoint: IAggregateValues = {N: 0, mean: 0, std: 1, M2: 1}
        runningMomentsExecutionTime.runningStats.set(index, startPoint);
    }
    // Initialise statistics tracking
    const totalEpochTrainLoss: number[] = [];
    const epochValLoss: number[] = [];
    const epochValExecutionTime: number[] = [];
    const epochValStdLoss: number[] = [];
    const averageSearchTime: number[] = [];
    const averageSearchTimeTemplate: number[] = [];
    

    // Add info to context
    context.batchedTrainingExamples = this.batchedTrainExamples;
    context.train = true;
    
    // Init experience replay buffer with leaf features of each query
    for (let l=0;l<queries.length;l++){
      for(let z=0;z<queries[l].length;z++){
        const queryKey: string = `${l}`+`${z}`;

        // To ensure an unchanged context each time
        const clonedContext = _.cloneDeep(context);
        clonedContext.batchedTrainingExamples = this.batchedTrainExamples;

        // Obtain leaf features and set in experiencebuffer to initialise
        const leafFeatures = await this.executeQueryInitFeaturesBuffer(queries[l][z], clonedContext);
        experienceBuffer.setLeafFeaturesQuery(queryKey, leafFeatures);
        this.cleanBatchTrainingExamples()
      }
    }


    // Main training loop
    for (let epoch=0;epoch<nEpoch;epoch++){
      let epochTrainLoss = [];
      for (let i=0;i<queries.length;i++){
        const searchTimes = [];
        console.log(`Query Template ${i+1}/${queries.length}`);
        for(let j=0;j<queries[i].length;j++){
          const queryKey: string = `${i}`+`${j}`;
          // const bindings = await this.queryOfType<QueryFormatTypeInner, IQueryBindingsEnhanced>(queries[i][j], context, 'bindings');
          for (let k=0;k<nSimTrain;k++){
            const clonedContext = _.cloneDeep(context);
            clonedContext.batchedTrainingExamples = this.batchedTrainExamples;
    
            const trainResult = await this.executeTrainQuery(queries[i][j], clonedContext, queryKey, experienceBuffer);
            searchTimes.push(trainResult[0]);
        
            const experiences: IExperience[] = [];
            const features = []; 
            // Sample experiences from the buffer if we have enough prior executions
            if (experienceBuffer.getSize()>1){
              for (let z=0;z<nExpPerSim;z++){
                  const experience: [IExperience, IExperienceKey] = experienceBuffer.getRandomExperience();
                  experiences.push(experience[0]);
                  const feature = experienceBuffer.getFeatures(experience[1].query)!;
                  features.push(feature);
                }
                const loss = await this.trainModelExperienceReplay(experiences, features);
                epochTrainLoss.push(loss);
              }
            this.cleanBatchTrainingExamples();
            }
            // averageSearchTime.push(searchTimes.reduce((a, b) => a + b, 0) / searchTimes.length);
          }
        averageSearchTimeTemplate.push(searchTimes.reduce((a,b) => a+b, 0) / searchTimes.length);
      }
    const clonedContext = _.cloneDeep(context);
    clonedContext.batchedTrainingExamples = this.batchedTrainExamples;
    console.log(averageSearchTimeTemplate);
    const [avgExecution, avgExecutionTemplate, stdExecutionTemplate, avgLoss, stdLoss, avgSearchTimeTemplate, stdSearchTimeTemplate] = 
    await this.validatePerformance(valQueries, nSimVal, runningMomentsExecutionTime, experienceBuffer, clonedContext);

    const avgLossTrain = epochTrainLoss.reduce((a, b) => a + b, 0) / epochTrainLoss.length;

    console.log(`Epoch ${epoch+1}/${nEpoch}: Train Loss: ${avgLossTrain}, Validation Execution time: ${avgExecution}, Loss: ${avgLoss}, Std: ${stdLoss}`);
    const checkPointLocation = path.join(nextModelLocation + "/chkp-"+epoch);

    fs.mkdir(checkPointLocation, (err)=>{
        if (err){
            return console.error(err);
        }
    });

    const epochStatisticsLocation = pathEpochInfos.map(x=>path.join(checkPointLocation, x));
    console.log(epochStatisticsLocation)

    totalEpochTrainLoss.push(avgLossTrain); epochValLoss.push(avgLoss); epochValExecutionTime.push(avgExecution); epochValStdLoss.push(stdLoss);    
    this.writeEpochFiles(epochStatisticsLocation, [totalEpochTrainLoss, epochValLoss, epochValStdLoss, epochValExecutionTime, 
      avgExecutionTemplate, stdExecutionTemplate, avgSearchTimeTemplate, stdSearchTimeTemplate]);
  }
}

public async validatePerformance<QueryFormatTypeInner extends QueryFormatType>(
  queries: QueryFormatTypeInner[][], 
  nSimVal: number, 
  runningMomentsExecutionTime: IRunningMoments,
  experienceBuffer: ExperienceBuffer,
  context: QueryContext & QueryFormatTypeInner extends string ? QueryStringContext : QueryAlgebraContext
  ): Promise<[number, number[], number[], number, number, number[], number[]]>{
  
  const rawExecutionTimesTemplate: number[][] = [];
  const searchTimesTemplate: number[][] = [];
  for(let i=0;i<queries.length;i++){
    const rawExecutionTimes: number[] = []
    const searchTemplate: number[] = [];
    for (let j=0;j<queries[i].length;j++){
        for (let k=0;k<nSimVal;k++){
          const queryKey: string = `${i}`+`${j}`;

          const clonedContext = _.cloneDeep(context);
          clonedContext.batchedTrainingExamples = this.batchedTrainExamples;

          const valResults: number[] = await this.executeQueryValidation(queries[i][j], queryKey, clonedContext, experienceBuffer);
          rawExecutionTimes.push(valResults[0]);
          searchTemplate.push(valResults[1])
        }
    }
      rawExecutionTimesTemplate.push(rawExecutionTimes);
      searchTimesTemplate.push(searchTemplate);
  }
  // Get raw validaiton execution time statistics
  const flatExeTime: number[] = rawExecutionTimesTemplate.flat();
  const avgExecutionTime = flatExeTime.reduce((a, b) => a + b, 0) / flatExeTime.length;
  const avgExecutionTimeTemplate = rawExecutionTimesTemplate.map(x=>(x.reduce((a,b)=> a+b,0))/x.length);
  const avgSearchTimeTemplate = searchTimesTemplate.map(x=>(x.reduce((a,b)=> a+b,0))/x.length);

  const stdExecutionTimeTemplate = [];
  const stdSearchTimeTemplate = [];

  for (let k=0;k<avgExecutionTimeTemplate.length;k++){
      stdExecutionTimeTemplate.push(this.stdArray(rawExecutionTimesTemplate[k],avgExecutionTimeTemplate[k]))
      stdSearchTimeTemplate.push(this.stdArray(searchTimesTemplate[k], avgSearchTimeTemplate[k]));
  }
  // Get validation loss
  const MSE: number[] = [];
  for (const [_, trainExample] of this.batchedTrainExamples.trainingExamples.entries()){
      const singleMSE = (trainExample.qValue - trainExample.actualExecutionTime)**2;
      MSE.push(singleMSE);
  }
  const averageLoss = MSE.reduce((a,b)=> a+b,0)/MSE.length;
  const stdLoss = this.stdArray(MSE, averageLoss);

  // Clean training examples
  this.cleanBatchTrainingExamples();

  return [avgExecutionTime, avgExecutionTimeTemplate, stdExecutionTimeTemplate, averageLoss, stdLoss, avgSearchTimeTemplate, stdSearchTimeTemplate];
  }


  public async executeTrainQuery<QueryFormatTypeInner extends QueryFormatType>(
    query: QueryFormatTypeInner, 
    context: QueryContext & QueryFormatTypeInner extends string ? QueryStringContext : QueryAlgebraContext,
    queryKey: string,
    experienceBuffer: ExperienceBuffer
    ):Promise<[number, BindingsStream]>{
    const startT = this.getTimeSeconds();
    console.log("Starting query execution");
    const binding = await this.queryOfType<QueryFormatTypeInner, IQueryBindingsEnhanced>(query, context, 'bindings');
    console.log("Got binding");
    const endTSearch = this.getTimeSeconds();
    
    if(this.trainEpisode.joinsMade.length==0){
      console.log("We got an empty episode.. :(");
      this.disposeTrainEpisode();
      this.trainEpisode = {joinsMade: [], featureTensor: {hiddenStates:[], memoryCell:[]}, isEmpty:true};    
      return [endTSearch - startT, binding];
    }
    
    const joinOrderKeys: string[] = [];
    // Get joins made in this query to update
    for (let i = 1; i <this.trainEpisode.joinsMade.length+1;i++){
        const joinIndexId = this.trainEpisode.joinsMade.slice(0, i);
        joinOrderKeys.push(MediatorJoinReinforcementLearning.idxToKey(joinIndexId));
    }
    console.log("Waiting for this")
    const elapsed = await this.consumeStream(binding, experienceBuffer, startT, joinOrderKeys, queryKey, false, true);
    console.log(`Done: elapsed = ${elapsed}`);
    // Clear episode tensors
    this.disposeTrainEpisode();
    this.trainEpisode = {joinsMade: [], featureTensor: {hiddenStates:[], memoryCell:[]}, isEmpty:true};    
    return [endTSearch-startT, binding];
  }

  public async executeQueryValidation<QueryFormatTypeInner extends QueryFormatType>(
    query: QueryFormatTypeInner, 
    queryKey: string,
    context: QueryContext & QueryFormatTypeInner extends string ? QueryStringContext : QueryAlgebraContext,
    experienceBuffer: ExperienceBuffer
  ){
    const startTime: number = this.getTimeSeconds();
    const bindingsStream: BindingsStream = await this.queryOfType<QueryFormatTypeInner, IQueryBindingsEnhanced>(query, context, 'bindings');
    const endTimeSearch: number = this.getTimeSeconds();
    const joinOrderKeys: string[] = [];
    for (let i = 1; i <this.trainEpisode.joinsMade.length+1;i++){
        const joinIndexId = this.trainEpisode.joinsMade.slice(0, i);
        joinOrderKeys.push(MediatorJoinReinforcementLearning.idxToKey(joinIndexId));
    }    
    const executionTimeRaw: number = await this.consumeStream(bindingsStream, experienceBuffer, 
      startTime, joinOrderKeys, queryKey, true, false);
    this.disposeTrainEpisode();
    this.trainEpisode = {joinsMade: [], featureTensor: {hiddenStates:[], memoryCell:[]}, isEmpty:true};    
    return [executionTimeRaw, endTimeSearch - startTime];
  }

  public consumeStream(bindingStream: BindingsStream, 
    experienceBuffer: ExperienceBuffer,
    startTime: number, joinsMadeEpisode: string[], queryKey: string, 
    validation: boolean, recordExperience: boolean){
    const consumedStream: Promise<number> = new Promise((resolve, reject)=>{
      bindingStream.on('data', () =>{
      });
      bindingStream.on('end', () => {
        const endTime: number = this.getTimeSeconds();
        const elapsed: number = endTime-startTime;
        const statsY: IAggregateValues = this.runningMomentsExecution.runningStats.get(this.runningMomentsExecution.indexes[0])!;

        if (!validation){
          this.updateRunningMoments(statsY, elapsed);
        }
        if (recordExperience){
          console.log("Recording experience!!")
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
    this.trainEpisode = {joinsMade: [], featureTensor: {hiddenStates:[], memoryCell:[]}, isEmpty:true};
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
  ): Promise<ReturnType<QueryTypeOut['execute']>> {
    const result = await this.query<QueryFormatTypeInner>(query, context);
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
    /**
     * If there are no entries in queryToKey map we are at first run or after time-out, thus reload the model state.
     * We do this at start query to prevent unnecessary initialisation of features
     *  */    
    this.numQueryCalls += 1;

    if (context && this.experienceBuffer.queryToKey.size==0 && context.trainEndPoint){
      await this.loadState();
    }

    // Flag to determine if we should perform an initialisation query
    const initQuery = this.experienceBuffer.queryExists(query.toString())? false : true;
    console.log(`This is an init query ${initQuery}`)
    console.log(`Start query, number of query calls: ${this.numQueryCalls}`);
    // Prepare batchedTrainExamples so we can store our leaf features
    if (initQuery && context){
      context.batchedTrainingExamples = this.batchedTrainExamples;
    }
    
    /**
     * We execute training step on endpoint if we
     * 1. Get a context
     * 2. The context says we are doing endpoint training
     * 3. If we haven't already called this.querySingleTrainStep (To prevent infinite recursion)
     * 4. If the query already is initialised in the experienceBuffer
     */
    if (context && context.trainEndPoint && !this.breakRecursion && !initQuery){
      // Execute query step with given query
      this.breakRecursion = true;
      await this.querySingleTrainStep(query, context);
      this.breakRecursion = false;
      context.batchedTrainingExamples = this.batchedTrainExamples;
      // Reset train episode
      this.trainEpisode = {joinsMade: [], featureTensor: {hiddenStates:[], memoryCell:[]}, isEmpty:true};
      console.log(`End of training step, num query calls: ${this.numQueryCalls}`);
      return this.returnNop()
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
    // Initialise the running moments from file (The running moments file should only be passed on first execution)
    if(actionContext.get(KeysRlTrain.runningMomentsFeatureFile)){
      if (this.modelInstance.initMoments){
        console.warn(chalk.red("WARNING: Reloading Moments When They Are Already Initialised"));
      }

      const loadedRunningMomentsFeature: IRunningMoments = 
      this.modelInstance.loadRunningMoments(actionContext.get(KeysRlTrain.runningMomentsFeatureFile)!);
      this.runningMomentsFeatures = loadedRunningMomentsFeature;
      // Signal we loaded the moments
      this.modelInstance.initMoments=true;
    }

    // If no path was passed and the moments are not initialised we create mean 0 std 1 moments
    else if (!this.modelInstance.initMoments){
      console.warn(chalk.red("INFO: Running Query Without Initialised Running Moments"));
      for (const index of this.runningMomentsFeatures.indexes){
        const startPoint: IAggregateValues = {N: 0, mean: 0, std: 1, M2: 1}
        this.runningMomentsFeatures.runningStats.set(index, startPoint);
      }
      this.modelInstance.initMoments=true;
    }

    const baseIRI: string | undefined = actionContext.get(KeysInitQuery.baseIRI);
    await this.modelInstance.initModelRandom();
    actionContext = actionContext
      .setDefault(KeysInitQuery.queryTimestamp, new Date())
      .setDefault(KeysRdfResolveQuadPattern.sourceIds, new Map())
      // Set the default logger if none is provided
      .setDefault(KeysCore.log, this.actorInitQuery.logger)
      .setDefault(KeysInitQuery.functionArgumentsCache, this.defaultFunctionArgumentsCache)
      .setDefault(KeysRlTrain.trainEpisode, this.trainEpisode)
      .setDefault(KeysRlTrain.modelInstance, this.modelInstance)
      .setDefault(KeysRlTrain.runningMomentsFeatures, this.runningMomentsFeatures);

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
    console.log("Episode before exeuction")
    console.log(this.trainEpisode);
    const output = await this.actorInitQuery.mediatorQueryOperation.mediate({
      context: actionContext,
      operation,
    });
    
    console.log("Episode after exeuction")
    console.log(this.trainEpisode);

    output.context = actionContext;
    // If we are in initialisation run, we initialise the query with its features and return empty bindings
    if (initQuery){
      // Get the leaf features resulting from query
      console.log("Here is our train episode on init query:");
      console.log(this.trainEpisode)
      const leafFeatures: IResultSetRepresentation = {hiddenStates: this.batchedTrainExamples.leafFeatures.hiddenStates.map(x=>x.clone()),
        memoryCell: this.batchedTrainExamples.leafFeatures.memoryCell.map(x=>x.clone())};
      
      // Clean the examples
      this.cleanBatchTrainingExamples();
      this.disposeTrainEpisode();
      this.trainEpisode = {joinsMade: [], featureTensor: {hiddenStates:[], memoryCell:[]}, isEmpty:true};
      this.experienceBuffer.initQueryInformation(query.toString(), leafFeatures);    
      console.log(`Finished our init query, nQueryCalls: ${this.numQueryCalls}`);
      return this.returnNop();
    }

    // Reset train episode if we are not training (Bit of a shoddy workaround, because we need some episode information 
    // If we train
    if (!actionContext.get(KeysRlTrain.train) && context && !context.trainEndPoint){
      this.trainEpisode = {joinsMade: [], featureTensor: {hiddenStates:[], memoryCell:[]}, isEmpty:true};
    }
    console.log("We are at end of 'normal' exection during train episode");
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
  public saveModel(runningMomentsPath: string){
    this.modelInstance.saveModel();
    this.modelInstance.saveRunningMoments(this.runningMomentsFeatures, runningMomentsPath);

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
      actualExecutionTime, this.modelInstance.getModel(), 4);
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
     * 
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
      actualExecutionTimes, this.modelInstance.getModel(), 4);
    return loss;
  }

  public disposeTrainEpisode(){
    this.trainEpisode.featureTensor.hiddenStates.map(x=>x.dispose());
    this.trainEpisode.featureTensor.memoryCell.map(x=>x.dispose()); 
  }
  /**
   * 
   * @returns Empty binding stream
   */
  public returnNop(){
    const nop: IQueryOperationResult = {
      bindingsStream: new SingletonIterator(this.BF.bindings()),
      metadata: () => Promise.resolve({
        cardinality: { type: 'exact', value: 1 },
        canContainUndefs: false,
        variables: [],
      }),
      type: 'bindings',
    };
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
