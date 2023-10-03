import { ActorRdfJoin, IActionRdfJoin } from '@comunica/bus-rdf-join';
import { KeysQueryOperation, KeysRlTrain } from '@comunica/context-entries';
import { Actor, IActorReply, IMediatorArgs, Mediator } from '@comunica/core';
import { IMediatorTypeJoinCoefficients } from '@comunica/mediatortype-join-coefficients';
import { IBatchedTrainingExamples, IQueryOperationResult, ITrainEpisode } from '@comunica/types';
import * as tf from '@tensorflow/tfjs-node';
import * as path from 'path';
import * as fs from 'fs';
import { Operation } from 'sparqlalgebrajs/lib/algebra';
import { ISingleResultSetRepresentation, InstanceModelGCN } from '@comunica/actor-rdf-join-inner-multi-reinforcement-learning-tree';
import * as rdfjs from '@rdfjs/types';
/**
 * A comunica join-reinforcement-learning mediator.
 */
export class MediatorJoinReinforcementLearning extends Mediator<ActorRdfJoin, IActionRdfJoin, IMediatorTypeJoinCoefficients, IQueryOperationResult> {
  public readonly cpuWeight: number;
  public readonly memoryWeight: number;
  public readonly timeWeight: number;
  public readonly ioWeight: number;
  public predicateVectors: Map<string,number[]>;

  public constructor(args: IMediatorJoinCoefficientsFixedArgs) {
    super(args);
    this.readPredicateVectors()
  }
  /**
   * Function to read in prebuilt predicate vectors, this is to easily allow us to make these vectors in python
   * For actual we could make an actor that creates these vectors, but that would require a lot of work to make RDF2Vec work on javascript
   */
  public readPredicateVectors(){
    const vectorLocation = path.join(__dirname, "..", "models/predicate-vectors-depth-1/vectors_depth_1.txt");
    this.predicateVectors = new Map();
    let keyedVectors = fs.readFileSync(vectorLocation, 'utf-8').trim().split('\n');
    for (const vector of keyedVectors){
      const keyVectorArray: string[] = vector.split('[sep]');
      const vectorRepresentation: number[] = keyVectorArray[1].trim().split(" ").map(Number);
      this.predicateVectors.set(keyVectorArray[0], vectorRepresentation);
    }
  }

  public async mediateActor(action: IActionRdfJoinReinforcementLearning): Promise<ActorRdfJoin> {
    const episode: ITrainEpisode|undefined = action.context.get(KeysRlTrain.trainEpisode);
    if (episode==undefined){
      throw new Error("Passed undefined training episode to mediator");
    }
    if (episode.isEmpty && action.entries.length > 1){
      await this.initialiseFeatures(action);
    }
    const testResults: IActorReply<ActorRdfJoin, IActionRdfJoinReinforcementLearning, IMediatorTypeReinforcementLearning, IQueryOperationResult>[] =
    this.publish(action);
    return await this.mediateWith(action, testResults);
  }

  protected async mediateWith(
    action: IActionRdfJoinReinforcementLearning,
    testResults: IActorReply<ActorRdfJoin, IActionRdfJoinReinforcementLearning, IMediatorTypeReinforcementLearning, IQueryOperationResult>[],
  ): Promise<ActorRdfJoin> {
    // Obtain test results
    console.log("Start mediate with");
    const errors: Error[] = [];
    const promises = testResults
      .map(({ reply }) => reply)
      .map(promise => promise.catch(error => {
        errors.push(error);
      }));

    const coefficients = await Promise.all(promises);

    // Calculate costs
    let costs: (number | undefined)[] = coefficients
      // eslint-disable-next-line array-callback-return
      .map((coeff, i) => {
        if (coeff) {
          return coeff.iterations * this.cpuWeight +
            coeff.persistedItems * this.memoryWeight +
            coeff.blockingItems * this.timeWeight +
            coeff.requestTime * this.ioWeight;
        }
      });
    const maxCost = Math.max(...(<number[]> costs.filter(cost => cost !== undefined)));

    // If we have a limit indicator in the context,
    // increase cost of entries that have a number of iterations that is higher than the limit AND persist items.
    // In these cases, join operators that produce results early on will be preferred.
    const limitIndicator: number | undefined = action.context.get(KeysQueryOperation.limitIndicator);
    if (limitIndicator) {
      costs = costs.map((cost, i) => {
        if (cost !== undefined && coefficients[i]!.persistedItems > 0 && coefficients[i]!.iterations > limitIndicator) {
          return cost + maxCost;
        }
        return cost;
      });
    }

    // Determine index with lowest cost
    let minIndex = -1;
    let minValue = Number.POSITIVE_INFINITY;
    for (const [ i, cost ] of costs.entries()) {
      if (cost !== undefined && (minIndex === -1 || cost < minValue)) {
        minIndex = i;
        minValue = cost;
      }
    }

    // Reject if all actors rejected
    if (minIndex < 0) {
      throw new Error(`All actors rejected their test in ${this.name}\n${
        errors.map(error => error.message).join('\n')}`);
    }

    // Return actor with lowest cost
    const bestActor = testResults[minIndex].actor;
  
    // If the actor gave us a next join to execute we update our action and proceed to update trainEpisode
    const bestActorReply: IMediatorTypeReinforcementLearning = await testResults[minIndex].reply
    if (bestActorReply.joinIndexes){
      // Update the train episode if we get representations
      if (!bestActorReply.predictedQValue || !bestActorReply.representation){
        throw new Error("Got join indexes but not q-value or representation");
      }
      const trainEpisode: ITrainEpisode|undefined = action.context.get(KeysRlTrain.trainEpisode);
      if (!trainEpisode){
        throw new Error("Mediator passed undefined trainEpisode");
      }

      // First remove entries of join and dispose
      const unsortedIndexes = [...bestActorReply.joinIndexes];
      const idx = bestActorReply.joinIndexes.sort((a,b)=> b-a);

      action.nextJoinIndexes = idx;
      // Check if this works with multiple query executions
      trainEpisode.joinsMade.push(unsortedIndexes);
      for (const i of idx){
          // Dispose tensors before removing pointer to avoid "floating" tensors
          trainEpisode.learnedFeatureTensor.hiddenStates[i].dispose(); trainEpisode.learnedFeatureTensor.memoryCell[i].dispose();
          // Remove features associated with index
          trainEpisode.learnedFeatureTensor.hiddenStates.splice(i,1); trainEpisode.learnedFeatureTensor.memoryCell.splice(i,1);
      }
      // Add new join representation
      // trainEpisode.estimatedQValues.push(bestActorReply.predictedQValue);
      trainEpisode.learnedFeatureTensor.hiddenStates.push(bestActorReply.representation.hiddenState);
      trainEpisode.learnedFeatureTensor.memoryCell.push(bestActorReply.representation.memoryCell);

      // Update training data batch
      // TODO: turn checks off for benchmarking and actual execution!!
      const batchToUpdate: IBatchedTrainingExamples|undefined = action.context.get(KeysRlTrain.batchedTrainingExamples)!;
      if (!batchToUpdate){
        throw new Error("Mediator is passed undefined as BatchedTrainingExamples object");
      }
      if (!batchToUpdate.trainingExamples){
        throw new Error("Mediator is passed invalid BatchedTrainingExamples object");
      }
      const key: string = MediatorJoinReinforcementLearning.idxToKey(trainEpisode.joinsMade);

      // We only add qValue to batch if not already in trainingBatch. If it is in training batch we don't update it
      // because we expect qValues for the same order to be consistent over a training episode due to the deterministic nature
      // of network forward passes
      if(!batchToUpdate.trainingExamples.get(key)){
        batchToUpdate.trainingExamples.set(MediatorJoinReinforcementLearning.idxToKey(trainEpisode.joinsMade), {qValue: bestActorReply.predictedQValue.dataSync()[0], actualExecutionTime: -1, N: 0});
      }
      bestActorReply.predictedQValue.dispose();
    }
    if (!action.context.get(KeysRlTrain.batchedTrainingExamples)){
      throw new Error("Mediator did not receive a batched training example object");
    }

    // Emit calculations in logger
    if (bestActor.includeInLogs) {
      Actor.getContextLogger(action.context)?.debug(`Determined physical join operator '${bestActor.logicalType}-${bestActor.physicalName}'`, {
        entries: action.entries.length,
        variables: await Promise.all(action.entries
          .map(async entry => (await entry.output.metadata()).variables.map(variable => variable.value))),
        costs: Object.fromEntries(costs.map((coeff, i) => [
          `${testResults[i].actor.logicalType}-${testResults[i].actor.physicalName}`,
          coeff,
        ])),
        coefficients: Object.fromEntries(coefficients.map((coeff, i) => [
          `${testResults[i].actor.logicalType}-${testResults[i].actor.physicalName}`,
          coeff,
        ])),
      });
    }
    console.log("End mediate with")
    return bestActor;
  }

  protected async initialiseFeatures(action: IActionRdfJoinReinforcementLearning){  
    if (!action.context.get(KeysQueryOperation.operation)){
      throw new Error("Action context does not contain any query operations");
    }    
    const vectorSize: number = 128;

    const leafFeatures = await this.getLeafFeatures(action, vectorSize);    
    const sharedVariables = await this.getSharedVariableTriplePatterns(action);
    const graphViews: IQueryGraphViews = MediatorJoinReinforcementLearning.createQueryGraphViews(action);

    // Take models from context
    const models = (action.context.get(KeysRlTrain.modelInstanceGCN) as InstanceModelGCN).getModels();

    // Update running moments only at leaf
    const runningMomentsFeatures: IRunningMoments = action.context.get(KeysRlTrain.runningMomentsFeatures)!;
    this.updateAllMoments(runningMomentsFeatures, leafFeatures);
    this.standardiseFeatures(runningMomentsFeatures, leafFeatures);

    // Feature tensors created here, after no longer needed (corresponding result sets are joined) they need to be disposed
    const featureTensorLeaf: tf.Tensor[] = leafFeatures.map(x=>tf.tensor(x,[1, x.length]));
    const memoryCellLeaf: tf.Tensor[] = featureTensorLeaf.map(x=>tf.zeros(x.shape))

    const resultSetRepresentationLeaf: IResultSetRepresentation = {hiddenStates: featureTensorLeaf, memoryCell: memoryCellLeaf};

    if (!action.context.get(KeysRlTrain.trainEpisode)){
      throw new Error("No train episode given in context");
    }

    const episode: ITrainEpisode = action.context.get(KeysRlTrain.trainEpisode)!;
    const batchToUpdate: IBatchedTrainingExamples|undefined = action.context.get(KeysRlTrain.batchedTrainingExamples)!;

    if (!batchToUpdate){
      throw new Error("Mediator is passed undefined as BatchedTrainingExamples object");
    }

    // Clone of tensors, will prob make memory leaks!
    // Outdated, investigate if batchedTrainEpisode is even necessary
    if (batchToUpdate.leafFeatures.hiddenStates.length==0){
      batchToUpdate.leafFeatures = {hiddenStates: resultSetRepresentationLeaf.hiddenStates.map(x=>x.clone()), 
        memoryCell: resultSetRepresentationLeaf.memoryCell.map(x=>x.clone())};  
    }

    // Obtain different query graph representations using GCN
    try{
      const subjSubjRepresentation = models.modelSubjSubj.forwardPass(tf.stack(featureTensorLeaf).squeeze(), tf.tensor(graphViews.subSubView));
      const objObjRepresentation = models.modelObjObj.forwardPass(tf.stack(featureTensorLeaf).squeeze(), tf.tensor(graphViews.objObjView));
      const objSubjRepresentation = models.modelObjSubj.forwardPass(tf.stack(featureTensorLeaf).squeeze(), tf.tensor(graphViews.objSubView));  
    }
    catch(err){
      console.log(err)
    }
    const subjSubjRepresentation = models.modelSubjSubj.forwardPass(tf.stack(featureTensorLeaf).squeeze(), tf.tensor(graphViews.subSubView));
    const objObjRepresentation = models.modelObjObj.forwardPass(tf.stack(featureTensorLeaf).squeeze(), tf.tensor(graphViews.objObjView));
    const objSubjRepresentation = models.modelObjSubj.forwardPass(tf.stack(featureTensorLeaf).squeeze(), tf.tensor(graphViews.objSubView));  
    // Concatinate all representations 
    try{
      const learnedRepresentation = tf.concat([subjSubjRepresentation, objObjRepresentation, objSubjRepresentation], 1);
    }
    catch(err){
      console.log(err)
    }
    const learnedRepresentation = tf.concat([subjSubjRepresentation, objObjRepresentation, objSubjRepresentation], 1);
    const learnedRepresentationList = tf.split(learnedRepresentation, learnedRepresentation.shape[0]);
    const learnedRepresentationResultSet: IResultSetRepresentation = {
      hiddenStates: learnedRepresentationList,
      memoryCell: learnedRepresentationList.map(x=>tf.zeros(x.shape))
    };

    // Initialise episode
    episode.learnedFeatureTensor = learnedRepresentationResultSet;
    episode.leafFeatureTensor = featureTensorLeaf
    episode.sharedVariables=sharedVariables;
    episode.isEmpty = false;
    episode.graphViews = graphViews;
    console.log("Episode after making features")
    console.log(episode);

  }

  public static createQueryGraphViews(action: IActionRdfJoinReinforcementLearning){
    // Duplicate code, remove for refactor
    const patterns: Operation[] = action.entries.map((x)=>{
      if (!MediatorJoinReinforcementLearning.isPattern(x.operation)){
        throw new Error("Found a non pattern during feature initialisation");
      }
      else{
          return x.operation;
      }
    });
    const subjectObjects: rdfjs.Term[][] = patterns.map(x => [x.subject, x.object]);
    // const subObjGraph = MediatorJoinReinforcementLearning.createSubjectObjectView(subjectObjects);
    const objSubGraph = MediatorJoinReinforcementLearning.createObjectSubjectView(subjectObjects);
    const subSubGraph = MediatorJoinReinforcementLearning.createSubjectSubjectView(subjectObjects);
    const objObjGraph = MediatorJoinReinforcementLearning.createObjectObjectView(subjectObjects);
    const graphViews: IQueryGraphViews = {subSubView: subSubGraph, objObjView: objObjGraph, 
     objSubView: objSubGraph};
    return graphViews;
  }

  // This is a directed graph where if triple pattern $i$ subject is triple pattern $j$ object, we have A_{ij} = 1, but A_{ji} = 0
  // So row i column j = 1, but row j column i = 0. This denotes linear queries. I believe this is currently redundant
  public static createSubjectObjectView(subjectObjects: rdfjs.Term[][]): number[][]{
    const adjMatrixSubObj = new Array(subjectObjects.length).fill(0).map(x=>new Array(subjectObjects.length).fill(0));
    for (let i = 0; i<subjectObjects.length; i++){
      // Add self connection
      adjMatrixSubObj[i][i] = 1;
      // Set subject->object (or object->subject) connections
      const outerTerm = subjectObjects[i][0];
      // Only valid if we have Variables
      if (outerTerm.termType == 'Variable'){
        for (let j = i; j<subjectObjects.length; j++){
          const innerTerm = subjectObjects[j][1];
          if (innerTerm.termType== 'Variable' && outerTerm.value == innerTerm.value){
            adjMatrixSubObj[i][j] = 1;
          }
        }
      }
    }
    return adjMatrixSubObj;
  }

  // This is a directed graph where if triple pattern $i$ object is triple pattern $j$ subject, we have A_{ij} = 1, but A_{ji} = 0
  // So row i column j = 1, but row j column i = 0. This denotes linear queries.
  public static createObjectSubjectView(subjectObjects: rdfjs.Term[][]): number[][]{
    const adjMatrixObjSub = new Array(subjectObjects.length).fill(0).map(x=>new Array(subjectObjects.length).fill(0));
    for (let i = 0; i<subjectObjects.length; i++){
      // Add self connection
      adjMatrixObjSub[i][i] = 1;
      // Set subject->object (or object->subject) connections
      const outerTerm = subjectObjects[i][1];
      // Only valid if we have Variables
      if (outerTerm.termType == 'Variable'){
        for (let j = i; j<subjectObjects.length; j++){
          const innerTerm = subjectObjects[j][0];
          if (innerTerm.termType== 'Variable' && outerTerm.value == innerTerm.value){
            adjMatrixObjSub[i][j] = 1;
          }
        }
      }
    }
    return adjMatrixObjSub;

  }

  // Represent star-shaped queries, where if triple pattern i and j share a subject, we have A_{ij} = A_{ji} = 1
  public static createSubjectSubjectView(subjectObjects: rdfjs.Term[][]): number[][]{
    const adjMatrixSubSub = new Array(subjectObjects.length).fill(0).map(x=>new Array(subjectObjects.length).fill(0));
    for (let i = 0; i<subjectObjects.length; i++){
      // Add self connection
      adjMatrixSubSub[i][i] = 1;
      // Set subject->subject connections
      const outerTerm = subjectObjects[i][0];
      // Only valid if we have Variables
      if (outerTerm.termType == 'Variable'){
        for (let j = i; j<subjectObjects.length; j++){
          const innerTerm = subjectObjects[j][0];
          if (innerTerm.termType== 'Variable' && outerTerm.value == innerTerm.value){
            adjMatrixSubSub[i][j] = 1;
            adjMatrixSubSub[j][i] = 1;
          }
        }
      }
    }
    return adjMatrixSubSub;
  }

  // Represents inverse star shaped queries, where if triple pattern i and j share an object, we have A_{ij} = A_{ji} = 1
  public static createObjectObjectView(subjectObjects: rdfjs.Term[][]): number[][]{
    const adjMatrixObjObj = new Array(subjectObjects.length).fill(0).map(x=>new Array(subjectObjects.length).fill(0));
    for (let i = 0; i<subjectObjects.length; i++){
      // Add self connection
      adjMatrixObjObj[i][i] = 1;
      // Set subject->subject connections
      const outerTerm = subjectObjects[i][1];
      // Only valid if we have Variables
      if (outerTerm.termType == 'Variable'){
        for (let j = i; j<subjectObjects.length; j++){
          const innerTerm = subjectObjects[j][1];
          if (innerTerm.termType== 'Variable' && outerTerm.value == innerTerm.value){
            adjMatrixObjObj[i][j] = 1;
            adjMatrixObjObj[j][i] = 1;
          }
        }
      }
    }
    return adjMatrixObjObj;
  }

  public async getLeafFeatures(action: IActionRdfJoinReinforcementLearning, vectorSize: number){
    const patterns: Operation[] = action.entries.map((x)=>{
      if (!MediatorJoinReinforcementLearning.isPattern(x.operation)){
        throw new Error("Found a non pattern during feature initialisation");
      }
      else{
          return x.operation;
      }
    });

    const leafFeatures: number[][] = [];
    for(let i=0;i<action.entries.length;i++){
      // Encode triple pattern information
      const triple = [patterns[i].subject, patterns[i].predicate, patterns[i].object];
      const isVariable: number[] = triple.map(x => x.termType=='Variable' ? 1 : 0);
      const isNamedNode: number[] = triple.map(x => x.termType=='NamedNode' ? 1 : 0)
      const isLiteral: number[] = triple.map(x=> x.termType=='Literal' ? 1 : 0);
      const cardinalityLeaf: number = (await action.entries[i].output.metadata()).cardinality.value;

      // Get predicate embedding
      let predicateEmbedding: number[] = [];
      if (this.predicateVectors.get(patterns[i].predicate.value)){
        predicateEmbedding = this.predicateVectors.get(patterns[i].predicate.value)!;
      }
      else{
        // Zero vector represents unknown predicate
        predicateEmbedding = new Array(vectorSize).fill(0);
      }
      // leafFeatures.push([cardinalityLeaf].concat(isVariable[0]));
      leafFeatures.push([cardinalityLeaf].concat(isVariable, isNamedNode, isLiteral, predicateEmbedding));
    }
    return leafFeatures;
  }

  public async getSharedVariableTriplePatterns(action: IActionRdfJoinReinforcementLearning){
    const patterns: Operation[] = action.entries.map((x)=>{
      if (!MediatorJoinReinforcementLearning.isPattern(x.operation)){
        throw new Error("Found a non pattern during construction of shared variables in RL actor");
      }
      else{
          return x.operation;
      }
    });
    
    const sharedVariables: number[][] = [];
    for(let i=0;i<action.entries.length;i++){
      const outerEntryConnections: number[] = new Array(action.entries.length).fill(0);
      // Get the 'outer' triple we use to compare
      const tripleOuter = [patterns[i].subject, patterns[i].predicate, patterns[i].object];
      const variablesFull = tripleOuter.filter(x => x.termType=='Variable');
      const variables = variablesFull.map(x=>x.value);
      for (let j=0;j<action.entries.length;j++){
        const tripleInner = [patterns[j].subject.value, patterns[j].predicate.value, patterns[j].object.value];
        for (const term of tripleInner){
          if (variables.includes(term)){
            outerEntryConnections[j] = 1;
          }
        }
      }
      sharedVariables.push([...outerEntryConnections])
    }
    return sharedVariables;
  }

  protected static isPattern(inputInterface: any){
    return ('subject' in inputInterface && 'predicate' in inputInterface && 'object' in inputInterface);
  }

  public static idxToKey(indexes: number[][]){
    return indexes.flat().toString().replaceAll(',', '');
  }

  public static keyToIdx(key:string){
    const chars = key.split('');
    const idx: number[][]=[];
    for (let i=0;i<chars.length;i+=2){
      idx.push([+chars[i], +chars[i+1]]);
    }
    return idx;
  }

  public standardiseFeatures(runningMomentsFeatures: IRunningMoments, features: number[][]){
    for (const idx of runningMomentsFeatures.indexes){
      const momentsAtIndex: IAggregateValues = runningMomentsFeatures.runningStats.get(idx)!;
      if (!momentsAtIndex){
        console.error("Accessing moments index that does not exist");
      }
      for (const feature of features){
        feature[idx] = (feature[idx] - (momentsAtIndex.mean))/momentsAtIndex.std;
      }    
    }
  }

  public updateAllMoments(runningMoments: IRunningMoments, newFeatures: number[][]){
    for (const index of runningMoments.indexes){
      for (const newFeature of newFeatures){
        this.updateMoment(runningMoments.runningStats.get(index)!, newFeature[index]);
      }
    }
  }

  public updateMoment(toUpdateAggregate: IAggregateValues, newValue: number){
    toUpdateAggregate.N +=1;
    const delta = newValue - toUpdateAggregate.mean; 
    toUpdateAggregate.mean += delta / toUpdateAggregate.N;
    const newDelta = newValue - toUpdateAggregate.mean;
    toUpdateAggregate.M2 += delta * newDelta;
    toUpdateAggregate.std = Math.sqrt(toUpdateAggregate.M2 / toUpdateAggregate.N);
  }
}


export interface IMediatorJoinCoefficientsFixedArgs
  extends IMediatorArgs<ActorRdfJoin, IActionRdfJoin, IMediatorTypeJoinCoefficients, IQueryOperationResult> {
  /**
   * Weight for the CPU cost
   */
  cpuWeight: number;
  /**
   * Weight for the memory cost
   */
  memoryWeight: number;
  /**
   * Weight for the execution time cost
   */
  timeWeight: number;
  /**
   * Weight for the I/O cost
   */
  ioWeight: number;
}


// Should be moved to mediatortype instance

export interface IMediatorTypeReinforcementLearning extends IMediatorTypeJoinCoefficients{
  /**
   * predicted Q-value as tensor to keep track of gradients
   */
  predictedQValue?: tf.Tensor;
  /**
   * New representation obtained from Tree LSTM as tensor
   */
  representation?: ISingleResultSetRepresentation
  /**
   * The indexes of the two results sets joined
   */
  joinIndexes?: number[];
  /**
   * Array of all joins in join plan, used as unique index for join order for specific query
   */
}

export interface IActionRdfJoinReinforcementLearning extends IActionRdfJoin{
  nextJoinIndexes?: number[];
}

export interface IResultSetRepresentation{
  hiddenStates: tf.Tensor[];
  memoryCell: tf.Tensor[];
}

export interface IResultSetRepresentationGraph{
  hiddenStates: tf.Tensor[];
  memoryCell: tf.Tensor[];
  graphViews: IQueryGraphViews;
}

export interface IRunningMoments{
  /*
  * Indexes of features to normalise and track
  */
  indexes: number[];
  /*
  * The running values of mean, std, n, mn mapped to index
  * These are the quantities required for the Welfold's online algorithm
  */
  runningStats: Map<number, IAggregateValues>;
}

export interface IAggregateValues{
  N: number;
  /*
  * Aggregate mean
  */
  mean: number;
  /*
  * Aggregate standard deviation
  */
  std: number;
  /*
  * Aggregate M2
  */
  M2: number;
}

export interface IQueryGraphViews{
  /**
   * Adjacency matrix for Subject to Subject connections of triple patterns
   */
  subSubView: number[][];
  /**
   * Adjacency matrix for Object to Object connections of triple patterns
   */
  objObjView: number[][];
  /**
   * Adjacency matrix for Object to Subject connections of triple patterns
   */
  objSubView: number[][];
}