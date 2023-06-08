import { ActorRdfJoin, IActionRdfJoin } from '@comunica/bus-rdf-join';
import { KeysQueryOperation, KeysRlTrain } from '@comunica/context-entries';
import { ActionContext, Actor, IAction, IActorOutput, IActorReply, IActorTest, IMediatorArgs, Mediator } from '@comunica/core';
import { IMediatorTypeJoinCoefficients } from '@comunica/mediatortype-join-coefficients';
import { IBatchedTrainingExamples, IQueryOperationResult, ITrainEpisode } from '@comunica/types';
import * as tf from '@tensorflow/tfjs-node';
import * as path from 'path';
import * as fs from 'fs';
import { Operation, Pattern } from 'sparqlalgebrajs/lib/algebra';
import { ISingleResultSetRepresentation } from '../../actor-rdf-join-inner-multi-reinforcement-learning-tree/lib';
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
    const vectorLocation = path.join(__dirname, "..", "models/predicate-vectors-small/vectors.txt");
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
    if (episode.isEmpty){
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
          trainEpisode.featureTensor.hiddenStates[i].dispose(); trainEpisode.featureTensor.memoryCell[i].dispose();
          // Remove features associated with index
          trainEpisode.featureTensor.hiddenStates.splice(i,1); trainEpisode.featureTensor.memoryCell.splice(i,1);
      }
      // Add new join representation
      // trainEpisode.estimatedQValues.push(bestActorReply.predictedQValue);
      trainEpisode.featureTensor.hiddenStates.push(bestActorReply.representation.hiddenState);
      trainEpisode.featureTensor.memoryCell.push(bestActorReply.representation.memoryCell);

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
    
    return bestActor;
  }

  protected async initialiseFeatures(action: IActionRdfJoinReinforcementLearning){  
    if (!action.context.get(KeysQueryOperation.operation)){
      throw new Error("Action context does not contain any query operations");
    }    
    // Matrix of vectors that represent the triple pattern result sets in the query
    const patterns: Operation[] = action.entries.map((x)=>{
      if (!this.isPattern(x.operation)){
        throw new Error("Found a non pattern during feature initialisation");
      }
      else{
          return x.operation;
      }
    });

    // Initialise leaf features obtained from triple patterns
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
      const vectorSize: number = this.predicateVectors.keys().next().value;
      if (this.predicateVectors.get(patterns[i].predicate.value)){
        predicateEmbedding = this.predicateVectors.get(patterns[i].predicate.value)!;
      }
      else{
        // Zero vector represents unknown predicate
        predicateEmbedding = new Array(vectorSize).fill(0);
      }
      // leafFeatures.push([cardinalityLeaf].concat(isVariable[0]));
      leafFeatures.push([cardinalityLeaf].concat(isVariable, isNamedNode, isLiteral, action.entries.length, predicateEmbedding));
    }
    
    const sharedVariables = await this.getSharedVariableTriplePatterns(action);

    // Update running moments only at leaf
    const runningMomentsFeatures: IRunningMoments = action.context.get(KeysRlTrain.runningMomentsFeatures)!;
    this.updateAllMoments(runningMomentsFeatures, leafFeatures);
    this.standardiseFeatures(runningMomentsFeatures, leafFeatures);
    // Feature tensors created here, after no longer needed (corresponding result sets are joined) they need to be disposed
    const featureTensorLeaf: tf.Tensor[] = leafFeatures.map(x=>tf.tensor(x,[1, x.length]));
    const memoryCellLeaf: tf.Tensor[] = featureTensorLeaf.map(x=>tf.zeros(x.shape));
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
    if (batchToUpdate.leafFeatures.hiddenStates.length==0){
      batchToUpdate.leafFeatures = {hiddenStates: resultSetRepresentationLeaf.hiddenStates.map(x=>x.clone()), 
        memoryCell: resultSetRepresentationLeaf.memoryCell.map(x=>x.clone())};  
    }

    episode.featureTensor=resultSetRepresentationLeaf;
    episode.sharedVariables=sharedVariables;
    episode.isEmpty = false;
  }

  public async getSharedVariableTriplePatterns(action: IActionRdfJoinReinforcementLearning){
    const patterns: Operation[] = action.entries.map((x)=>{
      if (!this.isPattern(x.operation)){
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

  protected isPattern(inputInterface: any){
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