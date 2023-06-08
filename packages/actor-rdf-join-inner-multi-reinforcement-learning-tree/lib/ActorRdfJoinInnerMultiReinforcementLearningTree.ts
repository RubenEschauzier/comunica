import {
  ActorQueryOperation,
} from '@comunica/bus-query-operation';
import type {
  IActionRdfJoin,
  IActorRdfJoinOutputInner,
  IActorRdfJoinArgs,
  MediatorRdfJoin,
} from '@comunica/bus-rdf-join';
import { ActorRdfJoin } from '@comunica/bus-rdf-join';
import type { MediatorRdfJoinEntriesSort } from '@comunica/bus-rdf-join-entries-sort';
import { KeysRlTrain } from '@comunica/context-entries';
import type { IMediatorTypeJoinCoefficients } from '@comunica/mediatortype-join-coefficients';
import type { MetadataBindings, IJoinEntry, IActionContext, IJoinEntryWithMetadata, ITrainEpisode } from '@comunica/types';
import { Factory } from 'sparqlalgebrajs';
import * as tf from '@tensorflow/tfjs-node';
import { IActionRdfJoinReinforcementLearning, IMediatorTypeReinforcementLearning, IResultSetRepresentation } from '@comunica/mediator-join-reinforcement-learning';
import { IModelOutput, ModelTreeLSTM } from './treeLSTM';
import { InstanceModel } from './instanceModel';
import { Operation } from 'sparqlalgebrajs/lib/algebra';

/**
 * A Multi Smallest RDF Join Actor.
 * It accepts 3 or more streams, joins the smallest two, and joins the result with the remaining streams.
 */
export class ActorRdfJoinInnerMultiReinforcementLearningTree extends ActorRdfJoin implements IActorRdfJoinInnerMultiReinforcementLearningTree {
  // public readonly mediatorJoinEntriesSort: MediatorRdfJoinEntriesSort;
  public readonly mediatorJoin: MediatorRdfJoin;
  public readonly mediatorJoinEntriesSort: MediatorRdfJoinEntriesSort;

  public static readonly FACTORY = new Factory();

  // Extend interface, define config properties
  public constructor(args: IActorRdfJoinInnerMultiReinforcementLearningTree) {
    super(args, {
      logicalType: 'inner',
      physicalName: 'multi-smallest',
      limitEntries: 3,
      limitEntriesMin: true,
    });
  }

  public getPossibleJoins(numJoinable: number, sharedVariables: number[][]){
    let orders: number[][] = [];
    for (let i=0; i<numJoinable;i++){
      for (let j=i+1; j<numJoinable;j++){
        // Only allow joins with shared variables
        if (sharedVariables[i][j]==1 || sharedVariables[j][i]==1){
          orders.push([i,j]);
        }
      }
    }
    return orders;
  }

  public chooseUsingProbability(probArray: Float32Array): number{
    // const cumsum = (arr: Float32Array|number[]) => arr.map((sum => value => sum += value)(0));
    let cumSumNew = [];
    let runningSum = 0;
    for (let j=0; j<probArray.length; j++){
      runningSum+= probArray[j];
      cumSumNew.push(runningSum);
    }
    const pick: number = Math.random();
    for (let i=0;i<cumSumNew.length;i++){
      /* If we land at our draw, we return the index of the draw*/
      if(pick > cumSumNew[i]){}
      else{
        return i;
      }
    }
    return cumSumNew.length-1;

  }

  public indexOfMax(arr: Float32Array) {
    if (arr.length === 0) {
        return -1;
    }
    let max = arr[0];
    let maxIndex = 0;
    for (var i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
            maxIndex = i;
            max = arr[i];
        }
    }
    return maxIndex;
  }

  public updateSharedVariables(sharedVariables: number[][], idx: number[]){
    // 1. First we add shared variables for result set of join
    const newSharedVariables = [];
    for (let k=0; k<sharedVariables[idx[0]].length;k++){
      newSharedVariables.push(Math.max(sharedVariables[idx[0]][k], sharedVariables[idx[1]][k]))
    }
    // 2. Remove variables involved in join
    newSharedVariables.splice(idx[0], 1); newSharedVariables.splice(idx[1], 1);

    // 3. Remove triple patterns in sharedVariables that are involved in the join
    sharedVariables.splice(idx[0], 1); sharedVariables.splice(idx[1], 1);

    // 4. Then we remove result sets involved in join in 2d and add 1 or 0 based on connection in new join entry
    for (let i=0; i<sharedVariables.length; i++){
      sharedVariables[i].splice(idx[0], 1); sharedVariables[i].splice(idx[1], 1);
      // Add zero or one based on connectivity result set join
      sharedVariables[i].push(newSharedVariables[i]);
    }
    // 5. Add self connection to result set shared variables, note that self-connections shouldn't be necessary, but seems nice to include
    newSharedVariables.push(1);

    // 6. Add shared variables of result set of join to all result set connectivities
    sharedVariables.push(newSharedVariables);
    return sharedVariables;
  }

  /**
   * Order the given join entries using the join-entries-sort bus.
   * @param {IJoinEntryWithMetadata[]} entries An array of join entries.
   * @param context The action context.
   * @return {IJoinEntryWithMetadata[]} The sorted join entries.
  */
  public async sortJoinEntries(
    entries: IJoinEntryWithMetadata[],
    context: IActionContext,
  ): Promise<IJoinEntryWithMetadata[]> {
    return (await this.mediatorJoinEntriesSort.mediate({ entries, context })).entries;
  }


  protected async getOutput(action: IActionRdfJoinReinforcementLearning): Promise<IActorRdfJoinOutputInner> {    
    // Determine the two smallest streams by sorting (e.g. via cardinality)
    const entries = [...action.entries];

    const trainEpisode: ITrainEpisode|undefined = action.context.get(KeysRlTrain.trainEpisode);
    if (!trainEpisode){
      throw new Error("Found undefined trainEpisode in RL actor")
    }
    if (!action.nextJoinIndexes){
      throw new Error("Indexes to join are not defined");
    }
    const idx = action.nextJoinIndexes;
    if (idx[0]<idx[1]){
      throw new Error("Found incorrectly sorted join indexes");
    }
    if (idx[0]==idx[1]){
      throw new Error("Found join indexes with same values");
    }
    const joinEntry1 = entries[idx[0]];
    const joinEntry2 = entries[idx[1]];
    entries.splice(idx[0], 1); entries.splice(idx[1], 1);

    // Join the two selected streams, and then join the result with the remaining streams
    // Sort these entries on cardinality to ensure the left relation is smallest!
    const sortedJoinCandidates: IJoinEntry[] = await this.sortJoinEntries(
      await ActorRdfJoin.getEntriesWithMetadatas([ joinEntry1, joinEntry2 ]),
      action.context,
    );

    const firstEntry: IJoinEntry = {
      output: ActorQueryOperation.getSafeBindings(await this.mediatorJoin
        .mediate({ type: action.type, entries: [ sortedJoinCandidates[0], sortedJoinCandidates[1] ], context: action.context })),
      operation: ActorRdfJoinInnerMultiReinforcementLearningTree.FACTORY
        .createJoin([ sortedJoinCandidates[0].operation, sortedJoinCandidates[1].operation ], false),
    };

    // Important for model execution that new entry is always entered last
    entries.push(firstEntry);
    // Update shared variables to reflect executed join
    trainEpisode.sharedVariables = this.updateSharedVariables(trainEpisode.sharedVariables, idx);
    
    return {
      result: await this.mediatorJoin.mediate({
        type: action.type,
        entries,
        context: action.context,
      }),
    };
  }

  protected async getJoinCoefficients(
    action: IActionRdfJoin,
    metadatas: MetadataBindings[],
  ): Promise<IMediatorTypeReinforcementLearning> {
    // If we have leq 2 entries in our action then we are either: At first execution, last execution or we don't call this actor. To ensure the actor initialises and
    // resets properly we have to initialise sharedVariables

    const modelTreeLSTM: ModelTreeLSTM = (action.context.get(KeysRlTrain.modelInstance) as InstanceModel).getModel();
    const bestOutput = tf.tidy(():[tf.Tensor, tf.Tensor, tf.Tensor, number[]]=>{
      // Get possible join indexes, this disregards order of joins, is that a good assumption for for example hash join?
      const trainEpisode: ITrainEpisode = action.context.get(KeysRlTrain.trainEpisode)!;
    
      const possibleJoinIndexes = this.getPossibleJoins(action.entries.length, trainEpisode.sharedVariables);
      const possibelJoinIndexesUnSorted = [...possibleJoinIndexes];

      const qValuesEst: number[] = [];
      // All elements in these two lists should be disposed
      const qValuesEstTensor: tf.Tensor[] = [];
      const featureRepresentations: ISingleResultSetRepresentation[]=[];
      for (const joinCombination of possibleJoinIndexes){
  
        // Clone features to make our prediction
        const clonedFeatures: IResultSetRepresentation = {hiddenStates: trainEpisode.featureTensor.hiddenStates.map(x=>tf.clone(x)),
        memoryCell: trainEpisode.featureTensor.memoryCell.map(x=>tf.clone(x))};

        // const joinCombinationUnsorted: number[] = [...joinCombination];
        const modelOutput: IModelOutput = modelTreeLSTM.forwardPass(clonedFeatures, joinCombination);

        // Negative of Q-value so we minimize execution time
        const qValueScalar: tf.Scalar = tf.sub(0, tf.squeeze(modelOutput.qValue));
        if (qValueScalar.dataSync().length>1){
          throw new Error("Non scalar returned as Q-value");
        }

        qValuesEst.push(qValueScalar.dataSync()[0]);
        qValuesEstTensor.push(qValueScalar);
        featureRepresentations.push({hiddenState: modelOutput.hiddenState, memoryCell: modelOutput.memoryCell})

      }
      // Choose according to softmaxed probabilities
      const estProb = tf.softmax(tf.stack(qValuesEstTensor)).dataSync() as Float32Array;
      let chosenIdx = 0;
      // Interesting exploration idea: https://arxiv.org/pdf/1612.05628.pdf
      if(action.context.get(KeysRlTrain.train)){
        chosenIdx = this.chooseUsingProbability(estProb);
      }
      else{
        chosenIdx = this.indexOfMax(estProb);
      }

      return [qValuesEstTensor[chosenIdx], featureRepresentations[chosenIdx].hiddenState, 
      featureRepresentations[chosenIdx].memoryCell, possibelJoinIndexesUnSorted[chosenIdx]]
    });

    const featureRepresentation: ISingleResultSetRepresentation = {hiddenState: bestOutput[1], memoryCell: bestOutput[2]};
    return {iterations: 0, persistedItems: 0, blockingItems: 0, requestTime:0,
      predictedQValue: bestOutput[0],
      representation: featureRepresentation,
      joinIndexes: bestOutput[3]
    }
  }
}
export interface ISingleResultSetRepresentation {
  hiddenState: tf.Tensor;
  memoryCell: tf.Tensor;
}

export interface IActorRdfJoinInnerMultiReinforcementLearningTree extends IActorRdfJoinArgs {
  /**
   * The join entries sort mediator
   */
  mediatorJoinEntriesSort: MediatorRdfJoinEntriesSort;

  /**
   * A mediator for joining Bindings streams
   */
  mediatorJoin: MediatorRdfJoin;
}

// export interface IActorRdfJoinMultiSmallestArgs extends IActorRdfJoinArgs {
//     /**
//      * A mediator for joining Bindings streams
//      */
//     mediatorJoin: MediatorRdfJoin;  
// }
