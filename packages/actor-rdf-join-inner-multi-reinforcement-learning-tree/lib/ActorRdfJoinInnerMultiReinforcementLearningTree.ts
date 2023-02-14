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

/**
 * A Multi Smallest RDF Join Actor.
 * It accepts 3 or more streams, joins the smallest two, and joins the result with the remaining streams.
 */
export class ActorRdfJoinInnerMultiReinforcementLearningTree extends ActorRdfJoin implements IActorRdfJoinInnerMultiReinforcementLearningTree {
  // public readonly mediatorJoinEntriesSort: MediatorRdfJoinEntriesSort;
  public readonly mediatorJoin: MediatorRdfJoin;

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

  /**
   * Order the given join entries using the join-entries-sort bus.
   * @param {IJoinEntryWithMetadata[]} entries An array of join entries.
   * @param context The action context.
   * @return {IJoinEntryWithMetadata[]} The sorted join entries.
  */

  public getPossibleJoins(numJoinable: number){
    let orders: number[][] = [];
    for (let i=0; i<numJoinable;i++){
      for (let j=i+1; j<numJoinable;j++){
        orders.push([i,j]);
      }
    }
    return orders;
  }

  public chooseUsingProbability(probArray: Float32Array|number[]): number{
    // const cumsum = (arr: Float32Array|number[]) => arr.map((sum => value => sum += value)(0));
    let cumSumNew = [];
    let runningSum = 0;
    for (let j=0; j<probArray.length; j++){
      runningSum+= probArray[j];
      cumSumNew.push(runningSum);
    }

    // const cumProb: Float32Array|number[] = cumsum(probArray);

    const pick: number = Math.random();

    for (let i=0;i<cumSumNew.length;i++){
      /* If we land at our draw, we return the index of the draw*/
      if(pick > cumSumNew[i]){
      }
      else{
        return i;
      }
    }
    return cumSumNew.length-1;

  }

  protected async getOutput(action: IActionRdfJoinReinforcementLearning): Promise<IActorRdfJoinOutputInner> {
    // Determine the two smallest streams by sorting (e.g. via cardinality)
    const entries = [...action.entries];
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
    const firstEntry: IJoinEntry = {
      output: ActorQueryOperation.getSafeBindings(await this.mediatorJoin
        .mediate({ type: action.type, entries: [ joinEntry1, joinEntry2 ], context: action.context })),
      operation: ActorRdfJoinInnerMultiReinforcementLearningTree.FACTORY
        .createJoin([ joinEntry1.operation, joinEntry2.operation ], false),
    };

    // Important for model execution that new entry is always entered last
    entries.push(firstEntry);
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
    const startTensors: number = tf.memory().numTensors;
    const modelTreeLSTM: ModelTreeLSTM = (action.context.get(KeysRlTrain.modelInstance) as InstanceModel).getModel();

    const bestOutput = tf.tidy(():[tf.Tensor, tf.Tensor, tf.Tensor, number[]]=>{
      // Get possible join indexes, this disregards order of joins, is that a good assumption for for example hash join?
      const trainEpisode: ITrainEpisode = action.context.get(KeysRlTrain.trainEpisode)!;
    
      const possibleJoinIndexes = this.getPossibleJoins(action.entries.length);
      const possibelJoinIndexesUnSorted = [...possibleJoinIndexes];

      const qValuesEst: number[] = [];
      // All elements in these two lists should be disposed
      const qValuesEstTensor: tf.Tensor[] = [];
      const featureRepresentations: ISingleResultSetRepresentation[]=[];

      for (const joinCombination of possibleJoinIndexes){
  
        // Clone features to make our prediction
        const clonedFeatures: IResultSetRepresentation = {hiddenStates: trainEpisode.featureTensor.hiddenStates.map(x=>tf.clone(x)),
        memoryCell: trainEpisode.featureTensor.memoryCell.map(x=>tf.clone(x))};

        const joinCombinationUnsorted: number[] = [...joinCombination];
        const modelOutput: IModelOutput = modelTreeLSTM.forwardPass(clonedFeatures, joinCombination);
        
        const qValueScalar: tf.Scalar = tf.squeeze(modelOutput.qValue);
        if (qValueScalar.dataSync().length>1){
          throw new Error("Non scalar returned as Q-value");
        }
        qValuesEst.push(qValueScalar.dataSync()[0]);
        qValuesEstTensor.push(qValueScalar);
        featureRepresentations.push({hiddenState: modelOutput.hiddenState, memoryCell: modelOutput.memoryCell})

      }
      // Choose according to softmaxed probabilities
      const estProb = tf.softmax(tf.stack(qValuesEstTensor)).dataSync() as Float32Array;
      const chosenIdx = this.chooseUsingProbability(estProb);

      // Clone all tensors belonging to best join
      // const bestQTensor: tf.Tensor = tf.clone(qValuesEstTensor[chosenIdx]);
      // const bestFeatureRep: ISingleResultSetRepresentation = {hiddenState: tf.clone(featureRepresentations[chosenIdx].hiddenState), 
      // memoryCell: tf.clone(featureRepresentations[chosenIdx].memoryCell)};
      // const bestJoinIdx: number[] = possibelJoinIndexesUnSorted[chosenIdx];

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
   * A mediator for joining Bindings streams
   */
  mediatorJoin: MediatorRdfJoin;
}

export interface IActorRdfJoinMultiSmallestArgs extends IActorRdfJoinArgs {
    /**
   * The join entries sort mediator
   */
    mediatorJoinEntriesSort: MediatorRdfJoinEntriesSort;
    /**
     * A mediator for joining Bindings streams
     */
    mediatorJoin: MediatorRdfJoin;  
}
