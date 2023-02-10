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
    const modelTreeLSTM: ModelTreeLSTM = (action.context.get(KeysRlTrain.modelInstance) as InstanceModel).getModel();

    // const trainEpisodeTest: ITrainEpisode = action.context.get(KeysRlTrain.trainEpisode)!;
    // console.log("Last feature before model execution");
    // console.log(trainEpisodeTest.featureTensor.hiddenStates[trainEpisodeTest.featureTensor.hiddenStates.length-1].dataSync().slice(0,10));

    const bestOutput = tf.tidy(():[tf.Tensor, tf.Tensor, tf.Tensor, number[]]=>{
      // Get possible join indexes, this disregards order of joins, is that a good assumption for for example hash join?
      const trainEpisode: ITrainEpisode = action.context.get(KeysRlTrain.trainEpisode)!;
    
      const possibleJoinIndexes = this.getPossibleJoins(action.entries.length);
      let bestQValue: number = -Infinity;
      let bestJoinIdx: number[] = [];
  
      let bestQTensor: tf.Tensor = tf.tensor([]);
      let bestFeatureRep: ISingleResultSetRepresentation = {hiddenState: tf.tensor([]), memoryCell: tf.tensor([])};
      // let bestFeatureRep: tf.Tensor[] = [];
      // console.log("In actor:");
      // console.log(trainEpisode.featureTensor.hiddenStates.length);
      // console.log(trainEpisode.featureTensor.memoryCell.length)
      // console.log(trainEpisode.featureTensor.hiddenStates[trainEpisode.featureTensor.hiddenStates.length-1].dataSync())
  
      for (const joinCombination of possibleJoinIndexes){
  
        // Clone features to make our prediction
        const clonedFeatures: IResultSetRepresentation = {hiddenStates: trainEpisode.featureTensor.hiddenStates.map(x=>tf.clone(x)),
        memoryCell: trainEpisode.featureTensor.memoryCell.map(x=>tf.clone(x))};
        // console.log(trainEpisode.featureTensor.hiddenStates[trainEpisode.featureTensor.hiddenStates.length-1].dataSync())
        // console.log("INPUT NORMAL FWPASS HiddenState")
        // console.log(clonedFeatures.hiddenStates.map(x=>x.dataSync()))
        const joinCombinationUnsorted: number[] = [...joinCombination];
        const modelOutput: IModelOutput = modelTreeLSTM.forwardPass(clonedFeatures, joinCombination);
        // console.log("Here join combi normal execution");
        // console.log(joinCombinationUnsorted);

        // // This should be a parameter:
        // const train=true;
        // if (train){
        //   const grads = tf.grads(this.modelTreeLSTM.forwardPassGradsTest);
        //   grads([clonedFeatures.hiddenStates, clonedFeatures.memoryCell]);
        // }
        
        const qValueScalar: tf.Scalar = tf.squeeze(modelOutput.qValue);
        if (qValueScalar.dataSync().length>1){
          throw new Error("Non scalar returned as Q-value");
        }
  
        if (qValueScalar.dataSync()[0]>bestQValue){
          bestQValue = qValueScalar.dataSync()[0];
          bestJoinIdx = joinCombinationUnsorted;

          // Select new optimal tensors
          bestQTensor = modelOutput.qValue;
          bestFeatureRep.hiddenState = modelOutput.hiddenState; bestFeatureRep.memoryCell = modelOutput.memoryCell;
        }
      }
      return [bestQTensor, bestFeatureRep.hiddenState, bestFeatureRep.memoryCell, bestJoinIdx]
    });
    // console.log(`Chosen idx: ${bestOutput[3]}, forward pass normal`)
    // console.log(bestOutput[0].dataSync());

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
