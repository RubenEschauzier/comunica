import {
  ActorQueryOperation,
} from '@comunica/bus-query-operation';
import type {
  IActionRdfJoin,
  IActorRdfJoinOutputInner,
  IActorRdfJoinArgs,
  MediatorRdfJoin,
} from '@comunica/bus-rdf-join';
import { ActorRdfJoin} from '@comunica/bus-rdf-join';
import type { MediatorRdfJoinEntriesSort } from '@comunica/bus-rdf-join-entries-sort';
import type { IMediatorTypeJoinCoefficients } from '@comunica/mediatortype-join-coefficients';
import type { MetadataBindings, IJoinEntry, IActionContext, IJoinEntryWithMetadata } from '@comunica/types';
import { Factory } from 'sparqlalgebrajs';
import { StateSpaceTree, NodeStateSpace } from '@comunica/mediator-join-reinforcement-learning';
import { graphConvolutionModel } from './GraphNeuralNetwork';
import * as tf from '@tensorflow/tfjs';
import {KeysRdfJoinReinforcementLearning} from '@comunica/context-entries'
import { ActionContextKey } from '@comunica/core';

/**
 * A Multi Smallest RDF Join Actor.
 * It accepts 3 or more streams, joins the smallest two, and joins the result with the remaining streams.
 */
export class ActorRdfJoinInnerMultiReinforcementLearning extends ActorRdfJoin {
  public readonly mediatorJoinEntriesSort: MediatorRdfJoinEntriesSort;
  public readonly mediatorJoin: MediatorRdfJoin;
  public static readonly FACTORY = new Factory();
  
  private train: boolean;
  public joinState: StateSpaceTree;
  private model: graphConvolutionModel;
  numCalls: number;
  test: any;

  prevEstimatedCost: number;
  /* Node id to entries index mapping*/
  nodeid_mapping: Map<number,number>;

  public constructor(args: IActorRdfJoinInnerMultiReinforcementLearningArgs) {
    super(args, {
      logicalType: 'inner',
      physicalName: 'multi-reinforcement-learning',
      limitEntries: 3,
      limitEntriesMin: true,
    });
    this.train = args['trainingMode'];
    this.model = new graphConvolutionModel();
    this.model.loadModel();
    this.nodeid_mapping = new Map<number, number>();
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


  public getPossibleJoins(nodesJoinable: number[]){
    let orders: number[][] = [];
    for (const [arrIndex, nodeIndex] of nodesJoinable.entries()){
      for(const secondNodeIndex of nodesJoinable.slice(arrIndex+1)){
        orders.push([nodeIndex, secondNodeIndex]);
      }
    }
    return orders;
  }
  

  protected async getOutput(action: IActionRdfJoin): Promise<IActorRdfJoinOutputInner> {
    /*Create initial mapping between entries and nodes in the state tree. We need this mapping due to the indices of nodes increasing as more joins are made, while the
      number of streams decreases with more joins. 
    */
    if (this.nodeid_mapping.size == 0){
      for(let i:number=0;i<action.entries.length;i++){
        this.nodeid_mapping.set(i,i);
      }
    }

    const entries: IJoinEntry[] = action.entries

    /* Find node ids from to be joined streams*/
    const joined_nodes: NodeStateSpace[] = this.joinState.nodesArray[this.joinState.numNodes-1].children;
    const ids: number[] = joined_nodes.map(a => a.id).sort();
    const indexJoins: number[] = ids.map(a => this.nodeid_mapping.get(a)!).sort();

    const toBeJoined1 = entries[indexJoins[0]];
    const toBeJoined2 = entries[indexJoins[1]];

    entries.splice(indexJoins[1],1); entries.splice(indexJoins[0],1);
    const firstEntry: IJoinEntry = {
      output: ActorQueryOperation.getSafeBindings(await this.mediatorJoin
        .mediate({ type: action.type, entries: [ toBeJoined1, toBeJoined2 ], context: action.context })),
      operation: ActorRdfJoinInnerMultiReinforcementLearning.FACTORY
        .createJoin([ toBeJoined1.operation, toBeJoined2.operation ], false),
    };
    entries.push(firstEntry);



    /* Update the mapping to reflect the newly executed join*/
    /* Find maximum and minimum index in the join entries array to update the mapping */
    const max_idx: number = this.nodeid_mapping.get(Math.max(...ids))!;
    const min_idx: number = this.nodeid_mapping.get(Math.min(...ids))!;

    /* Update the mappings by keeping track of the position of each entry in the nodeid mapping and updating according to position in the array*/
    for (let [key, value] of this.nodeid_mapping){
      if (value > max_idx){
        this.nodeid_mapping.set(key, value-2);
      }
      else if(value > min_idx && value < max_idx){
        this.nodeid_mapping.set(key, value-1);
      }
    }
    
    /* Remove joined nodes from the index mapping*/
    this.nodeid_mapping.set(ids[0], -1); this.nodeid_mapping.set(ids[1], -1);

    /* Add the joined entry to the index mapping, here we assume the joined streams are added at the end of the array, need to check this assumption*/
    const join_id: number = this.joinState.nodesArray[this.joinState.numNodes-1].id;
    this.nodeid_mapping.set(join_id, action.entries.length-1);
    if(this.train){
      const joinPlanKey: ActionContextKey<number[]> = KeysRdfJoinReinforcementLearning.queryJoinPlan;
      // const previousJoins: number[][] = action.context.get(joinPlanKey)!;

      action.context.setEpisodeState(this.joinState);
    }
    return {
      result: await this.mediatorJoin.mediate({
        type: action.type,
        entries,
        context: action.context,
      }),
      physicalPlanMetadata: {
        selected: [ toBeJoined1, toBeJoined2 ],
      },
    };
  }

  protected async getJoinCoefficients(
    action: IActionRdfJoin,
    metadatas: MetadataBindings[],
  ): Promise<IMediatorTypeJoinCoefficients> {

    /**
     * Execute the RL agent on all possible joins from the current join state and return the best join state.
     * @param {IActionRdfJoin} action An array containing information on the joins
     * @param {MetadataBindings[]} metadatas An array with all  metadatas of the bindingstreams
     * @returns {Promise<IMediatorTypeJoinCoefficients>} Interface with estimated cost of the best join
    */
    console.log("Getting joincoefficients");
    if (!this.joinState){
      const metadatas: MetadataBindings[] = await ActorRdfJoinInnerMultiReinforcementLearning.getMetadatas(action.entries);
      this.joinState = new StateSpaceTree();
      for (const [i, metadata] of metadatas.entries()){
        // Create nodes with estimated cardinality as feature
        let newNode: NodeStateSpace = new NodeStateSpace(i, metadata.cardinality.value);
        newNode.setDepth(0);
        this.joinState.addLeave(newNode);
      }
      /* Set the number of leave nodes in the tree, this will not change during execution */
      this.joinState.setNumLeaveNodes(this.joinState.numNodes); 
    }

    let nodeIndexes: number[] = [... Array(this.joinState.numNodes).keys()];

    /*  Remove already joined nodes from consideration by traversing the node indices in reverse order 
        which does not disturb the indexing of the array  */
    for (let i:number=this.joinState.numNodes-1;i>=0;i--){
      if (this.joinState.nodesArray[i].joined==true){
        nodeIndexes.splice(i,1);
      }
    }
    /*  Generate possible joinState combinations  */
    const possibleJoins: number[][] = this.getPossibleJoins(nodeIndexes);
    let bestJoinCost: number = Infinity;
    let bestJoin: number[] = [-1, -1];

    for (const joinCombination of possibleJoins){
      /*  We generate the join trees associated with the possible join combinations, note the tradeoff between memory usage (create new trees) 
          and cpu usage (delete the new node)
      */
      const newParent: NodeStateSpace = new NodeStateSpace(this.joinState.numNodes, 0);
      this.joinState.addParent(joinCombination, newParent);
      const adjTensor = tf.tensor2d(this.joinState.adjacencyMatrix);

      let featureMatrix = [];
      for(var i=0;i<this.joinState.nodesArray.length;i++){
        featureMatrix.push(this.joinState.nodesArray[i].cardinality);
      }

      /*  Scale features using Min-Max scaling  */
      // const maxCardinality: number = Math.max(...featureMatrix);
      // const minCardinality: number = Math.min(...featureMatrix);
      // featureMatrix = featureMatrix.map((vM, iM) => (vM - minCardinality) / (maxCardinality - minCardinality));

      /* Execute forward pass */
      const forwardPassOutput = this.model.forwardPass(tf.tensor2d(featureMatrix, [this.joinState.numNodes,1]), adjTensor);
      // Casten naar tensor <tf.Tensor> en miss <any>
      if (forwardPassOutput instanceof tf.Tensor){
        const outputArray= await forwardPassOutput.data();
        if (outputArray[outputArray.length-1]<bestJoinCost){
          bestJoin = joinCombination;
          bestJoinCost = outputArray[outputArray.length-1];
        }
      }
      this.joinState.removeLastAdded();
    }
    
    /*  Update the actors joinstate, which will be used by the RL medaitor to update its joinstate too  */
    const newParent: NodeStateSpace = new NodeStateSpace(this.joinState.numNodes, bestJoinCost);
    this.joinState.addParent(bestJoin, newParent);    
    this.prevEstimatedCost = bestJoinCost;

    return {
      iterations: bestJoinCost,
      persistedItems: 0,
      blockingItems: 0,
      requestTime: 0,
    };

    /* We return 0 for request time, not sure what I should be doing there */
  }

  updateState(newState: StateSpaceTree){
    this.joinState = newState;
  }
}

export interface IActorRdfJoinInnerMultiReinforcementLearningArgs extends IActorRdfJoinArgs {
  /**
   * The join entries sort mediator
   */
  mediatorJoinEntriesSort: MediatorRdfJoinEntriesSort;
  /**
   * A mediator for joining Bindings streams
   */
  mediatorJoin: MediatorRdfJoin;
  /**
   * 
   */
  trainingMode: boolean;
}
