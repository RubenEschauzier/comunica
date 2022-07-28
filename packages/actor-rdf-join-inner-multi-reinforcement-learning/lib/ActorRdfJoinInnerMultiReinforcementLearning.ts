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


/* Debugging info : https://stackoverflow.com/questions/20936486/node-js-maximum-call-stack-size-exceeded
* https://blog.jcoglan.com/2010/08/30/the-potentially-asynchronous-loop/
*/

/**
 * A Multi Smallest RDF Join Actor.
 * It accepts 3 or more streams, joins the smallest two, and joins the result with the remaining streams.
 */
export class ActorRdfJoinInnerMultiReinforcementLearning extends ActorRdfJoin {
  public readonly mediatorJoinEntriesSort: MediatorRdfJoinEntriesSort;
  public readonly mediatorJoin: MediatorRdfJoin;
  public static readonly FACTORY = new Factory();
  
  /* Online train indicates if the model should retrain after each query execution*/
  private onlineTrain: boolean;

  /* Offline train indicates if the model is being pretrained, this involves a different method of query execution and episode generation*/
  private offlineTrain: boolean;
  public joinState: StateSpaceTree;
  private model: graphConvolutionModel;
  numCalls: number;
  test: any;

  prevEstimatedCost: number;
  /* Node id to entries index mapping*/
  nodeid_mapping: Map<number,number>;
  nodeid_mapping_test: Map<number,number>;

  public constructor(args: IActorRdfJoinInnerMultiReinforcementLearningArgs) {
    super(args, {
      logicalType: 'inner',
      physicalName: 'multi-reinforcement-learning',
      limitEntries: 3,
      limitEntriesMin: true,
    });
    this.onlineTrain = args['onlineTrain'];
    this.offlineTrain = args['offlineTrain']
    this.model = new graphConvolutionModel();
    this.model.loadModel();
    this.nodeid_mapping = new Map<number, number>();
  }

  private initialiseStates(action: IActionRdfJoin, metadatas: MetadataBindings[]): void {
    /**
     * Initialise any internal states that need to be initialised.
     * @param {IActionRdfJoin} action An array containing information on the joins
     * @param {MetadataBindings[]} metadatas An array with all  metadatas of the bindingstreams
     * @returns {Promise<void>} Changes only internal states
     */
    if (!this.joinState || this.joinState.isEmpty()){
      this.joinState = new StateSpaceTree();
      for (const [i, metadata] of metadatas.entries()){
        // Create nodes with estimated cardinality as feature
        let newNode: NodeStateSpace = new NodeStateSpace(i, metadata.cardinality.value);
        newNode.setDepth(0);
        this.joinState.addLeave(newNode);
      }
      /* Set the number of leave nodes in the tree, this will not change during execution */
      this.joinState.setNumLeaveNodes(this.joinState.numNodes); 
      action.context.setEpisodeState(this.joinState);
    }

    if (this.nodeid_mapping.size == 0){
        for(let i:number=0;i<action.entries.length;i++){
          this.nodeid_mapping.set(i,i);
        }
    }
  }

  public softMax(logits: tf.Tensor): tf.Tensor{
    return tf.div(tf.exp(logits), tf.sum(tf.exp(logits), undefined, false));
  }

  public chooseUsingProbability(probArray: Float32Array): number{
    const cumsum = (arr: Float32Array) => arr.map((sum => value => sum += value)(0));

    const cumProb: Float32Array = cumsum(probArray);
    const pick: number = Math.random();

    for (let i=0;i<cumProb.length;i++){
      /* If we land at our draw, we return the index of the draw*/
      if(pick > cumProb[i]){
      }
      else{
        return i;
      }
    }
    return cumProb.length;

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
   if (!this.offlineTrain){
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
    
    if (this.onlineTrain){
      // const joinPlanKey: ActionContextKey<number[]> = KeysRdfJoinReinforcementLearning.queryJoinPlan;
      // const previousJoins: number[][] = action.context.get(joinPlanKey)!;
      // let featureMatrix = [];
      // for(var i=0;i<this.joinState.nodesArray.length;i++){
      //   featureMatrix.push(this.joinState.nodesArray[i].cardinality);
      // }
      action.context.setEpisodeState(this.joinState);
    }
    /* If we're at the last time this is executed */
    if (this.joinState.numUnjoined == 2){
      this.nodeid_mapping = new Map<number, number>();
    }

    return {
      result: await this.mediatorJoin.mediate({
        type: action.type,
        entries,
        context: action.context,
      }),
      physicalPlanMetadata: {
        smallest: [ toBeJoined1, toBeJoined2 ],
      },
    }
  }
  else{
    /* Code for the offline generation of joins based the monte carlo tree search algorithm*/

    /* Temp code to make it run */
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
    
    if (this.onlineTrain){
      // const joinPlanKey: ActionContextKey<number[]> = KeysRdfJoinReinforcementLearning.queryJoinPlan;
      // const previousJoins: number[][] = action.context.get(joinPlanKey)!;
      // let featureMatrix = [];
      // for(var i=0;i<this.joinState.nodesArray.length;i++){
      //   featureMatrix.push(this.joinState.nodesArray[i].cardinality);
      // }
      action.context.setEpisodeState(this.joinState);
    }
    /* If we're at the last time this is executed */
    if (this.joinState.numUnjoined == 2){
      this.nodeid_mapping = new Map<number, number>();
    }

    return {
      result: await this.mediatorJoin.mediate({
        type: action.type,
        entries,
        context: action.context,
      }),
      physicalPlanMetadata: {
        smallest: [ toBeJoined1, toBeJoined2 ],
      },
    }
  }
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

    /* If we have no joinState or nodeid_mapping due to resets we initialise empty ones */

  if (!this.offlineTrain){
    /* Check if we need to initialise the joinstate / nodeidmapping */
    if (!this.joinState || this.joinState.isEmpty() || this.nodeid_mapping.size == 0){
      this.initialiseStates(action, metadatas);
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
      const forwardPassOutput = this.model.forwardPass(tf.tensor2d(featureMatrix, [this.joinState.numNodes,1]), adjTensor) as tf.Tensor[];

      /* Add estimated value to the array */
      /* THIS IS TEMP THIS SHOULD INCLUDE A POLICY ESTIMATION WITH PROBABILITIES TO IMPROVE TRAINING */
      if (forwardPassOutput instanceof tf.Tensor){
        const outputArray= await forwardPassOutput.data();
        if (outputArray[outputArray.length-1]<bestJoinCost){
          bestJoin = joinCombination;
          bestJoinCost = outputArray[outputArray.length-1];
        }
      }

      this.joinState.removeLastAdded();
    }
    /* Select join to explore based on probabilty obtained by softmax of the estimated join values*/
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
   }
   else{
    return this.getJoinCoefficientsOffline(action, metadatas)
   }

    /* We return 0 for request time, not sure what I should be doing there */
  }

  private async getJoinCoefficientsOffline(action: IActionRdfJoin,
    metadatas: MetadataBindings[],
  ): Promise<IMediatorTypeJoinCoefficients>
  {
    console.log(`BEFORE: NumNodes: ${action.context.getEpisodeState().numNodes}, Context ID: ${action.context.getEpisodeState().testId}`);
    /**
     * Execute the exploration phase of the offline training algorithm. We find
     */
    let nodeIndexes: number[] = [... Array(action.context.getEpisodeState().numNodes).keys()];

    /*  Remove already joined nodes from consideration by traversing the node indices in reverse order 
        which does not disturb the indexing of the array  */
    for (let i:number=action.context.getEpisodeState().numNodes-1;i>=0;i--){
      if (action.context.getEpisodeState().nodesArray[i].joined==true){
        nodeIndexes.splice(i,1);
      }
    }
    /*  Generate possible joinState combinations  */
    const possibleJoins: number[][] = this.getPossibleJoins(nodeIndexes);
    const joinValues: number[] = [];

    for (const joinCombination of possibleJoins){
      /*  We generate the join trees associated with the possible join combinations, note the tradeoff between memory usage (create new trees) 
          and cpu usage (delete the new node)
      */
      const newParent: NodeStateSpace = new NodeStateSpace(action.context.getEpisodeState().numNodes, 0);
      action.context.getEpisodeState().addParent(joinCombination, newParent);
      const adjTensor: tf.Tensor2D = tf.tensor2d(action.context.getEpisodeState().adjacencyMatrix);

      let featureMatrix: number[] = [];
      for(var i=0;i<action.context.getEpisodeState().nodesArray.length;i++){
        featureMatrix.push(action.context.getEpisodeState().nodesArray[i].cardinality);
      }

      /*  Scale features using Min-Max scaling  */
      // const maxCardinality: number = Math.max(...featureMatrix);
      // const minCardinality: number = Math.min(...featureMatrix);
      // featureMatrix = featureMatrix.map((vM, iM) => (vM - minCardinality) / (maxCardinality - minCardinality));

      /* Execute forward pass */
      const forwardPassOutput: tf.Tensor[] = this.model.forwardPass(tf.tensor2d(featureMatrix, [action.context.getEpisodeState().numNodes,1]), adjTensor) as tf.Tensor[];
      // const forwardPassOutput = this.model.forwardPass(tf.tensor2d(featureMatrix, [action.context.getEpisodeState().numNodes,1]), adjTensor) as tf.Tensor
      const valueOutput: tf.Tensor = forwardPassOutput[0];
      const policyOutput: tf.Tensor = forwardPassOutput[1];

      const outputArrayValue: Float32Array | Int32Array | Uint8Array = await valueOutput.data();
      const outputArrayPolicy: Float32Array | Int32Array | Uint8Array = await policyOutput.data();
      // const outputArrayValue = await forwardPassOutput.data()

      joinValues.push(outputArrayValue[outputArrayValue.length-1])

      action.context.getEpisodeState().removeLastAdded();

    }
    const predictedValueTensor: tf.Tensor = tf.tensor(joinValues);
    const predictedValueArray: Float32Array = await predictedValueTensor.data() as Float32Array
    const probabilities: Float32Array = await tf.softmax(predictedValueTensor).data() as Float32Array;
    const idx = this.chooseUsingProbability(probabilities);
    
    /* Add new explored node to the statespace*/
    const newParent: NodeStateSpace = new NodeStateSpace(action.context.getEpisodeState().numNodes, predictedValueArray[idx]);
    action.context.getEpisodeState().addParent(possibleJoins[idx], newParent);
    this.prevEstimatedCost = predictedValueArray[idx];

    console.log(`AFTER: NumNodes: ${action.context.getEpisodeState().numNodes}, Context ID: ${action.context.getEpisodeState().testId}`);

    return {
      iterations: 10,
      persistedItems: 0,
      blockingItems: 0,
      requestTime: 0,
    };
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
   * Online train indicates if the model should retrain after each query execution
   */
  onlineTrain: boolean;
  /**
   * Offline train indicates if the model is being pretrained, this involves a different method of query execution and episode generation
   */
  offlineTrain: boolean;
}
