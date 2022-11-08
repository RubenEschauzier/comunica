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
import {ActionContextKey, CONTEXT_KEY_LOGGER} from '@comunica/core'
import {KeysQueryOperation} from '@comunica/context-entries'
import * as RdfJs from 'rdf-js'
// import { graphConvolutionModel } from './GraphNeuralNetwork';
import * as tf from '@tensorflow/tfjs';
import { aggregateValues, MCTSJoinInformation, MCTSJoinPredictionOutput, MCTSMasterTree, modelHolder, runningMoments } from '@comunica/model-trainer';
import * as _ from 'lodash'
import { Variable } from 'rdf-data-factory';
import { getContextSourceUrl } from '@comunica/bus-rdf-resolve-quad-pattern';
import { FederatedQuadSource } from '@comunica/actor-rdf-resolve-quad-pattern-federated';

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
  // private model: graphConvolutionModel;
  

  prevEstimatedCost: number;
  explorationDegree: number;
  predicateEmbeddingSize: number;
  /* Node id to entries index mapping*/
  nodeid_mapping: Map<number, number>;
  nodeid_mapping_test: Map<number, number>;

  public constructor(args: IActorRdfJoinInnerMultiReinforcementLearningArgs) {
    super(args, {
      logicalType: 'inner',
      physicalName: 'multi-reinforcement-learning',
      limitEntries: 3,
      limitEntriesMin: true,
    });
    this.onlineTrain = args['onlineTrain'];
    this.offlineTrain = args['offlineTrain']
    this.explorationDegree = args['explorationDegree'];
    this.nodeid_mapping = new Map<number, number>();
    this.predicateEmbeddingSize = 128;
  }

  // private initialiseStates(action: IActionRdfJoin, metadatas: MetadataBindings[]): void {
  //   /**
  //    * Initialise any internal states that need to be initialised.
  //    * @param {IActionRdfJoin} action An array containing information on the joins
  //    * @param {MetadataBindings[]} metadatas An array with all  metadatas of the bindingstreams
  //    * @returns {Promise<void>} Changes only internal states
  //    */
  //   if (!this.joinState || this.joinState.isEmpty()){
  //     this.joinState = new StateSpaceTree();
  //     for (const [i, metadata] of metadatas.entries()){
  //       // Create nodes with estimated cardinality as feature
  //       let newNode: NodeStateSpace = new NodeStateSpace(i, metadata.cardinality.value);
  //       newNode.setDepth(0);
  //       this.joinState.addLeave(newNode);
  //     }
  //     /* Set the number of leave nodes in the tree, this will not change during execution */
  //     this.joinState.setNumLeaveNodes(this.joinState.numNodes); 
  //     action.context.setEpisodeState(this.joinState);
  //   }

  //   if (this.nodeid_mapping.size == 0){
  //       for(let i:number=0;i<action.entries.length;i++){
  //         this.nodeid_mapping.set(i,i);
  //       }
  //   }
  // }

  public softMax(logits: tf.Tensor): tf.Tensor{
    return tf.tidy(()=>{
      return tf.div(tf.exp(logits), tf.sum(tf.exp(logits), undefined, false));
    })
  }

  public maxPool(array1: number[], array2: number[]){
    const maxPooled = []
    for (let i=0;i<array1.length;i++){
      maxPooled.push(Math.max(array1[i], array2[i]))
    }
    return maxPooled;
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

  public getPossibleJoins(nodesJoinable: number[]){
    let orders: number[][] = [];
    for (const [arrIndex, nodeIndex] of nodesJoinable.entries()){
      for(const secondNodeIndex of nodesJoinable.slice(arrIndex+1)){
        orders.push([nodeIndex, secondNodeIndex]); 
      }
    }
    return orders;
  }
  public arrayEqual(a: number[], b: number[]){

  }

  protected async getOutput(action: IActionRdfJoin): Promise<IActorRdfJoinOutputInner> {
    /*Create initial mapping between entries and nodes in the state tree. We need this mapping due to the indices of nodes increasing as more joins are made, while the
      number of streams decreases with more joins. 
    */
    const passedNodeIdMapping = action.context.getNodeIdMapping();
    const entries: IJoinEntry[] = action.entries

    /* Find node ids from to be joined streams */
    const joined_nodes: NodeStateSpace[] = action.context.getEpisodeState().nodesArray[action.context.getEpisodeState().numNodes-1].children;
    const ids: number[] = joined_nodes.map(a => a.id).sort();
    // const indexJoins: number[] = ids.map(a => passedNodeIdMapping.get(a)!).sort();

    /* Get the join indexes determined in the exploration phase of the algorithm*/
    const indexJoins: number[] = action.context.getEpisodeState().joinIndexes[action.context.getEpisodeState().joinIndexes.length-1];

    const toBeJoined1 = entries[indexJoins[0]];
    const toBeJoined2 = entries[indexJoins[1]];

    entries.splice(indexJoins[1],1); entries.splice(indexJoins[0],1);

    /* Save the join indices */
    // action.context.addEpisodeStateJoinIndexes(indexJoins);
    // console.log(action.context.get)
    const firstEntry: IJoinEntry = {
      output: ActorQueryOperation.getSafeBindings(await this.mediatorJoin
        .mediate({ type: action.type, entries: [ toBeJoined1, toBeJoined2 ], context: action.context })),
      operation: ActorRdfJoinInnerMultiReinforcementLearning.FACTORY
        .createJoin([ toBeJoined1.operation, toBeJoined2.operation ], false),
    };
    entries.push(firstEntry);


    /* Update the mapping to reflect the newly executed join*/
    /* Find maximum and minimum index in the join entries array to update the mapping */
    const max_idx: number = passedNodeIdMapping.get(Math.max(...ids))!;
    const min_idx: number = passedNodeIdMapping.get(Math.min(...ids))!;

    /* Update the mappings by keeping track of the position of each entry in the nodeid mapping and updating according to position in the array*/
    for (let [key, value] of action.context.getNodeIdMapping()){
      if (value > max_idx){
        action.context.setNodeIdMapping(key, value-2);
      }
      else if(value > min_idx && value < max_idx){
        action.context.setNodeIdMapping(key, value-1);
      }
    }
    // Here we try to use a reduced form of cardinality for feature, since the comunica cardinality explodes when selectivity is small
    const selectivity: number = (await this.mediatorJoinSelectivity.mediate({entries: [toBeJoined1,toBeJoined2], context: action.context})).selectivity;
    const testCardSmaller: number = (joined_nodes[0].cardinality +joined_nodes[1].cardinality)*selectivity;
    const joinedPredicateEmbedding = this.maxPool(joined_nodes[0].predicateEmbedding, joined_nodes[1].predicateEmbedding);
    // console.log('New embedding')
    // console.dir(joinedPredicateEmbedding, {'maxArrayLength': null});


    // Add cardinality of the new join to current state, 
    const numNodes = action.context.getEpisodeState().numNodes
    const featuresLastNode: number[] = action.context.getEpisodeState().nodesArray[numNodes-1].features;
    const addedJoinMetaData: MetadataBindings = await entries[entries.length-1].output.metadata();
    // const newJoinCardinality: number = addedJoinMetaData.cardinality.value;
    action.context.getEpisodeState().nodesArray[numNodes-1].cardinality = testCardSmaller;
    featuresLastNode[0] = testCardSmaller;
    featuresLastNode[7] = addedJoinMetaData.variables.length;
    const first8: number[] = featuresLastNode.slice(0,8);
    const newFeatures: number[] = [...first8, ...joinedPredicateEmbedding]
    action.context.getEpisodeState().nodesArray[numNodes-1].features = newFeatures;


    // Update the running mean and std
    const indexesToTrack = action.context.getRunningMomentsMasterTree().indexes;
    for (const index of indexesToTrack){
      action.context.updateRunningMomentsMasterTree(newFeatures[index], index);
      const runningMoments = action.context.getRunningMomentsMasterTree().runningStats.get(index);
      if (runningMoments){
        newFeatures[index] = (newFeatures[index] - runningMoments.mean)/runningMoments.std;
      }
      else{
        console.error("Accessing index in running moments that does not exist (actor)");
      }
    }
    // console.dir(newFeatures, {'maxArrayLength': null});

    
    action.context.getJoinStateMasterTree(action.context.getEpisodeState().joinIndexes).featureMatrix[numNodes-1][0] = newFeatures[0];
    // console.log("Feature Matrix in Get Output");
    // console.dir(action.context.getJoinStateMasterTree(action.context.getEpisodeState().joinIndexes).featureMatrix[3], {'maxArrayLength': null});
    // console.log(action.context.getJoinStateMasterTree(action.context.getEpisodeState().joinIndexes).featureMatrix);

    // action.context.setJoinStateMasterTree(predictionOutput, featureMatrix, adjacencyMatrixCopy);
    // action.context.setPlanHolder(newState.flat().toString().replaceAll(',', ''));

    /* Remove joined nodes from the index mapping*/
    action.context.setNodeIdMapping(ids[0], -1); action.context.setNodeIdMapping(ids[1], -1);

    /* Add the joined entry to the index mapping, here we assume the joined streams are added at the end of the array, need to check this assumption*/
    const join_id: number = action.context.getEpisodeState().nodesArray[action.context.getEpisodeState().numNodes-1].id;
    action.context.setNodeIdMapping(join_id, action.entries.length-1);

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

  
    
  //  if (!this.offlineTrain){
  //   const entries: IJoinEntry[] = action.entries

  //   /* Find node ids from to be joined streams*/
  //   const joined_nodes: NodeStateSpace[] = this.joinState.nodesArray[this.joinState.numNodes-1].children;
  //   const ids: number[] = joined_nodes.map(a => a.id).sort();
  //   const indexJoins: number[] = ids.map(a => this.nodeid_mapping.get(a)!).sort();

  //   const toBeJoined1 = entries[indexJoins[0]];
  //   const toBeJoined2 = entries[indexJoins[1]];

  //   entries.splice(indexJoins[1],1); entries.splice(indexJoins[0],1);

  //   const firstEntry: IJoinEntry = {
  //     output: ActorQueryOperation.getSafeBindings(await this.mediatorJoin
  //       .mediate({ type: action.type, entries: [ toBeJoined1, toBeJoined2 ], context: action.context })),
  //     operation: ActorRdfJoinInnerMultiReinforcementLearning.FACTORY
  //       .createJoin([ toBeJoined1.operation, toBeJoined2.operation ], false),
  //   };
  //   entries.push(firstEntry);


  //   /* Update the mapping to reflect the newly executed join*/
  //   /* Find maximum and minimum index in the join entries array to update the mapping */
  //   const max_idx: number = this.nodeid_mapping.get(Math.max(...ids))!;
  //   const min_idx: number = this.nodeid_mapping.get(Math.min(...ids))!;

  //   /* Update the mappings by keeping track of the position of each entry in the nodeid mapping and updating according to position in the array*/
  //   for (let [key, value] of this.nodeid_mapping){
  //     if (value > max_idx){
  //       this.nodeid_mapping.set(key, value-2);
  //     }
  //     else if(value > min_idx && value < max_idx){
  //       this.nodeid_mapping.set(key, value-1);
  //     }
  //   }
    
  //   /* Remove joined nodes from the index mapping*/
  //   this.nodeid_mapping.set(ids[0], -1); this.nodeid_mapping.set(ids[1], -1);

  //   /* Add the joined entry to the index mapping, here we assume the joined streams are added at the end of the array, need to check this assumption*/
  //   const join_id: number = this.joinState.nodesArray[this.joinState.numNodes-1].id;
  //   this.nodeid_mapping.set(join_id, action.entries.length-1);
    
  //   if (this.onlineTrain){
  //     // const joinPlanKey: ActionContextKey<number[]> = KeysRdfJoinReinforcementLearning.queryJoinPlan;
  //     // const previousJoins: number[][] = action.context.get(joinPlanKey)!;
  //     // let featureMatrix = [];
  //     // for(var i=0;i<this.joinState.nodesArray.length;i++){
  //     //   featureMatrix.push(this.joinState.nodesArray[i].cardinality);
  //     // }
  //     action.context.setEpisodeState(this.joinState);
  //   }
  //   /* If we're at the last time this is executed */
  //   if (this.joinState.numUnjoined == 2){
  //     this.nodeid_mapping = new Map<number, number>();
  //   }

  //   return {
  //     result: await this.mediatorJoin.mediate({
  //       type: action.type,
  //       entries,
  //       context: action.context,
  //     }),
  //     physicalPlanMetadata: {
  //       smallest: [ toBeJoined1, toBeJoined2 ],
  //     },
  //   }
  // }
  // else{
  //   /* Code for the offline generation of joins based the monte carlo tree search algorithm*/

  //   /* Temp code to make it run */
  //   if (this.nodeid_mapping.size == 0){
  //     for(let i:number=0;i<action.entries.length;i++){
  //       this.nodeid_mapping.set(i,i);
  //     }
  //   }
  //   const entries: IJoinEntry[] = action.entries

  //   /* Find node ids from to be joined streams*/
  //   const joined_nodes: NodeStateSpace[] = action.context.getEpisodeState().nodesArray[action.context.getEpisodeState().numNodes-1].children;
  //   const ids: number[] = joined_nodes.map(a => a.id).sort();
  //   const indexJoins: number[] = ids.map(a => this.nodeid_mapping.get(a)!).sort();

  //   const toBeJoined1 = entries[indexJoins[0]];
  //   const toBeJoined2 = entries[indexJoins[1]];

  //   entries.splice(indexJoins[1],1); entries.splice(indexJoins[0],1);

  //   /* Save the join indices */
  //   // action.context.addEpisodeStateJoinIndexes(indexJoins);
  //   // console.log(action.context.get)

  //   const firstEntry: IJoinEntry = {
  //     output: ActorQueryOperation.getSafeBindings(await this.mediatorJoin
  //       .mediate({ type: action.type, entries: [ toBeJoined1, toBeJoined2 ], context: action.context })),
  //     operation: ActorRdfJoinInnerMultiReinforcementLearning.FACTORY
  //       .createJoin([ toBeJoined1.operation, toBeJoined2.operation ], false),
  //   };
  //   entries.push(firstEntry);


  //   /* Update the mapping to reflect the newly executed join*/
  //   /* Find maximum and minimum index in the join entries array to update the mapping */
  //   const max_idx: number = this.nodeid_mapping.get(Math.max(...ids))!;
  //   const min_idx: number = this.nodeid_mapping.get(Math.min(...ids))!;

  //   /* Update the mappings by keeping track of the position of each entry in the nodeid mapping and updating according to position in the array*/
  //   for (let [key, value] of this.nodeid_mapping){
  //     if (value > max_idx){
  //       this.nodeid_mapping.set(key, value-2);
  //     }
  //     else if(value > min_idx && value < max_idx){
  //       this.nodeid_mapping.set(key, value-1);
  //     }
  //   }
    
  //   /* Remove joined nodes from the index mapping*/
  //   this.nodeid_mapping.set(ids[0], -1); this.nodeid_mapping.set(ids[1], -1);

  //   /* Add the joined entry to the index mapping, here we assume the joined streams are added at the end of the array, need to check this assumption*/
  //   const join_id: number = action.context.getEpisodeState().nodesArray[action.context.getEpisodeState().numNodes-1].id;
  //   this.nodeid_mapping.set(join_id, action.entries.length-1);
    
  //   /* If we're at the last time this is executed */
  //   if (action.context.getEpisodeState().numUnjoined == 2){
  //     this.nodeid_mapping = new Map<number, number>();
  //   }

  //   return {
  //     result: await this.mediatorJoin.mediate({
  //       type: action.type,
  //       entries,
  //       context: action.context,
  //     }),
  //     physicalPlanMetadata: {
  //       smallest: [ toBeJoined1, toBeJoined2 ],
  //     },
  //   }
  // }
}


  // protected async getJoinCoefficients(
  //   action: IActionRdfJoin,
  //   metadatas: MetadataBindings[],
  // ): Promise<IMediatorTypeJoinCoefficients> {
  //   /**
  //    * Execute the RL agent on all possible joins from the current join state and return the best join state.
  //    * @param {IActionRdfJoin} action An array containing information on the joins
  //    * @param {MetadataBindings[]} metadatas An array with all  metadatas of the bindingstreams
  //    * @returns {Promise<IMediatorTypeJoinCoefficients>} Interface with estimated cost of the best join
  //   */

  //   /* If we have no joinState or nodeid_mapping due to resets we initialise empty ones */
  //   const passedNodeIdMapping = action.context.getNodeIdMapping();
  //   if(passedNodeIdMapping.size == 0){
  //     for(let i:number=0;i<action.entries.length;i++){
  //       passedNodeIdMapping.set(i,i);
  //     }
  //   }

  //   let nodeIndexes: number[] = [... Array(action.context.getEpisodeState().numNodes).keys()];

  //   /*  Remove already joined nodes from consideration by traversing the node indices in reverse order 
  //       which does not disturb the indexing of the array  */
  //   for (let i:number=action.context.getEpisodeState().numNodes-1;i>=0;i--){
  //     if (action.context.getEpisodeState().nodesArray[i].joined==true){
  //       nodeIndexes.splice(i,1);
  //     }
  //   }
  //   /*  Generate possible joinState combinations  */
  //   const possibleJoins: number[][] = this.getPossibleJoins(nodeIndexes);
  //   let valueProbabilityJoin: number[] = [];
  //   let estimatedOnlineValueJoin: number = 0;

  //   if (!this.offlineTrain){
  //     estimatedOnlineValueJoin = await this.getChosenJoinOnline(action, metadatas, possibleJoins);
  //   }
  //   else{  
  //     valueProbabilityJoin = await this.getChosenJoinOffline(action, metadatas, possibleJoins);  
  //   }

  //   const joined_nodes: NodeStateSpace[] = action.context.getEpisodeState().nodesArray[action.context.getEpisodeState().numNodes-1].children;
  //   const ids: number[] = joined_nodes.map(a => a.id).sort();
  //   const indexJoins: number[] = ids.map(a => passedNodeIdMapping.get(a)!).sort();

  //   /* Add the newly chosen joins to the joinIndexes in the stateSpaceTree */
  //   action.context.getEpisodeState().addJoinIndexes(indexJoins);

  //   /* Update our master tree with new predictions if we perform offline training */
  //   if (this.offlineTrain){
  //     const newState: number[][] = action.context.getEpisodeState().joinIndexes
  //     const prevPrediction: MCTSJoinInformation = action.context.getJoinStateMasterTree(newState);
  //     const predictionOutput: MCTSJoinPredictionOutput = {N: prevPrediction.N, state: newState, predictedValueJoin: valueProbabilityJoin[0], 
  //       predictedProbability: valueProbabilityJoin[1]};
      
  //     let featureMatrix: number[][] = [];
  //     // console.log(action.entries);
  //     const operations: tempContextEntries = action.context.get(KeysQueryOperation.operation)!;
  //     for(var i=0;i<action.context.getEpisodeState().nodesArray.length;i++){
  //       // const quad: RdfJs.Quad = Quads[i];
  //       // console.log(action.context.getNodeIdMapping());
  //       // console.log(quad)
  //       // const triple = [quad.subject, quad.predicate, quad.object];
  //       featureMatrix.push(action.context.getEpisodeState().nodesArray[i].features);
  //     }      

  //     let adjacencyMatrixCopy = []
  //     for (let i = 0; i<action.context.getEpisodeState().adjacencyMatrix.length; i++){
  //       adjacencyMatrixCopy.push(action.context.getEpisodeState().adjacencyMatrix[i].slice());
  //     }

  //     action.context.setJoinStateMasterTree(predictionOutput, featureMatrix, adjacencyMatrixCopy);
  //     action.context.setPlanHolder(newState.flat().toString().replaceAll(',', ''));
  //   }

  //   return {
  //     iterations: this.prevEstimatedCost,
  //     persistedItems: 0,
  //     blockingItems: 0,
  //     requestTime: 0,
  //   };

  // }

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
    const passedNodeIdMapping = action.context.getNodeIdMapping();
    if(passedNodeIdMapping.size == 0){
      for(let i:number=0;i<action.entries.length;i++){
        passedNodeIdMapping.set(i,i);
      }
    }

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
    // UGLY!!
    const temp0Tensor: tf.Tensor = tf.tensor([0]);
    let valueProbabilityJoin:  [number, number, number[], tf.Tensor[]] = [0, 0, [0], [temp0Tensor]];
    let estimatedOnlineValueJoin: number = 0;
    const MCTSearchTree: MCTSMasterTree = _.cloneDeep(action.context.getMasterTree());
    if (action.context.getValidation()){
      // estimatedOnlineValueJoin = await this.getChosenJoinOnline(action, metadatas, possibleJoins);
      valueProbabilityJoin = await this.MCTSOffline(action, action.context.getModelHolder(), MCTSearchTree); 

    }
    else{  
      valueProbabilityJoin = await this.MCTSOffline(action, action.context.getModelHolder(), MCTSearchTree); 
    }
    temp0Tensor.dispose();
    const joined_nodes: NodeStateSpace[] = action.context.getEpisodeState().nodesArray[action.context.getEpisodeState().numNodes-1].children;
    const newFeaturesJoin = [];
    for(let i=0;i<8;i++){
      newFeaturesJoin.push(joined_nodes[0].features[i]+joined_nodes[1].features[i]);
    }
    const ids: number[] = joined_nodes.map(a => a.id).sort();
    const indexJoins: number[] = ids.map(a => passedNodeIdMapping.get(a)!).sort();

    /* Add the newly chosen joins to the joinIndexes in the stateSpaceTree */
    action.context.getEpisodeState().addJoinIndexes(indexJoins);
    /* Update our master tree with new predictions if we perform offline training */
    if (this.offlineTrain){
      const valueAddedJoin: number = valueProbabilityJoin[0] as number;
      const priorProbJoin: number = valueProbabilityJoin[1] as number;
      const actualProbabilityVector: number[] = valueProbabilityJoin[2] as number[];
      const estimatedProbabilityVector: tf.Tensor[] = valueProbabilityJoin[3];

      const newState: number[][] = action.context.getEpisodeState().joinIndexes
      const prevPrediction: MCTSJoinInformation = MCTSearchTree.getJoinInformation(newState);

      let featureMatrix: number[][] = [];
      for(var i=0;i<action.context.getEpisodeState().nodesArray.length;i++){
        featureMatrix.push(action.context.getEpisodeState().nodesArray[i].features);
      }      
      let adjacencyMatrixCopy = []
      for (let i = 0; i<action.context.getEpisodeState().adjacencyMatrix.length; i++){
        adjacencyMatrixCopy.push(action.context.getEpisodeState().adjacencyMatrix[i].slice());
      }
      if (MCTSearchTree.masterMap.get(newState.flat().toString().replaceAll(',', ''))){
        MCTSearchTree.getJoinInformation(newState).predictedProbabilityVector?.map(x=>x.dispose());
      }
      const choiceToAdd: MCTSJoinInformation = {N: prevPrediction.N, meanValueJoin: valueAddedJoin, 
        totalValueJoin: prevPrediction.totalValueJoin, priorProbability: priorProbJoin, featureMatrix: featureMatrix, adjencyMatrix: adjacencyMatrixCopy, 
        actualProbabilityVector: actualProbabilityVector, predictedProbabilityVector: estimatedProbabilityVector};
      // console.log("Added choice feature matrix");
      // console.dir(featureMatrix, {'maxArrayLength': null});

      action.context.getMasterTree().updateJoinDirect(choiceToAdd, newState);
      action.context.setPlanHolder(newState.flat().toString().replaceAll(',', ''));
    }

    return {
      iterations: this.prevEstimatedCost,
      persistedItems: 0,
      blockingItems: 0,
      requestTime: 0,
    };

  }



  public async MCTSOffline(action: IActionRdfJoin, modelHolder: modelHolder, masterTree: MCTSMasterTree): Promise<[number, number, number[], tf.Tensor[]]>{
    const searchFactor = 5;
    const searchAction: IActionRdfJoin = _.cloneDeep(action);
    const state: StateSpaceTree = searchAction.context.getEpisodeState(); 
    // First expand the root of the seach
    const numPossibleChild: number = await this.expandChildStates(searchAction, modelHolder, masterTree);
    // Num sims from http://ceur-ws.org/Vol-2652/paper05.pdf, alpha join
    const numSimulations = (numPossibleChild) * searchFactor;

    // Add root to masterTee
    const adjMatrix: number[][] = searchAction.context.getEpisodeState().adjacencyMatrix;
    let featureMatrix: number[][] = [];
    for(let i=0;i<searchAction.context.getEpisodeState().nodesArray.length;i++){
      featureMatrix.push(searchAction.context.getEpisodeState().nodesArray[i].features);
    }
    // console.log("Before")
    // console.log(masterTree.getJoinInformation(state.getJoinIndexes()));

    searchAction.context.setJoinStateMasterTree({N:1, state:state.getJoinIndexes(), predictedValueJoin: 0, predictedProbability: 1}, featureMatrix, adjMatrix);
    masterTree.updateJoin({N:1, state:state.getJoinIndexes(), predictedValueJoin: 0, predictedProbability: 1}, featureMatrix, adjMatrix);
    // console.log("After")
    // console.log(masterTree.getJoinInformation(state.getJoinIndexes()));
    // Perform MCTS simulations

    for(let i=0; i<numSimulations; i++){
      const cloneAction: IActionRdfJoin = _.cloneDeep(searchAction);
      await this.MCTSearch(cloneAction, modelHolder, masterTree, state.getJoinIndexes());
      const rootInformation: MCTSJoinInformation = masterTree.getJoinInformation(state.getJoinIndexes());
      rootInformation.N += 1;
    }
    // Choose next join based on largest N
    const rootInformation: MCTSJoinInformation = masterTree.getJoinInformation(state.getJoinIndexes());
    const episodeState = searchAction.context.getEpisodeState();
    let nodeIndexes: number[] = [... Array(episodeState.numNodes).keys()];
    const possibleChildren: number[][] = this.getPossibleChildren(nodeIndexes, episodeState);
    const improvedProbabilitiesChildren: number[] = [];
    const joinKeysChildren: number[][][] = [];
    const estimatedValuesChildren: number[] = [];
    const estimatedProbabilitiesChildren: tf.Tensor[] = []
    // let bestJoin: number[] = [0,0]; let bestProbability = -1; let bestJoinKey: number[][] = [[]]; let estimatedValue = 0;
    for (const join of possibleChildren){
      // Construct join key
      const passedNodeIdMapping = searchAction.context.getNodeIdMapping();  
      const indexJoins: number[] = join.map(x => passedNodeIdMapping.get(x)!).sort();
      const joinKey = [...state.getJoinIndexes()];
      joinKey.push(indexJoins);

      // Get best probability of improved policy
      const improvedPolicyProbability = masterTree.getJoinInformation(joinKey).N / (rootInformation.N-1);
      improvedProbabilitiesChildren.push(improvedPolicyProbability);
      joinKeysChildren.push(joinKey);
      estimatedValuesChildren.push(masterTree.getJoinInformation(joinKey).meanValueJoin);
      // Dispose this tensor!!
      estimatedProbabilitiesChildren.push(masterTree.getJoinInformation(joinKey).estimatedProbabilityTensor!);

      // Dispose of tensor in masterTree when we don't need it anymore

      // if (improvedPolicyProbability > bestProbability){
      //   bestJoin = join;
      //   bestProbability = improvedPolicyProbability;
      //   bestJoinKey = joinKey;
      //   estimatedValue = masterTree.getJoinInformation(joinKey).meanValueJoin;
      // }
    }
    // // Set actual probability vector
    // rootInformation.actualProbabilityVector = improvedProbabilitiesChildren;
    // // Set estimated probabilities vector
    // rootInformation.predictedProbabilityVector = estimatedProbabilitiesChildren;

    const idx = this.chooseUsingProbability(improvedProbabilitiesChildren);
    // console.log(improvedProbabilitiesChildren);
    // console.log("Feature Matrix of Added Join");
    // console.log(masterTree.getJoinInformation(joinKeysChildren[idx]));
    const featureMatrixJoin: number[][] = masterTree.getJoinInformation(joinKeysChildren[idx]).featureMatrix;
    const featureAddedJoin: number[] = featureMatrixJoin[featureMatrixJoin.length-1];
    // console.log("FeatureAddedJoin");
    // console.log(featureAddedJoin)
    
    const joinEmbedding: number[] = featureAddedJoin.slice(8);
    const newFeatures: number[] = [0,0,0,0,0,0,0,1, ...joinEmbedding]

    const newParent: NodeStateSpace = new NodeStateSpace(action.context.getEpisodeState().numNodes, newFeatures, 0, joinEmbedding);
    action.context.getEpisodeState().addParent(possibleChildren[idx], newParent);

    // Add self connection to explored join, experimental
    const adjMatrixCopy = action.context.getEpisodeState().adjacencyMatrix;
    adjMatrixCopy[adjMatrixCopy.length-1][adjMatrixCopy[0].length-1] = 1;
    return [estimatedValuesChildren[idx], improvedProbabilitiesChildren[idx], improvedProbabilitiesChildren, estimatedProbabilitiesChildren];
  }
  
  public async MCTSearch(action: IActionRdfJoin, modelHolder: modelHolder, MCTSMasterTree: MCTSMasterTree, rootJoin: number[][]){
    /**
     * Call this to either expand node or select another node recursively
     */
    const currentJoinStateKey: number[][] = action.context.getEpisodeState().joinIndexes;
    const currentJoinStateInformation: MCTSJoinInformation = MCTSMasterTree.getJoinInformation(currentJoinStateKey)!;
    const clonedAction: IActionRdfJoin = _.cloneDeep(action);
    const state: StateSpaceTree = clonedAction.context.getEpisodeState();
    // console.log(`Join ${currentJoinStateKey}`);
    if (currentJoinStateInformation.priorProbability<0){
      console.error("We have prior probability smaller than 0 (Actor)");
    }
    // STILL CHECK IF TERMINAL STATE AND RETURN UPWARDS
    if (currentJoinStateInformation.N>0){
      // 2. If we already explored, call search recursively on selected child node based on UCT
      let nodeIndexes: number[] = [... Array(state.numNodes).keys()];
      const possibleChildren: number[] = nodeIndexes.filter(item => !state.nodesArray[item].joined);
      let possibleJoins: number[][] = this.getPossibleJoins(possibleChildren);

      if (possibleJoins.length == 0){
        // Terminal State propogate the q value upwards!
        const outputChild = await this.getOutputChild(clonedAction, modelHolder, MCTSMasterTree, currentJoinStateKey);  
        outputChild[1].dispose();
        // Propogate found value of join upwards
        const prevJoin = [...currentJoinStateKey];
        for (let i=1;i<currentJoinStateKey.length;i++){
          prevJoin.pop();
          if (!_.isEqual(prevJoin, rootJoin)){
            const joinInformationParent: MCTSJoinInformation = MCTSMasterTree.getJoinInformation(prevJoin);
            const newQ = (joinInformationParent.totalValueJoin + outputChild[0][0])/(joinInformationParent.N+1);
            joinInformationParent.N += 1; joinInformationParent.meanValueJoin=newQ; joinInformationParent.totalValueJoin += outputChild[0][0];
          } 
        }
        // Break out of recursion
        return;
      }

      let maxU = Number.NEGATIVE_INFINITY;
      let bestJoin = [0,0];
      for (const join of possibleJoins){
        const joinEntryIndexes: number[] = join.map(x => clonedAction.context.getNodeIdMapping().get(x)!);
        const newJoinKey: number[][] = [...currentJoinStateKey];
        newJoinKey.push(joinEntryIndexes);

        const joinInformation: MCTSJoinInformation = MCTSMasterTree.getJoinInformation(newJoinKey);
        // Here the UCB value is calculated according to https://web.stanford.edu/~surag/posts/alphazero.html, note that the parent # visits is equal to the total visits of 
        // All child actions.
        const U = joinInformation.meanValueJoin + this.explorationDegree * joinInformation.priorProbability*(Math.sqrt(currentJoinStateInformation.N)/(1+joinInformation.N));
        if (U >= maxU){
          maxU = U;
          bestJoin = join;
        }
      }
      // Execute the best join and recursively call the MCTSearch
      await this.executeJoinMCTS(clonedAction, bestJoin);
      await this.MCTSearch(clonedAction, modelHolder, MCTSMasterTree, rootJoin);
      return;
    }
    else{

      // 3. If we haven't, expand the child states and add them to MCTS
      /* Here we add x tensor if it has x child*/
      const nPossibleChild: number = await this.expandChildStates(clonedAction, modelHolder, MCTSMasterTree);
      
  
      // Add unexplored node to join state properly, feature matrix and Adj matrix are added in child expansion phase
      /* NO LEAK*/
      const outputChild = await this.getOutputChild(clonedAction, modelHolder, MCTSMasterTree, currentJoinStateKey);
      outputChild[1].dispose();
      /* NO LEAK */


      const joinInformationNode: MCTSJoinInformation = MCTSMasterTree.getJoinInformation(currentJoinStateKey); 
      joinInformationNode.meanValueJoin = outputChild[0][0]; joinInformationNode.totalValueJoin = outputChild[0][0]; joinInformationNode.N = 1;
      // Propogate found value of join upwards
      const prevJoin = [...currentJoinStateKey];
      for (let i=1;i<currentJoinStateKey.length;i++){
        prevJoin.pop();
        if (!_.isEqual(prevJoin, rootJoin)){
          const joinInformationParent: MCTSJoinInformation = MCTSMasterTree.getJoinInformation(prevJoin);
          const newQ = (joinInformationParent.totalValueJoin + outputChild[0][0])/(joinInformationParent.N+1);
          joinInformationParent.N += 1; joinInformationParent.meanValueJoin=newQ; joinInformationParent.totalValueJoin+= outputChild[0][0];
        } 
      }
    }
  }

  private async getOutputChild(action: IActionRdfJoin, modelHolder: modelHolder, masterTree: MCTSMasterTree, joinKey: number[][]): Promise<[number[], tf.Tensor]>{
    /*
    1. clone action
    2. execute proposed join on action
    3. Get features matrices / adjmatrices
    4. Return probability of child
    */

    const featureMatrix = masterTree.getJoinInformation(joinKey).featureMatrix;
    const adjMatrix = masterTree.getJoinInformation(joinKey).adjencyMatrix;
    /* Execute forward pass */
    const adjTensor: tf.Tensor2D = tf.tensor2d(adjMatrix);
    const inputFeaturesTensor: tf.Tensor2D = tf.tensor2d(featureMatrix, [featureMatrix.length,featureMatrix[0].length]);
    const forwardPassOutput: tf.Tensor[] = modelHolder.getModel().forwardPassTest(inputFeaturesTensor, adjTensor) as tf.Tensor[];

    const valueOutput = (await forwardPassOutput[0].data())[featureMatrix.length-1]
    const policyOutput = (await forwardPassOutput[1].data())[featureMatrix.length-1];

    const probTensor: tf.Tensor = forwardPassOutput[1].slice(featureMatrix.length-1, 1);

    inputFeaturesTensor.dispose(); forwardPassOutput.map(x => x.dispose()); adjTensor.dispose(); 

    return [[valueOutput, policyOutput], probTensor];
  }

  private async expandChildStates(action: IActionRdfJoin, modelHolder: modelHolder, masterTree: MCTSMasterTree){
    // Clone the state so we can recursively search
    const episodeState = _.cloneDeep(action.context.getEpisodeState());
    let nodeIndexes: number[] = [... Array(episodeState.numNodes).keys()];
    const possibleChildren: number[][] = this.getPossibleChildren(nodeIndexes, episodeState);
    if (possibleChildren.length>0){
      let priorProbabilitiesEstimated: number[] = []
      let estimatedProbabilitiesTensor: tf.Tensor[] = []
      for (const join of possibleChildren){
        // Get probabilties for each child
        const addedJoinKey: number[][] = await this.constructJoinState(action, join, masterTree);
        const predictionOutput = await this.getOutputChild(action, modelHolder, masterTree, addedJoinKey);
        priorProbabilitiesEstimated.push(predictionOutput[0][1]);
        estimatedProbabilitiesTensor.push(predictionOutput[1]);
      }
      // So ugly, so much waste :/
      const concatinatedProbTensor: tf.Tensor = tf.concat(estimatedProbabilitiesTensor)
  
      const priorProbabilitiesTensor: tf.Tensor = tf.tensor(priorProbabilitiesEstimated);
      const probabilitiesTensor: tf.Tensor = this.softMax(priorProbabilitiesTensor);
      const estimatedProbabilitiesTensorSoftMax: tf.Tensor = this.softMax(concatinatedProbTensor);
      // There is something wrong with probabilities, gives NaN sometimes;
      const probabilities = probabilitiesTensor.dataSync();
      for(let i=0;i<probabilities.length;i++){
        const joinId = possibleChildren[i];
        const joinEntryIndexes: number[] = joinId.map(x => action.context.getNodeIdMapping().get(x)!);
        // Get key to identify the explored child nodes
        const newJoinIndexes = [...episodeState.joinIndexes];
        newJoinIndexes.push(joinEntryIndexes);
  
        // Update the prior probabilities
        const toUpdateJoinInformation: MCTSJoinInformation = masterTree.getJoinInformation(newJoinIndexes);
        toUpdateJoinInformation.priorProbability = probabilities[i];
        toUpdateJoinInformation.estimatedProbabilityTensor = estimatedProbabilitiesTensorSoftMax.slice([i,0],[1,1]);
      }

      priorProbabilitiesTensor.dispose(); probabilitiesTensor.dispose(); concatinatedProbTensor.dispose(); 
      estimatedProbabilitiesTensorSoftMax.dispose(); estimatedProbabilitiesTensor.map(x=>x.dispose());  
    }
    return possibleChildren.length;
  } 

  private async constructJoinState(action: IActionRdfJoin, join: number[], masterTree: MCTSMasterTree){
    /**
     * Function that takes a not yet executed join,  executes it and creates the feature matrixes and adjacency matrixes and adds these to the masterTree
     * It is then the responsibility of another function to update the prior probability by taking these feature matrixes and executing the forwardPass.
     * returns: the indexes / key of the added join
     */
    /* NO LEAK HERE */
    const actionToExplore: IActionRdfJoin = _.cloneDeep(action);

    // Execute the join
    await this.executeJoinMCTS(actionToExplore, join);

    let featureMatrix: number[][] = [];

    for(let i=0;i<actionToExplore.context.getEpisodeState().nodesArray.length;i++){
      featureMatrix.push(actionToExplore.context.getEpisodeState().nodesArray[i].features);
    }
    const addedJoinIndexes = actionToExplore.context.getEpisodeState().joinIndexes;
    const joinInformationToAdd: MCTSJoinInformation = {N:0, meanValueJoin:0, priorProbability: 0, totalValueJoin:0, featureMatrix: featureMatrix, 
      adjencyMatrix: actionToExplore.context.getEpisodeState().adjacencyMatrix}
    // const joinInformationToAdd: MCTSJoinPredictionOutput = {N:0, predictedValueJoin:0, predictedProbability: 0, state: addedJoinIndexes};
    masterTree.updateJoinDirect(joinInformationToAdd, addedJoinIndexes)
    // masterTree.updateMasterMap(joinInformationToAdd, featureMatrix, actionToExplore.context.getEpisodeState().adjacencyMatrix);
    return addedJoinIndexes;


  }
  
  private async executeJoinMCTS(copiedAction: IActionRdfJoin, joinToExecute: number[]){
    const passedNodeIdMapping = copiedAction.context.getNodeIdMapping();
    const entriesToExecute: IJoinEntry[] = copiedAction.entries;
    const state: StateSpaceTree = copiedAction.context.getEpisodeState();
    const joinedNodes = joinToExecute.map(x => state.nodesArray[x]);
    const runningMeans: runningMoments = copiedAction.context.getRunningMomentsMasterTree();

    // Get the join indexes corresponding to node indexes passed in the join 
    const indexJoins: number[] = joinToExecute.map(x => passedNodeIdMapping.get(x)!).sort();
    const toBeJoined1 = entriesToExecute[indexJoins[0]]; const toBeJoined2 = entriesToExecute[indexJoins[1]];

    entriesToExecute.splice(indexJoins[1],1); entriesToExecute.splice(indexJoins[0],1);

    /* Save the join indices */
    const firstEntry: IJoinEntry = {
      output: ActorQueryOperation.getSafeBindings(await this.mediatorJoin
        .mediate({ type: copiedAction.type, entries: [ toBeJoined1, toBeJoined2 ], context: copiedAction.context })),
      operation: ActorRdfJoinInnerMultiReinforcementLearning.FACTORY
        .createJoin([ toBeJoined1.operation, toBeJoined2.operation ], false),
    };
    entriesToExecute.push(firstEntry);

    /* Update the mapping to reflect the newly executed join*/
    /* Find maximum and minimum index in the join entries array to update the mapping */
    const max_idx: number = indexJoins[1];
    const min_idx: number = indexJoins[0];

    /* Update the mappings by keeping track of the position of each entry in the nodeid mapping and updating according to position in the array*/
    for (let [key, value] of passedNodeIdMapping){
      if (value > max_idx){
        copiedAction.context.setNodeIdMapping(key, value-2);
      }
      else if(value > min_idx && value < max_idx){
        copiedAction.context.setNodeIdMapping(key, value-1);
      }
    }

    /* Remove joined nodes from the index mapping*/
    copiedAction.context.setNodeIdMapping(joinToExecute[0], -1); copiedAction.context.setNodeIdMapping(joinToExecute[1], -1);

    // Here we try to use a reduced form of cardinality for feature, since the comunica cardinality explodes when selectivity is small
    const selectivity: number = (await this.mediatorJoinSelectivity.mediate({entries: [toBeJoined1,toBeJoined2], context: copiedAction.context})).selectivity;
    const testCardSmaller: number = (joinedNodes[0].cardinality + joinedNodes[1].cardinality)*selectivity


    // Add node with features to the state
    const numNodes = state.numNodes;
    const addedJoinMetaData: MetadataBindings = await entriesToExecute[entriesToExecute.length-1].output.metadata();
    // Get zeros vector for predicate representation
    const predEmbedding1 = joinedNodes[0].predicateEmbedding; const predEmbedding2 = joinedNodes[1].predicateEmbedding
    const newJoinEmbedding = this.maxPool(predEmbedding1, predEmbedding2);
    const featureNode: number[] = [testCardSmaller, 0,0,0,0,0,0, addedJoinMetaData.variables.length, ...newJoinEmbedding];
    // Standardise new features
    for (const index of runningMeans.indexes){
      const runningStatsIndex: aggregateValues = runningMeans.runningStats.get(index)!;
      featureNode[index] = (featureNode[index] - runningStatsIndex.mean) / runningStatsIndex.std;
    }

    const newParent: NodeStateSpace = new NodeStateSpace(numNodes, featureNode, testCardSmaller, newJoinEmbedding);
    state.addParent(joinToExecute, newParent);

    // Add the joined entry to the index mapping
    copiedAction.context.setNodeIdMapping(newParent.id, copiedAction.entries.length-1);

    // Add join to join indexes
    state.joinIndexes.push(indexJoins);
  }


  private getPossibleChildren(nodeIndexes: number[], episodeState: StateSpaceTree): number[][]{
    // Get all unjoined nodes, these will form the possible actions
    const possibleChildren: number[] = nodeIndexes.filter(item => !episodeState.nodesArray[item].joined).sort();
    // Get all actions by finding all possible join combinations
    let possibleJoins: number[][] = this.getPossibleJoins(possibleChildren);
    return possibleJoins
  }

  private async MCTSOnline(action: IActionRdfJoin, modelHolder: modelHolder, masterTree: MCTSMasterTree){
      const searchFactor = 5;
      const searchAction: IActionRdfJoin = _.cloneDeep(action);
      const state: StateSpaceTree = searchAction.context.getEpisodeState(); 
      // First expand the root of the seach
      const numPossibleChild: number = await this.expandChildStates(searchAction, modelHolder, masterTree);
      // Num sims from http://ceur-ws.org/Vol-2652/paper05.pdf, alpha join
      const numSimulations = (numPossibleChild) * searchFactor;
  
      // Add root to masterTee
      const adjMatrix: number[][] = searchAction.context.getEpisodeState().adjacencyMatrix;
      let featureMatrix: number[][] = [];
      for(let i=0;i<searchAction.context.getEpisodeState().nodesArray.length;i++){
        featureMatrix.push(searchAction.context.getEpisodeState().nodesArray[i].features);
      }
      // console.log("Before")
      // console.log(masterTree.getJoinInformation(state.getJoinIndexes()));
  
      searchAction.context.setJoinStateMasterTree({N:1, state:state.getJoinIndexes(), predictedValueJoin: 0, predictedProbability: 1}, featureMatrix, adjMatrix);
      masterTree.updateJoin({N:1, state:state.getJoinIndexes(), predictedValueJoin: 0, predictedProbability: 1}, featureMatrix, adjMatrix);
      // console.log("After")
      // console.log(masterTree.getJoinInformation(state.getJoinIndexes()));
      // Perform MCTS simulations
  
      for(let i=0; i<numSimulations; i++){
        const cloneAction: IActionRdfJoin = _.cloneDeep(searchAction);
        await this.MCTSearch(cloneAction, modelHolder, masterTree, state.getJoinIndexes());
        const rootInformation: MCTSJoinInformation = masterTree.getJoinInformation(state.getJoinIndexes());
        rootInformation.N += 1;
      }
      // Choose next join based on largest N
      const rootInformation: MCTSJoinInformation = masterTree.getJoinInformation(state.getJoinIndexes());
      const episodeState = searchAction.context.getEpisodeState();
      let nodeIndexes: number[] = [... Array(episodeState.numNodes).keys()];
      const possibleChildren: number[][] = this.getPossibleChildren(nodeIndexes, episodeState);
      const improvedProbabilitiesChildren: number[] = [];
      const joinKeysChildren: number[][][] = [];
      const estimatedValuesChildren: number[] = [];
      const estimatedProbabilitiesChildren: tf.Tensor[] = []
      // let bestJoin: number[] = [0,0]; let bestProbability = -1; let bestJoinKey: number[][] = [[]]; let estimatedValue = 0;
      for (const join of possibleChildren){
        // Construct join key
        const passedNodeIdMapping = searchAction.context.getNodeIdMapping();  
        const indexJoins: number[] = join.map(x => passedNodeIdMapping.get(x)!).sort();
        const joinKey = [...state.getJoinIndexes()];
        joinKey.push(indexJoins);
  
        // Get best probability of improved policy
        const improvedPolicyProbability = masterTree.getJoinInformation(joinKey).N / (rootInformation.N-1);
        improvedProbabilitiesChildren.push(improvedPolicyProbability);
        joinKeysChildren.push(joinKey);
        estimatedValuesChildren.push(masterTree.getJoinInformation(joinKey).meanValueJoin);
        // Dispose this tensor!!
        estimatedProbabilitiesChildren.push(masterTree.getJoinInformation(joinKey).estimatedProbabilityTensor!);
      }
      const idx = this.chooseUsingProbability(improvedProbabilitiesChildren);
      const featureMatrixJoin: number[][] = masterTree.getJoinInformation(joinKeysChildren[idx]).featureMatrix;
      const featureAddedJoin: number[] = featureMatrixJoin[featureMatrixJoin.length-1];
      
      const joinEmbedding: number[] = featureAddedJoin.slice(8);
      const newFeatures: number[] = [0,0,0,0,0,0,0,1, ...joinEmbedding]
  
      const newParent: NodeStateSpace = new NodeStateSpace(action.context.getEpisodeState().numNodes, newFeatures, 0, joinEmbedding);
      action.context.getEpisodeState().addParent(possibleChildren[idx], newParent);
  
      // Add self connection to explored join, experimental
      const adjMatrixCopy = action.context.getEpisodeState().adjacencyMatrix;
      adjMatrixCopy[adjMatrixCopy.length-1][adjMatrixCopy[0].length-1] = 1;
      return [estimatedValuesChildren[idx], improvedProbabilitiesChildren[idx], improvedProbabilitiesChildren, estimatedProbabilitiesChildren];

  }

  // private async getChosenJoinOnline(action: IActionRdfJoin, metadatas: MetadataBindings[], possibleJoins: number[][]){
  //   // Needs to be redone
  //   let bestJoinCost: number = Infinity;
  //   let bestJoin: number[] = [-1, -1];

  //   for (const joinCombination of possibleJoins){
  //     /*  We generate the join trees associated with the possible join combinations, note the tradeoff between memory usage (create new trees) 
  //         and cpu usage (delete the new node)
  //     */
  //     // Feature array indicating that this is a join
  //     const zeroEmbedding = new Array(this.predicateEmbeddingSize).fill(0);
  //     const featureArray: number[] = [0,0,0,0,0,0,0,1, ...zeroEmbedding]
  //     const newParent: NodeStateSpace = new NodeStateSpace(action.context.getEpisodeState().numNodes, featureArray, 0, zeroEmbedding);
  //     action.context.getEpisodeState().addParent(joinCombination, newParent);
  //     const adjTensor = tf.tensor2d(action.context.getEpisodeState().adjacencyMatrix);

  //     let featureMatrix: number[][] = [];
  //     for(var i=0;i<action.context.getEpisodeState().nodesArray.length;i++){
  //       featureMatrix.push(action.context.getEpisodeState().nodesArray[i].features);
  //     }

  //     /*  Scale features using Min-Max scaling  */
  //     // const maxCardinality: number = Math.max(...featureMatrix);
  //     // const minCardinality: number = Math.min(...featureMatrix);
  //     // featureMatrix = featureMatrix.map((vM, iM) => (vM - minCardinality) / (maxCardinality - minCardinality));
  //     const cardinalities: number[] = featureMatrix.map((x) => x[0]);
  //     const maxCardinality: number = Math.max(...cardinalities);
  //     const minCardinality: number = Math.min(...cardinalities);

  //     // for (let i=0; i<featureMatrix.length;i++){
  //     //   featureMatrix[i][0] = (featureMatrix[i][0]-minCardinality) / (maxCardinality - minCardinality)
  //     // }


  //     /* Execute forward pass */
  //     const forwardPassOutput = action.context.getModelHolder().getModel().forwardPassTest(tf.tensor2d(featureMatrix, [action.context.getEpisodeState().numNodes,1]), adjTensor) as tf.Tensor[];

  //     /* Add estimated value to the array */
  //     if (forwardPassOutput instanceof tf.Tensor){
  //       const outputArray= await forwardPassOutput.data();
  //       if (outputArray[outputArray.length-1]<bestJoinCost){
  //         bestJoin = joinCombination;
  //         bestJoinCost = outputArray[outputArray.length-1];
  //       }
  //     }

  //     action.context.getEpisodeState().removeLastAdded();
  //   }
  //   const zeroEmbedding = new Array(this.predicateEmbeddingSize).fill(0);
  //   /* Select join to explore based on probabilty obtained by softmax of the estimated join values*/
  //   /*  Update the actors joinstate, which will be used by the RL medaitor to update its joinstate too  */
  //   const featureNode: number[] = [bestJoinCost, 0,0,0,0,0,0,1, ...zeroEmbedding]
  //   const newParent: NodeStateSpace = new NodeStateSpace(action.context.getEpisodeState().numNodes, featureNode, 0, zeroEmbedding);
  //   action.context.getEpisodeState().addParent(bestJoin, newParent);
  //   this.prevEstimatedCost = bestJoinCost;

  //   return bestJoinCost;
  // }

  // private async getJoinCoefficientsOffline(action: IActionRdfJoin,
  //   metadatas: MetadataBindings[],
  // ): Promise<IMediatorTypeJoinCoefficients>
  // {
  //   /**
  //    * Execute the exploration phase of the offline training algorithm. 
  //    */
  //   if (this.nodeid_mapping.size == action.entries.length){
  //     for(let i:number=0;i<action.entries.length;i++){
  //       this.nodeid_mapping.set(i,i);
  //     }
  //   }

  //   let nodeIndexes: number[] = [... Array(action.context.getEpisodeState().numNodes).keys()];

  //   /*  Remove already joined nodes from consideration by traversing the node indices in reverse order 
  //       which does not disturb the indexing of the array  */
  //   for (let i:number=action.context.getEpisodeState().numNodes-1;i>=0;i--){
  //     if (action.context.getEpisodeState().nodesArray[i].joined==true){
  //       nodeIndexes.splice(i,1);
  //     }
  //   }
  //   /*  Generate possible joinState combinations  */
  //   const possibleJoins: number[][] = this.getPossibleJoins(nodeIndexes);

  //   /* List of all predicted join values and probabilities*/
  //   const joinValues: number[] = [];
  //   const joinProbabilities: number[] = [];
  //   const qValues: number[] = [];

  //   /* Keeps track of the total visits to any of the possible join states*/
  //   let numVisited: number[] = [];

    
  //   for (const joinCombination of possibleJoins){
  //     /*  We generate the join trees associated with the possible join combinations, note the tradeoff between memory usage (create new trees) 
  //         and cpu usage (delete the new node)
  //     */
  //     const newParent: NodeStateSpace = new NodeStateSpace(action.context.getEpisodeState().numNodes, 0);
  //     action.context.getEpisodeState().addParent(joinCombination, newParent);
  //     const adjTensor: tf.Tensor2D = tf.tensor2d(action.context.getEpisodeState().adjacencyMatrix);

  //     let featureMatrix: number[] = [];
  //     for(var i=0;i<action.context.getEpisodeState().nodesArray.length;i++){
  //       featureMatrix.push(action.context.getEpisodeState().nodesArray[i].cardinality);
  //     }

  //     /* Execute forward pass */
  //     const forwardPassOutput: tf.Tensor[] = this.model.forwardPass(tf.tensor2d(featureMatrix, [action.context.getEpisodeState().numNodes,1]), adjTensor) as tf.Tensor[];

  //     const valueOutput: tf.Tensor = forwardPassOutput[0];
  //     const policyOutput: tf.Tensor = forwardPassOutput[1];

  //     const joined_nodes: NodeStateSpace[] = action.context.getEpisodeState().nodesArray[action.context.getEpisodeState().numNodes-1].children;
  //     const ids: number[] = joined_nodes.map(a => a.id).sort();
  //     const indexJoins: number[] = ids.map(a => this.nodeid_mapping.get(a)!).sort();

  //     /* Make copy of joinState to make a temporary key */
  //     const currentJoinState: number[][] = JSON.parse(JSON.stringify(action.context.getEpisodeState().joinIndexes));
  //     currentJoinState.push(indexJoins);

  //     /* Get any information of the possible join state in masterTree*/
  //     const joinStateInformation: MCTSJoinInformation = action.context.getJoinStateMasterTree(currentJoinState);
  //     numVisited.push(joinStateInformation.N);
  
  //     const outputArrayValue: Float32Array | Int32Array | Uint8Array = await valueOutput.data();
  //     const outputArrayPolicy: Float32Array | Int32Array | Uint8Array = await policyOutput.data();
      
  //     const qValue: number = (joinStateInformation.totalValueJoin + outputArrayValue[outputArrayValue.length-1])/(joinStateInformation.N+1);
  //     qValues.push(qValue);

  //     // const outputArrayValue = await forwardPassOutput.data()

  //     joinValues.push(outputArrayValue[outputArrayValue.length-1]);
  //     joinProbabilities.push(outputArrayPolicy[outputArrayPolicy.length-1]);

  //     action.context.getEpisodeState().removeLastAdded();

  //   }
  //   const predictedValueTensor: tf.Tensor = tf.tensor(joinValues);
  //   const predictedValueArray: Float32Array = await predictedValueTensor.data() as Float32Array;

  //   /* NOTE THESE ARE NOT REAL PROBABILITIES AND ARE PLACEHOLDERS!! */
  //   const predictedProbabilityTensor: tf.Tensor = tf.tensor(joinProbabilities);
  //   const predictedProbabilityArray: Float32Array = await predictedProbabilityTensor.data() as Float32Array;

  //   const joinUtility: number[] = []
  //   /* Calculate join utility according to q[i] + c * (sqrt(sum_i(visits))/1+visits_i)   */
  //   // console.log(`Number of join combinations possible: ${possibleJoins.length} `);
  //   // console.log(`Number of action entries: ${action.entries.length}`);
  //   // console.log(qValues);
  //   // console.log(numVisited);
  //   // console.log(sum(numVisited));
  //   // console.log(this.explorationDegree);

  //   for (let i = 0; i < qValues.length ; i++){
  //     joinUtility.push(qValues[i] + this.explorationDegree * (Math.sqrt(sum(numVisited)) / (1 + numVisited[i])))
  //   }
  //   const probabilities: Float32Array = await tf.softmax(joinUtility).data() as Float32Array;
  //   const idx = this.chooseUsingProbability(probabilities);
    
  //   /* Add new explored node to the statespace*/
  //   const newParent: NodeStateSpace = new NodeStateSpace(action.context.getEpisodeState().numNodes, predictedValueArray[idx]);
  //   action.context.getEpisodeState().addParent(possibleJoins[idx], newParent);
  //   this.prevEstimatedCost = predictedValueArray[idx];

  //   if (this.nodeid_mapping.size == 0){
  //     for(let i:number=0;i<action.entries.length;i++){
  //       this.nodeid_mapping.set(i,i);
  //     }
  //   }

  //   /* This is double work, should be aggressively optimized in terms of code re-use and readability!!*/
  //   const joined_nodes: NodeStateSpace[] = action.context.getEpisodeState().nodesArray[action.context.getEpisodeState().numNodes-1].children;
  //   const ids: number[] = joined_nodes.map(a => a.id).sort();
  //   const indexJoins: number[] = ids.map(a => this.nodeid_mapping.get(a)!).sort();

  //   /* Add the newly chosen joins to the joinIndexes in the stateSpaceTree */
  //   action.context.getEpisodeState().addJoinIndexes(indexJoins);
  //   /* Update our master tree with new predictions */
  //   const newState: number[][] = action.context.getEpisodeState().joinIndexes
  //   const prevPrediction: MCTSJoinInformation = action.context.getJoinStateMasterTree(newState);
  //   const predictionOutput: MCTSJoinPredictionOutput = {N: prevPrediction.N, state: newState, predictedValueJoin: predictedValueArray[idx], 
  //                                                       predictedProbability: predictedProbabilityArray[idx]};
  //   action.context.setJoinStateMasterTree(predictionOutput);
  //   return {
  //     iterations: this.prevEstimatedCost,
  //     persistedItems: 0,
  //     blockingItems: 0,
  //     requestTime: 0,
  //   };
  // } 

  
  // private async getChosenJoinOffline(action: IActionRdfJoin, metadatas: MetadataBindings[], possibleJoins: number[][]){
  //   /* List of all predicted join values, probabilities, and Q values */
  //   const joinValues: number[] = [];
  //   const joinProbabilities: number[] = [];
  //   const qValues: number[] = [];

  //   /* Keeps track of the total visits to any of the possible join states*/
  //   let numVisited: number[] = [];

  //   for (const joinCombination of possibleJoins){
  //     /*  We generate the join trees associated with the possible join combinations, note the tradeoff between memory usage (create new trees) 
  //         and cpu usage (delete the new node)
  //     */
  //     // Here we get the new estimated cardinality of to be added join
  //     const joinedNodes = joinCombination.map(x => action.context.getEpisodeState().nodesArray[x]);
  //     const toBeJoinedIndexes: number[] = joinCombination.map(x => action.context.getNodeIdMapping().get(x)!).sort();
  //     const toBeJoined1 = action.entries[toBeJoinedIndexes[0]]; const toBeJoined2 = action.entries[toBeJoinedIndexes[1]];
  
  //     const selectivity: number = (await this.mediatorJoinSelectivity.mediate({entries: [toBeJoined1,toBeJoined2], context: action.context})).selectivity;
  //     const predCardinality: number = (joinedNodes[0].cardinality + joinedNodes[1].cardinality)*selectivity;

  //     // Get number of variables in new join
  //     const uniqueVariablesJoin: Set<string> = new Set<string>();
  //     const variablesTest1 = (await toBeJoined1.output.metadata()).variables; const variablesTest2 = (await toBeJoined2.output.metadata()).variables; 
  //     variablesTest1.map(x => uniqueVariablesJoin.add(x.value)); variablesTest2.map(x => uniqueVariablesJoin.add(x.value))
  //     const numVariablesJoin: number = uniqueVariablesJoin.size;
  //     // Set new features of our to be evaluated join
  //     const featureNode: number[] = [predCardinality,0,0,0,0,0,0,numVariablesJoin];

  //     // Normalize these features according to tracked indexes
  //     for (const index of action.context.getRunningMomentsMasterTree().indexes){
  //       const aggValues: aggregateValues = action.context.getRunningMomentsMasterTree().runningStats.get(index)!;
  //       featureNode[index] = (featureNode[index]- aggValues.mean)/aggValues.std;
  //     }

  //     // Add new Node to state space
  //     const newParent: NodeStateSpace = new NodeStateSpace(action.context.getEpisodeState().numNodes, featureNode, 0);
  //     action.context.getEpisodeState().addParent(joinCombination, newParent);
  //     const adjTensor: tf.Tensor2D = tf.tensor2d(action.context.getEpisodeState().adjacencyMatrix);

  //     // Construct feature matrices
  //     let featureMatrix: number[][] = [];
  //     for(let i=0;i<action.context.getEpisodeState().nodesArray.length;i++){
  //       featureMatrix.push(action.context.getEpisodeState().nodesArray[i].features);
  //     }      

  //     /* Execute forward pass */
  //     const inputFeaturesTensor: tf.Tensor2D = tf.tensor2d(featureMatrix, [action.context.getEpisodeState().numNodes,featureMatrix[0].length])
  //     const forwardPassOutput: tf.Tensor[] = action.context.getModelHolder().getModel().forwardPassTest(inputFeaturesTensor, adjTensor) as tf.Tensor[];

  //     const valueOutput: tf.Tensor = forwardPassOutput[0];
  //     const policyOutput: tf.Tensor = forwardPassOutput[1];

  //     const joined_nodes: NodeStateSpace[] = action.context.getEpisodeState().nodesArray[action.context.getEpisodeState().numNodes-1].children;
  //     const ids: number[] = joined_nodes.map(a => a.id).sort();
  //     const indexJoins: number[] = ids.map(a => action.context.getNodeIdMappingKey(a)!).sort();

  //     /* Make copy of joinState to make a temporary key */
  //     const currentJoinState: number[][] = JSON.parse(JSON.stringify(action.context.getEpisodeState().joinIndexes));
  //     currentJoinState.push(indexJoins);

  //     /* Get any information of the possible join state in masterTree*/
  //     const joinStateInformation: MCTSJoinInformation = action.context.getJoinStateMasterTree(currentJoinState);
  //     numVisited.push(joinStateInformation.N);
  
  //     const outputArrayValue: Float32Array | Int32Array | Uint8Array = await valueOutput.data();
  //     const outputArrayPolicy: Float32Array | Int32Array | Uint8Array = await policyOutput.data();
      
  //     // Dispose of tensors
  //     valueOutput.dispose(); policyOutput.dispose(); forwardPassOutput.map((x) => x.dispose()); adjTensor.dispose(); inputFeaturesTensor.dispose();

  //     const qValue: number = (joinStateInformation.totalValueJoin + outputArrayValue[outputArrayValue.length-1])/(joinStateInformation.N+1);
  //     qValues.push(qValue);
  //     // const outputArrayValue = await forwardPassOutput.data()

  //     joinValues.push(outputArrayValue[outputArrayValue.length-1]);
  //     joinProbabilities.push(outputArrayPolicy[outputArrayPolicy.length-1]);

  //     action.context.getEpisodeState().removeLastAdded();

  //   }
    
  //   const predictedValueTensor: tf.Tensor = tf.tensor(joinValues);
  //   const predictedValueArray: Float32Array = await predictedValueTensor.data() as Float32Array;

  //   /* NOTE THESE ARE NOT REAL PROBABILITIES AND ARE PLACEHOLDERS!! */
  //   const predictedProbabilityTensor: tf.Tensor = tf.tensor(joinProbabilities);
  //   const predictedProbabilityArray: Float32Array = await predictedProbabilityTensor.data() as Float32Array;

  //   // Get estimated utilities of join
  //   const joinUtility: number[] = []
  //   for (let i = 0; i < qValues.length ; i++){
  //     joinUtility.push(qValues[i] + this.explorationDegree * (Math.sqrt(sum(numVisited)) / (1 + numVisited[i])))
  //   }
  //   const softMaxTensor: tf.Tensor = tf.softmax(joinUtility);
  //   const probabilities: Float32Array = await softMaxTensor.data() as Float32Array;
  //   // Choose join combination index
  //   const idx = this.chooseUsingProbability(probabilities);
  //   const ids: number[] = possibleJoins[idx]

  //   // We check if our selected join is in the master tree, if it is we select it and continue, if it isn't we expand to the node and perform random simulations
  //   const currentJoinState: number[][] = [...JSON.parse(JSON.stringify(action.context.getEpisodeState().joinIndexes))];

  //   const modelHolder: modelHolder = action.context.getModelHolder();
  //   const clonedActionForSearch: IActionRdfJoin = _.cloneDeep(action);
  //   // await this.MCTSearch(action, currentJoinState, ids, modelHolder);
  //   await this.MCTSOffline(clonedActionForSearch, modelHolder, clonedActionForSearch.context.getMasterTree());
  //   const indexJoins: number[] = ids.map(a => action.context.getNodeIdMappingKey(a)!).sort();
    
  //   // Add node to statespace tree, will need to be in loop and removed too
  //   const featureNode: number[] = [predictedValueArray[idx], 0,0,0,0,0,0,1];
  //   const newParent: NodeStateSpace = new NodeStateSpace(action.context.getEpisodeState().numNodes, featureNode, 0);
  //   action.context.getEpisodeState().addParent(possibleJoins[idx], newParent);

  //   // Make copy of our joinState as key to check if it is explored
  //   currentJoinState.push(indexJoins);

  //   predictedValueTensor.dispose(); predictedProbabilityTensor.dispose(); softMaxTensor.dispose();

  //   /* Add new explored node to the stateSpace*/
  //   // const featureNode: number[] = [predictedValueArray[idx], 0,0,0,0,0,0,1];
  //   // const newParent: NodeStateSpace = new NodeStateSpace(action.context.getEpisodeState().numNodes, featureNode, 0);
  //   // action.context.getEpisodeState().addParent(possibleJoins[idx], newParent);

  //   // Add self connection to explored join, experimental
  //   const adjMatrixCopy = action.context.getEpisodeState().adjacencyMatrix;
  //   adjMatrixCopy[adjMatrixCopy.length-1][adjMatrixCopy[0].length-1] = 1;


  //   this.prevEstimatedCost = predictedValueArray[idx];
    
  //   return [predictedValueArray[idx], predictedProbabilityArray[idx]]
  // }

    // private async expandChildStates(action: IActionRdfJoin, modelHolder: modelHolder){
  //   // Clone the state so we can recursively search
  //   const episodeState = _.cloneDeep(action.context.getEpisodeState());
  //   let nodeIndexes: number[] = [... Array(episodeState.numNodes).keys()];
  //   const possibleChildren: number[][] = this.getPossibleChildren(nodeIndexes, episodeState);

  //   let priorProbabilitiesEstimated: number[] = []
  //   for (const join of possibleChildren){
  //     // Get probabilties for each child
  //     const priorProbability = await this.getOutputChild(action, join, modelHolder);
  //     priorProbabilitiesEstimated.push(priorProbability[1]);
  //   }
  //   const priorProbabilitiesTensor: tf.Tensor = tf.tensor(priorProbabilitiesEstimated);
  //   const probabilitiesTensor: tf.Tensor = this.softMax(priorProbabilitiesTensor);
  //   // There is something wrong with probabilities, gives NaN sometimes;
  //   const probabilities = probabilitiesTensor.dataSync();

  //   for(let i=0;i<probabilities.length;i++){
  //     const joinId = possibleChildren[i];
  //     const joinEntryIndexes: number[] = joinId.map(x => action.context.getNodeIdMapping().get(x)!);
  //     // Get key to identify the explored child nodes
  //     const newJoinIndexes = [...episodeState.joinIndexes];
  //     newJoinIndexes.push(joinEntryIndexes);


  //     let featureMatrix: number[][] = [];
  //     for(let i=0;i<action.context.getEpisodeState().nodesArray.length;i++){
  //       featureMatrix.push(action.context.getEpisodeState().nodesArray[i].features);
  //     }
  //     // Initialise with empty matrixes, since either this join will be explored or unused, so matrix will be initialised in another execution or never used
  //     const initialisationNode: MCTSJoinPredictionOutput = {N:0, state: newJoinIndexes, predictedValueJoin: 0, predictedProbability: probabilities[i]};
  //     action.context.setJoinStateMasterTree(initialisationNode, [[]], [[]]);
  //   }
  //   priorProbabilitiesTensor.dispose(); probabilitiesTensor.dispose();
  // } 

  
  // public async MCTSearch(action: IActionRdfJoin, currentJoinState: number[][], prevJoin: number[], modelHolder: modelHolder){
  //   /**
  //    * Calling this from root requires copied action
  //    */
  //   // NOTE WE SHOULD NOT WORK WITH CLONED THING IF WE DO RECURSIVE, THIS WILL PREVENT `LOWER` NODES TO BE ADDED TO JOIN STATE AT ROOT!!

  //   const MCTSMasterTree: MCTSJoinInformation = action.context.getJoinStateMasterTree(currentJoinState)!;
  //   const clonedAction: IActionRdfJoin = _.cloneDeep(action);
  //   let nodeValue = [0,0];
  //   // Note that due to the current nature of MCTS we always end up with the 'root' join being explored, we will change that when algorithm is fully done
  //   if (MCTSMasterTree.priorProbability==0){
  //     nodeValue = await this.getOutputChild(clonedAction, prevJoin, modelHolder);
  //   }

  //   // Add join to execute to the local clone of the tree
  //   await this.executeJoinMCTS(clonedAction, prevJoin);
  //   const state: StateSpaceTree = clonedAction.context.getEpisodeState()

  //   if (MCTSMasterTree.priorProbability<0){
  //     console.error("We have prior probability smaller than 0 (Actor)");
  //   }

  //   if (MCTSMasterTree.priorProbability>0){
  //     // 2. If we already explored, call search recursively on selected child node based on UCT
  //     let nodeIndexes: number[] = [... Array(state.numNodes).keys()];
  //     const possibleChildren: number[] = nodeIndexes.filter(item => !state.nodesArray[item].joined);
  //     let possibleJoins: number[][] = this.getPossibleJoins(possibleChildren);

  //     let maxU = 0;
  //     let bestJoin = [0,0];

  //     for (const join of possibleJoins){
  //       const joinEntryIndexes: number[] = join.map(x => clonedAction.context.getNodeIdMapping().get(x)!);
  //       const newJoinKey: number[][] = [...currentJoinState];
  //       newJoinKey.push(joinEntryIndexes);
  //       const joinInformation: MCTSJoinInformation = action.context.getJoinStateMasterTree(newJoinKey);
  //       // Here the UCB value is calculated according to https://web.stanford.edu/~surag/posts/alphazero.html, note that the parent # visits is equal to the total visits of 
  //       // All child actions.
  //       const U = joinInformation.meanValueJoin + this.explorationDegree * joinInformation.priorProbability*(Math.sqrt(MCTSMasterTree.N)/1+joinInformation.N);
  //       if (U >= maxU){
  //         maxU = U;
  //         bestJoin = join;
  //       }
  //     }
  //     // Execute the join and pass the resulting state, MCTSMasterTree and Model to next execution
  //   }

  //   else{
  //     // 3. If we haven't, expand the child states and add them to MCTS
  //     await this.expandChildStates(clonedAction, modelHolder)
  //     const adjMatrix: number[][] = clonedAction.context.getEpisodeState().adjacencyMatrix;
  //     let featureMatrix: number[][] = [];
  //     for(let i=0;i<clonedAction.context.getEpisodeState().nodesArray.length;i++){
  //       featureMatrix.push(clonedAction.context.getEpisodeState().nodesArray[i].features);
  //     }
  //     // Add unexplored node to join state properly
  //     action.context.setJoinStateMasterTree({N:1, state:state.getJoinIndexes(), predictedValueJoin: nodeValue[0], predictedProbability: MCTSMasterTree.priorProbability},featureMatrix,adjMatrix)
  //     // Still need to propogate the found values upwards after we have way of choosing using UCB
  //     console.log(state.getJoinIndexes());
  //     console.log(state.getJoinIndexes().length);
  //   }
  // }

  
  // private async getOutputChild(action: IActionRdfJoin, join: number[], modelHolder: modelHolder): Promise<number[]>{
  //   /*
  //   1. clone action
  //   2. execute proposed join on action
  //   3. Get features matrices / adjmatrices
  //   4. Return probability of child
  //   */
  //   // Clone the action so we can recursively search and join entries
  //   const actionToExplore: IActionRdfJoin = _.cloneDeep(action);

  //   // Execute the join
  //   await this.executeJoinMCTS(actionToExplore, join);

  //   let featureMatrix: number[][] = [];

  //   for(let i=0;i<actionToExplore.context.getEpisodeState().nodesArray.length;i++){
  //     featureMatrix.push(actionToExplore.context.getEpisodeState().nodesArray[i].features);
  //   }

  //   /* Execute forward pass */
  //   const adjTensor: tf.Tensor2D = tf.tensor2d(actionToExplore.context.getEpisodeState().adjacencyMatrix);
  //   const inputFeaturesTensor: tf.Tensor2D = tf.tensor2d(featureMatrix, [actionToExplore.context.getEpisodeState().numNodes,featureMatrix[0].length]);
  //   const forwardPassOutput: tf.Tensor[] = modelHolder.getModel().forwardPassTest(inputFeaturesTensor, adjTensor) as tf.Tensor[];

  //   const valueOutput = (await forwardPassOutput[0].data())[featureMatrix.length-1]
  //   const policyOutput = (await forwardPassOutput[1].data())[featureMatrix.length-1];

  //   inputFeaturesTensor.dispose(); forwardPassOutput.map(x => x.dispose()); adjTensor.dispose(); 
  //   return [valueOutput, policyOutput];
  // }
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
  /**
   * Degree of exploration during offline training
   */
  explorationDegree: number;
}

interface tempContextEntries{
  type: string;
  input: RdfJs.Quad[];
}

// interface tempQuad{
//   termType: string;
//   value: string;
//   subject: Variable { termType: 'Variable', value: 'v0' },
//   predicate: NamedNode {
//     termType: 'NamedNode',
//     value: 'http://schema.org/caption'
//   },
//   object: Variable { termType: 'Variable', value: 'v1' },
//   graph: DefaultGraph { termType: 'DefaultGraph', value: '' },
//   type: 'pattern'

// }

function sum(arr: number[]) {
  var result = 0, n = arr.length || 0; //may use >>> 0 to ensure length is Uint32
  while(n--) {
    result += +arr[n]; // unary operator to ensure ToNumber conversion
  }
  return result;
}

