import type { ActorRdfJoin, IActionRdfJoin } from '@comunica/bus-rdf-join';
import { KeysQueryOperation } from '@comunica/context-entries';
import { ActionContext, IActorReply, IMediatorArgs, Logger } from '@comunica/core';
import { Actor, Mediator } from '@comunica/core';
import type { IMediatorTypeJoinCoefficients } from '@comunica/mediatortype-join-coefficients';
import type { IQueryOperationResult, MetadataBindings } from '@comunica/types';
import { EpisodeLogger } from '../../model-trainer/lib';
import { NodeStateSpace, StateSpaceTree } from './StateSpaceTree';
import {cloneDeep} from 'lodash';

/**
 * A comunica mediator that mediates for the reinforcement learning-based multi join actor
 * @template A The type of actor to mediator over.
 * @template I The input type of an actor.
 * @template T The test type of an actor.
 * @template O The output type of an actor.
 */
// Note to self: Get logger to make timings from the ActorInitQueryBase.ts and use the join logger from this one (or use another 'official' logger).
// Combine results by writing to the same file
// Ruben E: Changed to inplace at: Bus query operation, reinforcement learning actor and either slice or project actor.
export class MediatorJoinReinforcementLearning
  extends Mediator<ActorRdfJoin, IActionRdfJoin, IMediatorTypeJoinCoefficients, IQueryOperationResult> {
  public joinState: StateSpaceTree;
  public offlineTrain: boolean;
  public readonly cpuWeight: number;
  public readonly memoryWeight: number;
  public readonly timeWeight: number;
  public readonly ioWeight: number;

  public constructor(args: IMediatorJoinCoefficientsFixedArgs) {
    super(args);
    this.offlineTrain = args['offlineTrain'];

  }

  protected async mediateWith(action: IActionRdfJoin, testResults: IActorReply<ActorRdfJoin, IActionRdfJoin, IMediatorTypeJoinCoefficients, IQueryOperationResult>[],
    ): Promise<ActorRdfJoin> {
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

    // if (bestActor instanceof ActorRdfJoinInnerMultiReinforcementLearning){
    //   // Still need to check if we even have a state
    //   this.joinState = bestActor.joinState;
    //   const numNodes: number = this.joinState.nodesArray.length
    //   this.episodeLogger.logSelectedJoin(this.joinState.nodesArray[numNodes-1].children.map(x => x.id));
    // }
    return bestActor;
    
  }
  protected updateState(newState: StateSpaceTree){
    this.joinState = newState
  }

  public async mediateActor(action: IActionRdfJoin): Promise<ActorRdfJoin> {
    /* 
    * Mediate the actor.
    * If we perform offline training we simulate 'x' join plans according to our current neural network predictions. We do so by calling the mediateWith function 'x'
    * times from the start of the join execution plan.
    */
    /* At first join execution we initialise our StateSpaceTree with just leave nodes */
    if (!action.context.getEpisodeState()){
      let joinStateTest = new StateSpaceTree();
      for (let i=0; i<action.entries.length;i++){
        // Create nodes with estimated cardinality as feature
        const cardinalityNode: number = (await action.entries[i].output.metadata()).cardinality.value;
        let newNode: NodeStateSpace = new NodeStateSpace(i, cardinalityNode);
        newNode.setDepth(0);
        joinStateTest.addLeave(newNode);
      }
      /* Set the number of leave nodes in the tree, this will not change during execution */
      joinStateTest.setNumLeaveNodes(joinStateTest.numNodes); 

      action.context.setEpisodeState(joinStateTest);
    }
    return await this.mediateWith(action, this.publish(action));
        // const numNodesPerSim = 2;
        // const numSimsToExecute = Math.min(numNodesPerSim, action.entries.length);
        // let bestActor = null;
        // for (let i=0; i<numSimsToExecute;i++){
        //   // const actionToPass: IActionRdfJoin = {...action}
        //   const actionToPass = cloneDeep(action); 
  
        //   console.log(`NumNodes of Action: ${action.context.getEpisodeState().numNodes}`);
        //   console.log(`Before setting ID: ${actionToPass.context.getEpisodeState().testId}`);
        //   // console.log(action.context.getEpisodeState());
        //   /* Make deep copy of action episode to allow recursive join plan exploration */
        //   // const copyOfEpisode: EpisodeLogger = new EpisodeLogger();
        //   // copyOfEpisode.setState(actionToPass.context.getEpisodeState());
        //   // actionToPass.context = new ActionContext(actionToPass.context, copyOfEpisode);
          
        //   /* Assign ID to each joinState for debugging purposes (SHOULD BE GONE IF PUBLISHED)*/
        //   actionToPass.context.getEpisodeState().testId = i;
  
        
        //   console.log(`Simulation ${i}.`);
        //   console.log("Before mediateWith call");
        //   console.log(`Underlying action ID: ${action.context.getEpisodeState().testId}`);
        //   console.log(actionToPass.context.getEpisodeState().numNodes)
        //   console.log(actionToPass.context.getEpisodeState().testId);
  
        //   bestActor = await this.mediateWith(actionToPass, this.publish(actionToPass));
        //   /* Get explored join */
        //   const exploredJoin: StateSpaceTree = actionToPass.context.getEpisodeState();
        //   console.log("After mediateWith Call:");
        //   console.log(exploredJoin.numNodes);
        //   console.log(actionToPass.context.getEpisodeState().testId);
  
        // }
        /* We initialise this bestActor with null, but it will never keep this value */
    }
  }

  // protected mediateWithOfflineTrain(action: IActionRdfJoin, testResults: IActorReply<ActorRdfJoin, IActionRdfJoin, IMediatorTypeJoinCoefficients, IQueryOperationResult>[],
  //   ): Promise<ActorRdfJoin> {
  //   /* Constants that should be parameters*/
  //   const numNodesPerSim = 2;
  //   for (let i=0; i<Math.min(numNodesPerSim, action.entries.length);i++){
  //       const contextToPass = {... action.context}
  //   }
  // }

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
  /**
   * If we are offline training the model
   */
  offlineTrain: boolean;
}