import {Actor, IAction, IActorOutput, IActorReply, IActorTest, IMediatorArgs, Mediator } from '@comunica/core';
import { ActorModelInference, IActionModelInference, IActorModelInferenceOutput, IActorModelInferenceTest } from '@comunica/bus-model-inference';
/**
 * A comunica model inference mediator, that will select the model that can perform a certain task
 */
export class MediatorModelInference<A extends Actor<I, T, O>, I extends IAction, T extends IActorTest, O extends IActorOutput>
  extends Mediator<ActorModelInference, IActionModelInference, IActorTest, IActorModelInferenceOutput> {
  public constructor(args: IMediatorArgs<ActorModelInference, IActionModelInference, IActorTest, IActorModelInferenceOutput>) {
    super(args);
  }
  
  protected async mediateWith(action: IActionModelInference, 
    testResults: IActorReply<ActorModelInference, IActionModelInference, IActorModelInferenceTest, IActorModelInferenceOutput>[]): Promise<ActorModelInference> {
    const errors: Error[] = [];
    const promises = testResults
      .map(({ reply }) => reply)
      .map(promise => promise.catch(error => {
        errors.push(error);
      }));
    const expectedPerformance = await Promise.all(promises);

    // Determine index with lowest cost
    let minIndex = -1;
    let minValue = Number.POSITIVE_INFINITY;
    for (const [ i, cost ] of expectedPerformance.entries()) {
      if (cost !== undefined && (minIndex === -1 || cost.expectedPerformance < minValue)) {
        minIndex = i;
        minValue = cost.expectedPerformance;
      }
    }

    // Reject if all actors rejected
    if (minIndex < 0) {
      throw new Error(`All actors rejected their test in ${this.name}\n${
        errors.map(error => error.message).join('\n')}`);
    }

    // Return actor with lowest cost
    const bestActor = testResults[minIndex].actor;

    Actor.getContextLogger(action.context)?.debug(`Determined model '${bestActor.physicalName}'`, {
      bestActor: bestActor,
      bestExpectedPerformance: minValue,
    })
  
    return bestActor;
  }
}
