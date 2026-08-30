import { ActorContextPreprocess, IActionContextPreprocess, IActorContextPreprocessOutput, IActorContextPreprocessArgs } from '@comunica/bus-context-preprocess';
import { TestResult, IActorTest, passTestVoid } from '@comunica/core';

/**
 * A comunica Set Adaptive Join Controller Context Preprocess Actor.
 */
export class ActorContextPreprocessSetStemsAdaptiveJoinController extends ActorContextPreprocess {
  public constructor(args: IActorContextPreprocessArgs) {
    super(args);
  }

  public async test(action: IActionContextPreprocess): Promise<TestResult<IActorTest>> {
    return passTestVoid(); // TODO implement
  }

  public async run(action: IActionContextPreprocess): Promise<IActorContextPreprocessOutput> {
    return { context: action.context }; // TODO implement
  }
}
