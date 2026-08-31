import { ActorContextPreprocess, IActionContextPreprocess, IActorContextPreprocessOutput, IActorContextPreprocessArgs } from '@comunica/bus-context-preprocess';
import { TestResult, IActorTest, passTestVoid } from '@comunica/core';
import { StemsAdaptiveJoinComponent } from './StemsAdaptiveJoinComponent';
import { AdaptiveJoinController } from './AdaptiveJoinController';
import { KeysRdfJoin } from '@comunica/context-entries';

/**
 * A comunica Set Adaptive Join Controller Context Preprocess Actor.
 */
export class ActorContextPreprocessSetStemsAdaptiveJoinController extends ActorContextPreprocess {
  public constructor(args: IActorContextPreprocessArgs) {
    super(args);
  }

  public async test(action: IActionContextPreprocess): Promise<TestResult<IActorTest>> {
    return passTestVoid();
  }

  public async run(action: IActionContextPreprocess): Promise<IActorContextPreprocessOutput> {
    let context = action.context;

    context = context.set(KeysRdfJoin.adaptiveJoinController, new AdaptiveJoinController());

    return { context };
  }
}
