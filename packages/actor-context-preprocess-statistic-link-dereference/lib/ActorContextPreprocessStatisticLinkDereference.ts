import type { StatisticsHolder } from '@comunica/actor-context-preprocess-set-defaults';
import type {
  IActionContextPreprocess,
  IActorContextPreprocessOutput,
  IActorContextPreprocessArgs }
  from '@comunica/bus-context-preprocess';
import { ActorContextPreprocess } from '@comunica/bus-context-preprocess';
import { KeysStatisticsTracker, KeysTrackableStatistics } from '@comunica/context-entries';
import type { IActorTest } from '@comunica/core';
import { StatisticLinkDereference } from './StatisticLinkDereference';

/**
 * A comunica Statistic Link Dereference Context Preprocess Actor.
 */
export class ActorContextPreprocessStatisticLinkDereference extends ActorContextPreprocess {
  public constructor(args: IActorContextPreprocessArgs) {
    super(args);
  }

  public async test(_action: IActionContextPreprocess): Promise<IActorTest> {
    return true;
  }

  public async run(action: IActionContextPreprocess): Promise<IActorContextPreprocessOutput> {
    const statisticsHolder: StatisticsHolder | undefined = action.context.get(KeysStatisticsTracker.statistics);

    if (!statisticsHolder) {
      throw new Error('Tried to track link dereference statistic without statisticsHolder object in context');
    }

    // Create StatisticLinkDereference with bound log function with context entry already filled in
    const statisticLinkDereference: StatisticLinkDereference = new StatisticLinkDereference(
      ((message: string, data?: (() => any)) => this.logInfo(action.context, message, data)),
    );

    statisticsHolder.set(KeysTrackableStatistics.dereferencedLinks, statisticLinkDereference);

    return { context: action.context };
  }
}
