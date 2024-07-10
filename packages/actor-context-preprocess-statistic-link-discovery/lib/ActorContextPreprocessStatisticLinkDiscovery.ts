import type { StatisticsHolder } from '@comunica/actor-context-preprocess-set-defaults';
import type { IActionContextPreprocess, IActorContextPreprocessOutput, IActorContextPreprocessArgs } from '@comunica/bus-context-preprocess';
import { ActorContextPreprocess } from '@comunica/bus-context-preprocess';
import { KeysStatisticsTracker, KeysTrackableStatistics } from '@comunica/context-entries';
import type { IActorTest } from '@comunica/core';
import { StatisticLinkDiscovery } from './StatisticLinkDiscovery';
import { IActionContext } from '@comunica/types';

/**
 * A comunica Statistic Link Discovery Context Preprocess Actor.
 */
export class ActorContextPreprocessStatisticLinkDiscovery extends ActorContextPreprocess {

  public constructor(args: IActorContextPreprocessArgs) {
    super(args);
  }

  public async test(_action: IActionContextPreprocess): Promise<IActorTest> {
    return true;
  }

  public async run(action: IActionContextPreprocess): Promise<IActorContextPreprocessOutput> {
    const statisticsHolder: StatisticsHolder | undefined = action.context.get(KeysStatisticsTracker.statistics);

    if (!statisticsHolder) {
      throw new Error('Tried to track link discovery statistic without statisticsHolder object in context');
    }

    // Create StatisticLinkDiscovery with bound log function with context entry already filled in
    const statisticLinkDiscovery: StatisticLinkDiscovery = new StatisticLinkDiscovery(
      ((message: string, data?: (() => any)) => this.logInfo(action.context, message, data)).bind(this)
    );

    statisticsHolder.set(KeysTrackableStatistics.discoveredLinks, statisticLinkDiscovery);

    return { context: action.context };
  }
}
