import type { IActionContextPreprocess, IActorContextPreprocessOutput, IActorContextPreprocessArgs } from '@comunica/bus-context-preprocess';
import { ActorContextPreprocess } from '@comunica/bus-context-preprocess';
import { KeysInitQuery, KeysStatisticsTracker, KeysTrackableStatistics } from '@comunica/context-entries';
import type { IActorTest } from '@comunica/core';
import { StatisticLinkDiscovery } from './StatisticLinkDiscovery';
import { StatisticsHolder } from '@comunica/actor-context-preprocess-set-defaults';


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
    const statisticsHolder: StatisticsHolder = action.context.get(KeysStatisticsTracker.statistics)!;
    statisticsHolder.set(KeysTrackableStatistics.discoveredLinks, new StatisticLinkDiscovery());

    return { context: action.context };
  }
}
