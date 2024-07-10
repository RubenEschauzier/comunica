import type { StatisticsHolder } from '@comunica/actor-context-preprocess-set-defaults';
import type { IActionContextPreprocess, IActorContextPreprocessOutput, IActorContextPreprocessArgs }
  from '@comunica/bus-context-preprocess';
import { ActorContextPreprocess } from '@comunica/bus-context-preprocess';
import { KeysStatisticsTracker, KeysTrackableStatistics } from '@comunica/context-entries';
import type { IActorTest } from '@comunica/core';
import type { IStatisticDereferencedLinks, IStatisticDiscoveredLinks } from '@comunica/types';
import { AggregateStatisticTraversedTopology } from './AggregateStatisticTraversedTopology';

/**
 * A comunica Aggregate Statistic Traversed Topology Context Preprocess Actor.
 */
export class ActorContextPreprocessAggregateStatisticTraversedTopology extends ActorContextPreprocess {
  public constructor(args: IActorContextPreprocessArgs) {
    super(args);
  }

  public async test(_action: IActionContextPreprocess): Promise<IActorTest> {
    return true;
  }

  public async run(action: IActionContextPreprocess): Promise<IActorContextPreprocessOutput> {
    const statisticsHolder: StatisticsHolder | undefined = action.context.get(KeysStatisticsTracker.statistics);

    if (!statisticsHolder) {
      throw new Error('Tried to track topology statistic without statisticsHolder object in context');
    }

    if (!statisticsHolder.get(KeysTrackableStatistics.discoveredLinks) ||
        !statisticsHolder.get(KeysTrackableStatistics.dereferencedLinks)) {
      throw new Error('Statistic aggregator did not find required statistics trackers in context');
    }

    const discoverStatistic: IStatisticDiscoveredLinks = statisticsHolder.get(KeysTrackableStatistics.discoveredLinks)!;
    const dereferenceStatistic: IStatisticDereferencedLinks = statisticsHolder.get(
      KeysTrackableStatistics.dereferencedLinks,
    )!;

    // Create AggregateStatisticTraversedTopology with bound log function with context entry already filled in
    const statisticLinkDereference: AggregateStatisticTraversedTopology = new AggregateStatisticTraversedTopology(
      discoverStatistic,
      dereferenceStatistic,
      (message: string, data?: (() => any)) => this.logInfo(action.context, message, data),
    );

    statisticsHolder.set(KeysTrackableStatistics.traversedTopology, statisticLinkDereference);

    return { context: action.context };
  }
}
