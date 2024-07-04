import { ActorContextPreprocess, IActionContextPreprocess, IActorContextPreprocessOutput, IActorContextPreprocessArgs } from '@comunica/bus-context-preprocess';
import { KeysInitQuery, KeysStatisticsTracker, KeysTrackableStatistics } from '@comunica/context-entries';
import { IActorTest } from '@comunica/core';
import type { IActionContextKey, IStatisticDereferencedLinks, IStatisticDiscoveredLinks } from '@comunica/types';
import { AggregateStatisticTraversedTopology } from './AggregateStatisticTraversedTopology';
/**
 * A comunica Aggregate Statistic Traversed Topology Context Preprocess Actor.
 */
export class ActorContextPreprocessAggregateStatisticTraversedTopology extends ActorContextPreprocess {
  public constructor(args: IActorContextPreprocessArgs) {
    super(args);
  }

  public async test(action: IActionContextPreprocess): Promise<IActorTest> {
    return true;
  }

  public async run(action: IActionContextPreprocess): Promise<IActorContextPreprocessOutput> {
    const statisticsMap = <Map<IActionContextKey<any>, any>> action.context.get(KeysStatisticsTracker.statistics)!;
    if (!statisticsMap.get(KeysTrackableStatistics.discoveredLinks) || 
    !statisticsMap.get(KeysTrackableStatistics.dereferencedLinks)
    ){
      throw new Error("Statistic aggregator did not find required statistics trackers in context");
    }

    const discoverStatistic: IStatisticDiscoveredLinks = statisticsMap.get(KeysTrackableStatistics.discoveredLinks)!;
    const dereferenceStatistic: IStatisticDereferencedLinks = statisticsMap.get(KeysTrackableStatistics.dereferencedLinks)!;

    statisticsMap.set(KeysTrackableStatistics.traversedTopology,
      new AggregateStatisticTraversedTopology(discoverStatistic, dereferenceStatistic));
    return { context: action.context };
  }
}

