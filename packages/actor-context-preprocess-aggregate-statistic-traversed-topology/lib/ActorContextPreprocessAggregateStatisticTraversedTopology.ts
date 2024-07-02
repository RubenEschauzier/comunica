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
    if (!action.context.get(KeysTrackableStatistics.discoveredLinks) || 
    !action.context.get(KeysTrackableStatistics.dereferencedLinks)){
      throw new Error("Statistic aggregator did not find required statistics trackers in context");
    }
    return true;
  }

  public async run(action: IActionContextPreprocess): Promise<IActorContextPreprocessOutput> {
    const statisticsMap = <Map<IActionContextKey<any>, any>> action.context.get(KeysStatisticsTracker.statistics)!;

    const discoverStatistic: IStatisticDiscoveredLinks = action.context.get(KeysTrackableStatistics.discoveredLinks)!;
    const dereferenceStatistic: IStatisticDereferencedLinks = action.context.get(KeysTrackableStatistics.dereferencedLinks)!;

    statisticsMap.set(
      KeysTrackableStatistics.traversedTopology,
      new AggregateStatisticTraversedTopology(action.context.get(KeysInitQuery.queryString)!,
      discoverStatistic, dereferenceStatistic),
    );
    return { context: action.context };
  }
}

