import type { IActionContextPreprocess, IActorContextPreprocessOutput, IActorContextPreprocessArgs }
  from '@comunica/bus-context-preprocess';
import { ActorContextPreprocess } from '@comunica/bus-context-preprocess';
import { KeysInitQuery, KeysStatisticsTracker, KeysTrackableStatistics } from '@comunica/context-entries';
import type { IActorTest } from '@comunica/core';
import type { IActionContextKey } from '@comunica/types';
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
    const statisticsMap = <Map<IActionContextKey<any>, any>> action.context.get(KeysStatisticsTracker.statistics)!;
    statisticsMap.set(
      KeysTrackableStatistics.dereferencedLinks,
      new StatisticLinkDereference(
        action.context.get(KeysInitQuery.queryString)!,
        action.context.get(KeysStatisticsTracker.statisticsLogger),
      ),
    );
    return { context: action.context };
  }
}
