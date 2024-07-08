import { StatisticsHolder } from '@comunica/actor-context-preprocess-set-defaults';
import { KeysStatisticsTracker, KeysTrackableStatistics } from '@comunica/context-entries';
import { ActionContext, Bus } from '@comunica/core';
import { ActorContextPreprocessAggregateStatisticTraversedTopology } from '@comunica/actor-context-preprocess-aggregate-statistic-traversed-topology'
import { AggregateStatisticTraversedTopology } from '../lib/AggregateStatisticTraversedTopology';
import { StatisticLinkDiscovery } from '@comunica/actor-context-preprocess-statistic-link-discovery';
import { StatisticLinkDereference } from '@comunica/actor-context-preprocess-statistic-link-dereference';

describe('AggregateStatisticTraversedTopology', () => {
  let bus: any;
  let linkDiscovery: StatisticLinkDiscovery;
  let linkDereference: StatisticLinkDereference;
  let statisticsHolder: StatisticsHolder;

  beforeEach(() => {
    bus = new Bus({ name: 'bus' });
    linkDiscovery = new StatisticLinkDiscovery();
    linkDereference = new StatisticLinkDereference();
    statisticsHolder = new StatisticsHolder();
    statisticsHolder.set(KeysTrackableStatistics.discoveredLinks, linkDiscovery);
    statisticsHolder.set(KeysTrackableStatistics.dereferencedLinks, linkDereference);

  });

  describe('An AggregateStatisticTraversedTopology instance', () => {
    let actor: ActorContextPreprocessAggregateStatisticTraversedTopology;

    beforeEach(() => {
      actor = new ActorContextPreprocessAggregateStatisticTraversedTopology({ name: 'actor', bus });
    });

    it('should test', async() => {
      await expect(actor.test({ context: new ActionContext() })).resolves.toBeTruthy();
    });

    describe('run', () => {
      it('with only a statisticsHolder with needed statistic trackers', async() => {
        const contextIn = new ActionContext({ [KeysStatisticsTracker.statistics.name]: statisticsHolder });
        const { context: contextOut } = await actor.run({ context: contextIn });
      
        const expectedStatisticHolder = new StatisticsHolder([
          [KeysTrackableStatistics.discoveredLinks.name, linkDiscovery], 
          [KeysTrackableStatistics.dereferencedLinks.name, linkDereference],
          [KeysTrackableStatistics.traversedTopology.name,
            new AggregateStatisticTraversedTopology(linkDiscovery, linkDereference)],
        ]);

        expect(contextOut).toEqual(new ActionContext({
          [KeysStatisticsTracker.statistics.name]: expectedStatisticHolder,
        }));

      });

      it('should error with empty context', async() => {
        await expect(actor.run({ context: new ActionContext() })).rejects.toThrow(new Error(
          'Tried to track topology statistic without statisticsHolder object in context',
        ));
      });

      it('should error with empty statisticHolder', async () => {
        const contextIn = new ActionContext( { [KeysStatisticsTracker.statistics.name]: new StatisticsHolder() })
        await expect(actor.run({ context: contextIn })).rejects.toThrow(new Error(
          'Statistic aggregator did not find required statistics trackers in context',
        ));
      });

      it('should error with statisticsHolder with not all statistic trackers', async () => {
        const contextIn = new ActionContext( { [KeysStatisticsTracker.statistics.name]: new StatisticsHolder() })
        await expect(actor.run({ context: contextIn })).rejects.toThrow(new Error(
          'Statistic aggregator did not find required statistics trackers in context',
        ));
      })
    });
  });
});
