import { StatisticsHolder } from '@comunica/actor-context-preprocess-set-defaults';
import type { ILink } from '@comunica/bus-rdf-resolve-hypermedia-links';
import { KeysStatisticsTracker, KeysTrackableStatistics } from '@comunica/context-entries';
import { ActionContext, Bus } from '@comunica/core';
import type { IDiscoverEventData } from '@comunica/types';
import { ActorContextPreprocessStatisticLinkDiscovery } from '../lib';
import { StatisticLinkDiscovery } from '../lib/StatisticLinkDiscovery';

describe('StatisticLinkDiscovery', () => {
  let bus: any;
  let statisticLinkDiscovery: StatisticLinkDiscovery;
  let mockLogFunction: ((message: string, data?: (() => any)) => void);

  beforeEach(() => {
    bus = new Bus({ name: 'bus' });
    jest.useFakeTimers();
    jest.setSystemTime(new Date('2021-01-01T00:00:00Z').getTime());
    mockLogFunction = jest.fn((message: string, data?: (() => any)) => {});
    statisticLinkDiscovery = new StatisticLinkDiscovery(mockLogFunction);
  });

  describe('An StatisticLinkDereference instance should', () => {
    let actor: ActorContextPreprocessStatisticLinkDiscovery;
    let parent: ILink;
    let child: ILink;
    let discoveredEventData: IDiscoverEventData;
    let cb: (arg0: IDiscoverEventData) => void;

    beforeEach(() => {
      actor = new ActorContextPreprocessStatisticLinkDiscovery({ name: 'actor', bus });
      discoveredEventData = {
        edge: [ 'parent', 'child' ],
        metadataChild: [{}],
        metadataParent: [{}],
      };
      parent = {
        url: 'parent',
        metadata: {
          key: 2,
        },
      };
      child = {
        url: 'child',
        metadata: {
          childkey: 5,
        },
      };
      cb = jest.fn((arg0: IDiscoverEventData) => {});
    });

    it('attach an event listener', () => {
      statisticLinkDiscovery.addListener(cb);
      expect(statisticLinkDiscovery.statisticEvents.listeners('data')).toEqual(
        [ cb ],
      );
    });

    it('update edge list links', () => {
      statisticLinkDiscovery.setDiscoveredLink(child, parent);
      expect(statisticLinkDiscovery.edgeList).toEqual(
        new Set([ JSON.stringify([ parent.url, child.url ]) ]),
      );
    });

    it('should not update edge list when duplicate edge is found', () => {
      statisticLinkDiscovery.setDiscoveredLink(child, parent);
      statisticLinkDiscovery.setDiscoveredLink(child, parent);
      expect(statisticLinkDiscovery.edgeList).toEqual(
        new Set([ JSON.stringify([ parent.url, child.url ]) ]),
      );
      expect(statisticLinkDiscovery.metadata).toEqual({
        child: [{
          childkey: 5,
          discoveredTimestamp: Date.now(),
          discoverOrder: 0,
        }],
      });
    });

    it('update metadata links', () => {
      statisticLinkDiscovery.setDiscoveredLink(child, parent);
      expect(statisticLinkDiscovery.metadata).toEqual({
        child: [{
          childkey: 5,
          discoveredTimestamp: Date.now(),
          discoverOrder: 0,
        }],
      });
    });

    it('correctly aggregate metadata', () => {
      const parent2: ILink = { url: 'parent2', metadata: {}};
      statisticLinkDiscovery.setDiscoveredLink(child, parent);
      // Metadata upon second discovery from different parent can change
      child.metadata = { childkey: 10, extrakey: 'test' };
      statisticLinkDiscovery.setDiscoveredLink(child, parent2);

      expect(statisticLinkDiscovery.edgeList).toEqual(
        new Set([ JSON.stringify([ parent.url, child.url ]), JSON.stringify([ parent2.url, child.url ]) ]),
      );

      expect(statisticLinkDiscovery.metadata).toEqual(
        { child:
          [{
            childkey: 5,
            discoveredTimestamp: Date.now(),
            discoverOrder: 0,
          }, {
            childkey: 10,
            extrakey: 'test',
            discoveredTimestamp: Date.now(),
            discoverOrder: 1,
          }],
        },
      );
    });

    it('emit event', () => {
      statisticLinkDiscovery.addListener(cb);
      statisticLinkDiscovery.emit(discoveredEventData);
      expect(cb).toHaveBeenCalledWith({
        edge: [ 'parent', 'child' ],
        metadataChild: [{}],
        metadataParent: [{}],
      });
    });

    it('call log function', () => {
      statisticLinkDiscovery.setDiscoveredLink(child, parent);
      expect(mockLogFunction).toHaveBeenCalledWith('Discover Event', expect.any(Function));
    });

    it('correctly log discovered link', async() => {
      const contextIn = new ActionContext({ [KeysStatisticsTracker.statistics.name]: new StatisticsHolder() });
      const { context: contextOut } = await actor.run({ context: contextIn });
      const spy = jest.spyOn(actor, <any> 'logInfo');

      const statisticsHolder: StatisticsHolder = contextOut.get(
        KeysStatisticsTracker.statistics,
      )!;

      const statisticLinkDiscoveryFromActor: StatisticLinkDiscovery = statisticsHolder.get(
        KeysTrackableStatistics.discoveredLinks,
      )!;

      statisticLinkDiscoveryFromActor.setDiscoveredLink(child, parent);
      expect(spy).toHaveBeenCalledWith(contextIn, 'Discover Event', expect.anything());
      // Test the anonymous function passed to variable 'data'
      expect((<Function> spy.mock.calls[0][2])()).toEqual({
        edge: [ 'parent', 'child' ],
        metadataChild: [{
          childkey: 5,
          discoverOrder: 0,
          discoveredTimestamp: Date.now(),
        }],
        metadataParent: undefined,
      });
    });
  });

  afterEach(() => {
    jest.useRealTimers();
  });
});
