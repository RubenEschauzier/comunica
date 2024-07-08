import type { ILink } from '@comunica/bus-rdf-resolve-hypermedia-links';
import type { IDiscoverEventData } from '@comunica/types';
import { StatisticLinkDiscovery } from '../lib/StatisticLinkDiscovery';

describe('StatisticLinkDiscovery', () => {
  let statisticLinkDiscovery: StatisticLinkDiscovery;

  beforeEach(() => {
    jest.useFakeTimers();
    jest.setSystemTime(new Date('2021-01-01T00:00:00Z').getTime());
    statisticLinkDiscovery = new StatisticLinkDiscovery();
  });

  describe('An StatisticLinkDereference instance should', () => {
    let parent: ILink;
    let child: ILink;
    let discoveredEventData: IDiscoverEventData;
    let cb: (arg0: IDiscoverEventData) => void;

    beforeEach(() => {      
      discoveredEventData = {
        edge: ['parent', 'child'],
        metadataDiscoveredNode: [{}],
        metadataParentDiscoveredNode: [{}]
      }
      parent = {
        url: 'parent',
        metadata: {
            'key': 2
        }
      }
      child = {
        url: 'child',
        metadata: {
            'childkey': 5
        }
      }
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
        new Set( [ JSON.stringify([ parent.url, child.url ]) ])
      );
    });

    it('should not update edge list when duplicate edge is found', () => {
        statisticLinkDiscovery.setDiscoveredLink(child, parent);
        statisticLinkDiscovery.setDiscoveredLink(child, parent);
        expect(statisticLinkDiscovery.edgeList).toEqual(
            new Set( [ JSON.stringify([ parent.url, child.url ]) ])
        );
        expect(statisticLinkDiscovery.metadata).toEqual({
            'child': [{
                'childkey': 5, 
                'discoveredTimestamp': Date.now(),
                'discoverOrder': 0 
        }]});
    })

    it('update metadata links', () => {
        statisticLinkDiscovery.setDiscoveredLink(child, parent);
        expect(statisticLinkDiscovery.metadata).toEqual({
            'child': [{
                'childkey': 5, 
                'discoveredTimestamp': Date.now(),
                'discoverOrder': 0 
        }]});
    });

    it('correctly aggregate metadata', () => {
        const parent2: ILink = { url: 'parent2',
            metadata: { }
        }
        statisticLinkDiscovery.setDiscoveredLink(child, parent);
        // Metadata upon second discovery from different parent can change
        child.metadata = {'childkey': 10, 'extrakey': 'test' };
        statisticLinkDiscovery.setDiscoveredLink(child, parent2);

        expect(statisticLinkDiscovery.edgeList).toEqual(
            new Set([JSON.stringify([ parent.url, child.url ]), JSON.stringify([ parent2.url, child.url ])])
        );
        
        expect(statisticLinkDiscovery.metadata).toEqual(
            {'child': 
                [{ 'childkey': 5, 'discoveredTimestamp': Date.now(),'discoverOrder': 0 },
                { 'childkey': 10, 'extrakey': 'test', 'discoveredTimestamp': Date.now(),'discoverOrder': 1}]
            });
    });

    it('emit event', () => {
        statisticLinkDiscovery.addListener(cb);
        statisticLinkDiscovery.emit(discoveredEventData);
        expect(cb).toHaveBeenCalledWith({             
                edge: ['parent', 'child'],
                metadataDiscoveredNode: [{}],
                metadataParentDiscoveredNode: [{}] 
        });
    });
  });

  afterEach(() => {
    jest.useRealTimers();
  });
});
