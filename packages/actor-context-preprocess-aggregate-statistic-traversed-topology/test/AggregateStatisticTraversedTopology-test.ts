import { StatisticsHolder } from '@comunica/actor-context-preprocess-set-defaults';
import { StatisticLinkDereference } from '@comunica/actor-context-preprocess-statistic-link-dereference';
import { StatisticLinkDiscovery } from '@comunica/actor-context-preprocess-statistic-link-discovery';
import type { ILink } from '@comunica/bus-rdf-resolve-hypermedia-links';
import { KeysStatisticsTracker, KeysTrackableStatistics } from '@comunica/context-entries';
import { ActionContext, Bus } from '@comunica/core';
import type {
  ITopologyEventNotification,
  BindingsStream,
  FragmentSelectorShape,
  IActionContext,
  IQueryBindingsOptions,
  IQuerySource,
  QuerySourceReference,
} from '@comunica/types';
import type { Quad } from '@rdfjs/types';
import type { AsyncIterator } from 'asynciterator';
import type { Operation, Ask, Update } from 'sparqlalgebrajs/lib/algebra';
import { ActorContextPreprocessAggregateStatisticTraversedTopology } from '../lib';
import { AggregateStatisticTraversedTopology } from '../lib/AggregateStatisticTraversedTopology';

class MockQuerySource implements IQuerySource {
  public referenceValue: QuerySourceReference;

  public constructor(referenceValue: QuerySourceReference) {
    this.referenceValue = referenceValue;
  }

  public getSelectorShape: (context: IActionContext) => Promise<FragmentSelectorShape>;
  public queryBindings: (operation: Operation, context: IActionContext, options?: IQueryBindingsOptions | undefined)
  => BindingsStream;

  public queryQuads: (operation: Operation, context: IActionContext) => AsyncIterator<Quad>;
  public queryBoolean: (operation: Ask, context: IActionContext) => Promise<boolean>;
  public queryVoid: (operation: Update, context: IActionContext) => Promise<void>;
  public toString: () => string;
}

describe('AggregateStatisticTraversedTopology', () => {
  let bus: any;

  let parent: ILink;
  let child: ILink;
  let source: IQuerySource;

  let statisticLinkDiscovery: StatisticLinkDiscovery;
  let statisticLinkDereference: StatisticLinkDereference;
  let aggregateStatisticTraversedTopology: AggregateStatisticTraversedTopology;

  beforeEach(() => {
    bus = new Bus({ name: 'bus' });
    jest.useFakeTimers();
    jest.setSystemTime(new Date('2021-01-01T00:00:00Z').getTime());
    source = new MockQuerySource('url');
  });

  describe('An AggregateStatisticTraversedTopology instance should', () => {
    let actor: ActorContextPreprocessAggregateStatisticTraversedTopology;
    let cb: (arg0: ITopologyEventNotification) => void;
    let mockLogFunction: ((message: string, data?: (() => any)) => void);

    beforeEach(() => {
      actor = new ActorContextPreprocessAggregateStatisticTraversedTopology({ name: 'actor', bus });

      cb = jest.fn((arg0: ITopologyEventNotification) => {});
      mockLogFunction = jest.fn((message: string, data?: (() => any)) => {});
      statisticLinkDiscovery = new StatisticLinkDiscovery(mockLogFunction);
      statisticLinkDereference = new StatisticLinkDereference(mockLogFunction);
      aggregateStatisticTraversedTopology = new AggregateStatisticTraversedTopology(
        statisticLinkDiscovery,
        statisticLinkDereference,
        mockLogFunction,
      );

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
    });

    it('return topology representation on get', () => {
      statisticLinkDiscovery.setDiscoveredLink(child, parent);
      expect(aggregateStatisticTraversedTopology.getTrackedStatistics()).toEqual({
        edgeList: new Set([ JSON.stringify([ 'parent', 'child' ]) ]),
        metadata: { child: {
          childkey: [ 5 ],
          discoveredTimestamp: [ Date.now() ],
          discoverOrder: [ 0 ],
        }},
      });
    });

    it('receive discovery events', () => {
      statisticLinkDiscovery.setDiscoveredLink(child, parent);
      expect(aggregateStatisticTraversedTopology.edgeList).toEqual(
        new Set([ JSON.stringify([ 'parent', 'child' ]) ]),
      );
      expect(aggregateStatisticTraversedTopology.metadata).toEqual(
        { child: {
          childkey: [ 5 ],
          discoveredTimestamp: [ Date.now() ],
          discoverOrder: [ 0 ],
        }},
      );
    });

    it('correctly aggregate metadata for multiple discover events', () => {
      const parent2 = {
        url: 'parent2',
      };
      const child2 = {
        url: 'child',
        metadata: {
          childkey: 10,
          childkey2: 'test',
        },
      };

      statisticLinkDiscovery.setDiscoveredLink(child, parent);
      statisticLinkDiscovery.setDiscoveredLink(child2, parent2);
      expect(aggregateStatisticTraversedTopology.metadata).toEqual(
        { child: {
          childkey: [ 5, 10 ],
          childkey2: [ 'test' ],
          discoveredTimestamp: [ Date.now(), Date.now() ],
          discoverOrder: [ 0, 1 ],
        }},
      );
    });

    it('emit event on new discover statistic event', () => {
      aggregateStatisticTraversedTopology.addListener(cb);
      statisticLinkDiscovery.setDiscoveredLink(child, parent);
      expect(cb).toHaveBeenCalledWith(
        {
          type: 'discover',
        },
      );
    });

    it('receive dereference events', () => {
      statisticLinkDiscovery.setDiscoveredLink(child, parent);
      statisticLinkDereference.setDereferenced(child, source);

      expect(aggregateStatisticTraversedTopology.edgeList).toEqual(
        new Set([ JSON.stringify([ 'parent', 'child' ]) ]),
      );
      expect(aggregateStatisticTraversedTopology.metadata).toEqual(
        { child: {
          childkey: 5,
          discoveredTimestamp: [ Date.now() ],
          dereferencedTimestamp: Date.now(),
          discoverOrder: [ 0 ],
          dereferenceOrder: 0,
          type: 'MockQuerySource',
        }},
      );
    });

    it('gracefully deal with undiscovered dereference events (seeds URLs)', () => {
      statisticLinkDereference.setDereferenced(child, source);
      expect(aggregateStatisticTraversedTopology.edgeList).toEqual(
        new Set(),
      );
      expect(aggregateStatisticTraversedTopology.metadata).toEqual(
        { child: {
          childkey: 5,
          dereferencedTimestamp: Date.now(),
          dereferenceOrder: 0,
          type: 'MockQuerySource',
        }},
      );
    });

    it('overwrite metadata in event that two dereference events happen', () => {
      // This shouldn't happen, but this test case just tests if the functions fails gracefully
      statisticLinkDiscovery.setDiscoveredLink(child, parent);
      statisticLinkDereference.setDereferenced(child, source);
      statisticLinkDereference.setDereferenced(child, source);

      expect(aggregateStatisticTraversedTopology.edgeList).toEqual(
        new Set([ JSON.stringify([ 'parent', 'child' ]) ]),
      );
      expect(aggregateStatisticTraversedTopology.metadata).toEqual(
        { child: {
          childkey: 5,
          discoveredTimestamp: [ Date.now() ],
          dereferencedTimestamp: Date.now(),
          discoverOrder: [ 0 ],
          dereferenceOrder: 1,
          type: 'MockQuerySource',
        }},
      );
    });

    it('emit event on new dereference statistic event', () => {
      aggregateStatisticTraversedTopology.addListener(cb);
      statisticLinkDereference.setDereferenced(child, source);
      expect(cb).toHaveBeenCalledWith(
        {
          type: 'dereference',
        },
      );
    });

    it('attach an event listener', () => {
      aggregateStatisticTraversedTopology.addListener(cb);
      expect(aggregateStatisticTraversedTopology.statisticEvents.listeners('data')).toEqual(
        [ cb ],
      );
    });

    it('emit topology change notification event', () => {
      const topologyData: ITopologyEventNotification = {
        type: 'discover',
      };

      aggregateStatisticTraversedTopology.addListener(cb);
      aggregateStatisticTraversedTopology.emit(topologyData);
      expect(cb).toHaveBeenCalledWith(
        {
          type: 'discover',
        },
      );
    });

    it('call log function', () => {
      statisticLinkDereference.setDereferenced(child, source);
      expect(mockLogFunction).toHaveBeenCalledWith('Dereference Event', expect.any(Function));
    });

    it('correctly log topology update on discover event', async() => {
      const statisticsHolder = new StatisticsHolder();
      statisticsHolder.set(KeysTrackableStatistics.discoveredLinks, statisticLinkDiscovery);
      statisticsHolder.set(KeysTrackableStatistics.dereferencedLinks, statisticLinkDereference);

      const contextIn = new ActionContext({ [KeysStatisticsTracker.statistics.name]: statisticsHolder });
      const { context: contextOut } = await actor.run({ context: contextIn });
      const spy = jest.spyOn(actor, <any> 'logInfo');

      statisticLinkDiscovery.setDiscoveredLink(child, parent);
      expect(spy).toHaveBeenCalledWith(contextIn, 'Topology Discover Event', expect.anything());

      // Test the anonymous function passed to variable 'data'
      expect((<Function> spy.mock.calls[0][2])()).toEqual({
        edge: [ 'parent', 'child' ],
        metadataChild: {
          childkey: [ 5 ],
          discoveredTimestamp: [ Date.now() ],
          discoverOrder: [ 0 ],
        },
      });
    });
    it('correctly log topology update on dereference event', async() => {
      const statisticsHolder = new StatisticsHolder();
      statisticsHolder.set(KeysTrackableStatistics.discoveredLinks, statisticLinkDiscovery);
      statisticsHolder.set(KeysTrackableStatistics.dereferencedLinks, statisticLinkDereference);

      const contextIn = new ActionContext({ [KeysStatisticsTracker.statistics.name]: statisticsHolder });
      await actor.run({ context: contextIn });
      const spy = jest.spyOn(actor, <any> 'logInfo');

      statisticLinkDereference.setDereferenced(child, source);
      expect(spy).toHaveBeenCalledWith(contextIn, 'Topology Dereference Event', expect.anything());

      // Test the anonymous function passed to variable 'data'
      expect((<Function> spy.mock.calls[0][2])()).toEqual({
        dereferenced: 'child',
        metadata: {
          type: 'MockQuerySource',
          childkey: 5,
          dereferencedTimestamp: Date.now(),
          dereferenceOrder: 0,
        },
      });
    });
    it('correctly log both dereference and discover events', async() => {
      const statisticsHolder = new StatisticsHolder();
      statisticsHolder.set(KeysTrackableStatistics.discoveredLinks, statisticLinkDiscovery);
      statisticsHolder.set(KeysTrackableStatistics.dereferencedLinks, statisticLinkDereference);

      const contextIn = new ActionContext({ [KeysStatisticsTracker.statistics.name]: statisticsHolder });
      await actor.run({ context: contextIn });
      const spy = jest.spyOn(actor, <any> 'logInfo');

      statisticLinkDiscovery.setDiscoveredLink(child, parent);
      expect(spy).toHaveBeenCalledWith(contextIn, 'Topology Discover Event', expect.anything());

      statisticLinkDereference.setDereferenced(child, source);
      expect(spy).toHaveBeenCalledWith(contextIn, 'Topology Dereference Event', expect.anything());

      // Test the anonymous function passed to variable 'data' in discover event
      expect((<Function> spy.mock.calls[0][2])()).toEqual({
        edge: [ 'parent', 'child' ],
        metadataChild: {
          childkey: [ 5 ],
          discoveredTimestamp: [ Date.now() ],
          discoverOrder: [ 0 ],
        },
      });

      // Test the anonymous function passed to variable 'data' in dereference event
      expect((<Function> spy.mock.calls[1][2])()).toEqual({
        dereferenced: 'child',
        metadata: {
          type: 'MockQuerySource',
          childkey: 5,
          dereferencedTimestamp: Date.now(),
          dereferenceOrder: 0,
          discoveredTimestamp: [ Date.now() ],
          discoverOrder: [ 0 ],
        },
      });
    });
  });

  afterEach(() => {
    jest.useRealTimers();
  });
});
