import type { ILink } from '@comunica/bus-rdf-resolve-hypermedia-links';
import type { BindingsStream, FragmentSelectorShape, IActionContext, IQueryBindingsOptions, IQuerySource, QuerySourceReference } from '@comunica/types';
import type { Quad } from '@rdfjs/types';
import type { AsyncIterator } from 'asynciterator';
import type { Operation, Ask, Update } from 'sparqlalgebrajs/lib/algebra';
import { StatisticLinkDereference } from '../lib/StatisticLinkDereference';
import { ActionContext, Bus } from '@comunica/core';
import { KeysStatisticsTracker, KeysTrackableStatistics } from '@comunica/context-entries';
import { StatisticsHolder } from '@comunica/actor-context-preprocess-set-defaults';
import { ActorContextPreprocessStatisticLinkDereference } from '../lib';

class mockQuerySource implements IQuerySource {
  referenceValue: QuerySourceReference;

  public constructor(referenceValue: QuerySourceReference) {
    this.referenceValue = referenceValue;
  }

  getSelectorShape: (context: IActionContext) => Promise<FragmentSelectorShape>;
  queryBindings: (operation: Operation, context: IActionContext, options?: IQueryBindingsOptions | undefined) => BindingsStream;
  queryQuads: (operation: Operation, context: IActionContext) => AsyncIterator<Quad>;
  queryBoolean: (operation: Ask, context: IActionContext) => Promise<boolean>;
  queryVoid: (operation: Update, context: IActionContext) => Promise<void>;
  toString: () => string;
}

describe('StatisticLinkDereference', () => {
  let bus: any;
  let statisticLinkDereference: StatisticLinkDereference;
  let mockLogFunction: ((message: string, data?: (() => any)) => void);


  beforeEach(() => {
    bus = new Bus({ name: 'bus' });
    jest.useFakeTimers();
    jest.setSystemTime(new Date('2021-01-01T00:00:00Z').getTime());
    mockLogFunction = jest.fn((message: string, data?: (() => any)) => { });
    statisticLinkDereference = new StatisticLinkDereference(mockLogFunction);
  });

  describe('An StatisticLinkDereference instance should', () => {
    let actor: ActorContextPreprocessStatisticLinkDereference;
    let link: ILink;
    let source: IQuerySource;
    let cb: (arg0: ILink) => void;

    beforeEach(() => {
      actor = new ActorContextPreprocessStatisticLinkDereference({ name: 'actor', bus });
      link = { url: 'url', metadata: { key: 'value' }};
      cb = jest.fn((arg0: ILink) => {});

      source = new mockQuerySource('url');
    });

    it('attach an event listener', () => {
      statisticLinkDereference.addListener(cb);
      expect(statisticLinkDereference.statisticEvents.listeners('data')).toEqual(
        [ cb ],
      );
    });

    it('update dereferenced links', () => {
      statisticLinkDereference.setDereferenced(link, source);
      expect(statisticLinkDereference.dereferenceOrder).toEqual([
        {
          url: 'url',
          metadata: {
            type: 'mockQuerySource',
            dereferencedTimestamp: new Date('2021-01-01T00:00:00Z').getTime(),
            dereferenceOrder: 0,
            key: 'value',
          },
        },
      ]);
    });

    it('update in correct order', () => {
        statisticLinkDereference.setDereferenced(link, source);
        const link2 = { url: 'url2', metadata: { key: 'value'}};
        statisticLinkDereference.setDereferenced(link2, source);

        expect(statisticLinkDereference.dereferenceOrder).toEqual([
            {
            url: 'url',
            metadata: {
                type: 'mockQuerySource',
                dereferencedTimestamp: new Date('2021-01-01T00:00:00Z').getTime(),
                dereferenceOrder: 0,
                key: 'value',
            }},
            {
            url: 'url2',
            metadata: {
                type: 'mockQuerySource',
                dereferencedTimestamp: new Date('2021-01-01T00:00:00Z').getTime(),
                dereferenceOrder: 1,
                key: 'value',
            },
            },
            ]);
    });

    it('emit event', () => {
        statisticLinkDereference.addListener(cb);
        statisticLinkDereference.emit(
        {             
            url: 'url',
            metadata: {
                type: 'mockQuerySource',
                dereferencedTimestamp: new Date('2021-01-01T00:00:00Z').getTime(),
                dereferenceOrder: 0,
                key: 'value',
            }
        },
    );
        expect(cb).toHaveBeenCalledWith(
            {             
                url: 'url',
                metadata: {
                    type: 'mockQuerySource',
                    dereferencedTimestamp: new Date('2021-01-01T00:00:00Z').getTime(),
                    dereferenceOrder: 0,
                    key: 'value',
                }
            },
        );
    });
    
    it('call log function', () => {
      statisticLinkDereference.setDereferenced(link, source);
      expect(mockLogFunction).toHaveBeenCalledWith('Dereference Event', expect.any(Function))
    });

    it('correctly log dereferenced link', async () => {
      const contextIn = new ActionContext({ [KeysStatisticsTracker.statistics.name]: new StatisticsHolder() });
      const { context: contextOut } = await actor.run({ context: contextIn });
      const spy = jest.spyOn(actor, <any> 'logInfo');

      const statisticsHolder: StatisticsHolder = contextOut.get(
        KeysStatisticsTracker.statistics)!;

      const statisticLinkDiscoveryFromActor: StatisticLinkDereference = statisticsHolder.get(
        KeysTrackableStatistics.dereferencedLinks)!;

      statisticLinkDiscoveryFromActor.setDereferenced(link, source);
      expect(spy).toHaveBeenCalledWith(contextIn, 'Dereference Event', expect.anything());

      // Test the anonymous function passed to variable 'data'
      expect((<Function> spy.mock.calls[0][2])()).toEqual({
        url: 'url',
        metadata: { 
          type: "mockQuerySource",
          key: 'value',
          dereferencedTimestamp: Date.now(),
          dereferenceOrder: 0
         }
      })
    });

  });

  afterEach(() => {
    jest.useRealTimers();
  });
});
