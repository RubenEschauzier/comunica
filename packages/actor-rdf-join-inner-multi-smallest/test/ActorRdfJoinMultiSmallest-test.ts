import { ActorRdfJoinNestedLoop } from '@comunica/actor-rdf-join-inner-nestedloop';
import type { IActionRdfJoin } from '@comunica/bus-rdf-join';
import { ActorRdfJoin } from '@comunica/bus-rdf-join';
import type { IActionRdfJoinEntriesSort, MediatorRdfJoinEntriesSort } from '@comunica/bus-rdf-join-entries-sort';
import type { IActionRdfJoinSelectivity, IActorRdfJoinSelectivityOutput } from '@comunica/bus-rdf-join-selectivity';
import { KeysInitQuery } from '@comunica/context-entries';
import type { Actor, IActorTest, Mediator } from '@comunica/core';
import { ActionContext, Bus } from '@comunica/core';
import type { IActionContext } from '@comunica/types';
import { BindingsFactory } from '@comunica/utils-bindings-factory';
import { MetadataValidationState } from '@comunica/utils-metadata';
import type * as RDF from '@rdfjs/types';
import { ArrayIterator } from 'asynciterator';
import { DataFactory } from 'rdf-data-factory';
import type { IActorRdfJoinMultiSmallestTestSideData } from '../lib/ActorRdfJoinMultiSmallest';
import { ActorRdfJoinMultiSmallest } from '../lib/ActorRdfJoinMultiSmallest';
import '@comunica/utils-jest';

const DF = new DataFactory();
const BF = new BindingsFactory(DF);

describe('ActorRdfJoinMultiSmallest', () => {
  let bus: any;
  let context: IActionContext;

  beforeEach(() => {
    bus = new Bus({ name: 'bus' });
    context = new ActionContext({ [KeysInitQuery.dataFactory.name]: DF });
  });

  describe('The ActorRdfJoinMultiSmallest module', () => {
    it('should be a function', () => {
      expect(ActorRdfJoinMultiSmallest).toBeInstanceOf(Function);
    });

    it('should be a ActorRdfJoinMultiSmallest constructor', () => {
      expect(new (<any> ActorRdfJoinMultiSmallest)({ name: 'actor', bus }))
        .toBeInstanceOf(ActorRdfJoinMultiSmallest);
      expect(new (<any> ActorRdfJoinMultiSmallest)({ name: 'actor', bus }))
        .toBeInstanceOf(ActorRdfJoin);
    });

    it('should not be able to create new ActorRdfJoinMultiSmallest objects without \'new\'', () => {
      expect(() => {
        (<any> ActorRdfJoinMultiSmallest)();
      }).toThrow(`Class constructor ActorRdfJoinMultiSmallest cannot be invoked without 'new'`);
    });
  });

  describe('An ActorRdfJoinMultiSmallest instance', () => {
    let mediatorJoinSelectivity: Mediator<
    Actor<IActionRdfJoinSelectivity, IActorTest, IActorRdfJoinSelectivityOutput>,
    IActionRdfJoinSelectivity,
IActorTest,
IActorRdfJoinSelectivityOutput
>;
    let mediatorJoinEntriesSort: MediatorRdfJoinEntriesSort;
    let mediatorJoin: any;
    let actor: ActorRdfJoinMultiSmallest;
    let action3: () => IActionRdfJoin;
    let action4: () => IActionRdfJoin;
    let invocationCounter: any;

    beforeEach(() => {
      mediatorJoinSelectivity = <any> {
        mediate: async() => ({ selectivity: 1 }),
      };
      mediatorJoinEntriesSort = <any> {
        async mediate(action: IActionRdfJoinEntriesSort) {
          const entries = [ ...action.entries ]
            .sort((left, right) => left.metadata.cardinality.value - right.metadata.cardinality.value);
          return { entries };
        },
      };
      invocationCounter = 0;
      mediatorJoin = {
        async mediate(a: any) {
          if (a.entries.length === 2) {
            a.entries[0].operation.called = invocationCounter;
            a.entries[1].operation.called = invocationCounter;
            invocationCounter++;
            return new ActorRdfJoinNestedLoop({ name: 'actor', bus, mediatorJoinSelectivity })
              .run(a, undefined!);
          }
          return actor.run(a, await getSideData(a));
        },
      };
      actor = new ActorRdfJoinMultiSmallest(
        { name: 'actor', bus, mediatorJoin, mediatorJoinSelectivity, mediatorJoinEntriesSort },
      );
      action3 = () => ({
        type: 'inner',
        entries: [
          {
            output: {
              bindingsStream: new ArrayIterator<RDF.Bindings>([
                BF.bindings([
                  [ DF.variable('a'), DF.literal('a1') ],
                  [ DF.variable('b'), DF.literal('b1') ],
                ]),
                BF.bindings([
                  [ DF.variable('a'), DF.literal('a2') ],
                  [ DF.variable('b'), DF.literal('b2') ],
                ]),
              ]),
              metadata: () => Promise.resolve(
                {
                  state: new MetadataValidationState(),
                  cardinality: { type: 'estimate', value: 4 },
                  pageSize: 100,
                  requestTime: 10,

                  variables: [
                    { variable: DF.variable('a'), canBeUndef: false },
                    { variable: DF.variable('b'), canBeUndef: false },
                  ],
                },
              ),
              type: 'bindings',
            },
            operation: <any> {},
          },
          {
            output: {
              bindingsStream: new ArrayIterator<RDF.Bindings>([
                BF.bindings([
                  [ DF.variable('a'), DF.literal('a1') ],
                  [ DF.variable('c'), DF.literal('c1') ],
                ]),
                BF.bindings([
                  [ DF.variable('a'), DF.literal('a2') ],
                  [ DF.variable('c'), DF.literal('c2') ],
                ]),
              ]),
              metadata: () => Promise.resolve(
                {
                  state: new MetadataValidationState(),
                  cardinality: { type: 'estimate', value: 5 },
                  pageSize: 100,
                  requestTime: 20,

                  variables: [
                    { variable: DF.variable('a'), canBeUndef: false },
                    { variable: DF.variable('c'), canBeUndef: false },
                  ],
                },
              ),
              type: 'bindings',
            },
            operation: <any> {},
          },
          {
            output: {
              bindingsStream: new ArrayIterator<RDF.Bindings>([
                BF.bindings([
                  [ DF.variable('a'), DF.literal('a1') ],
                  [ DF.variable('b'), DF.literal('b1') ],
                ]),
                BF.bindings([
                  [ DF.variable('a'), DF.literal('a2') ],
                  [ DF.variable('b'), DF.literal('b2') ],
                ]),
              ]),
              metadata: () => Promise.resolve(
                {
                  state: new MetadataValidationState(),
                  cardinality: { type: 'estimate', value: 2 },
                  pageSize: 100,
                  requestTime: 30,

                  variables: [
                    { variable: DF.variable('a'), canBeUndef: false },
                    { variable: DF.variable('b'), canBeUndef: false },
                  ],
                },
              ),
              type: 'bindings',
            },
            operation: <any> {},
          },
        ],
        context,
      });
      action4 = () => ({
        type: 'inner',
        entries: [
          {
            output: {
              bindingsStream: new ArrayIterator<RDF.Bindings>([
                BF.bindings([
                  [ DF.variable('a'), DF.literal('a1') ],
                  [ DF.variable('b'), DF.literal('b1') ],
                ]),
                BF.bindings([
                  [ DF.variable('a'), DF.literal('a2') ],
                  [ DF.variable('b'), DF.literal('b2') ],
                ]),
              ]),
              metadata: () => Promise.resolve(
                {
                  state: new MetadataValidationState(),
                  cardinality: { type: 'estimate', value: 4 },
                  pageSize: 100,
                  requestTime: 10,

                  variables: [
                    { variable: DF.variable('a'), canBeUndef: false },
                    { variable: DF.variable('b'), canBeUndef: false },
                  ],
                },
              ),
              type: 'bindings',
            },
            operation: <any> {},
          },
          {
            output: {
              bindingsStream: new ArrayIterator<RDF.Bindings>([
                BF.bindings([
                  [ DF.variable('a'), DF.literal('a1') ],
                  [ DF.variable('c'), DF.literal('c1') ],
                ]),
                BF.bindings([
                  [ DF.variable('a'), DF.literal('a2') ],
                  [ DF.variable('c'), DF.literal('c2') ],
                ]),
              ]),
              metadata: () => Promise.resolve(
                {
                  state: new MetadataValidationState(),
                  cardinality: { type: 'estimate', value: 5 },
                  pageSize: 100,
                  requestTime: 20,

                  variables: [
                    { variable: DF.variable('a'), canBeUndef: false },
                    { variable: DF.variable('c'), canBeUndef: false },
                  ],
                },
              ),
              type: 'bindings',
            },
            operation: <any> {},
          },
          {
            output: {
              bindingsStream: new ArrayIterator<RDF.Bindings>([
                BF.bindings([
                  [ DF.variable('a'), DF.literal('a1') ],
                  [ DF.variable('b'), DF.literal('b1') ],
                ]),
                BF.bindings([
                  [ DF.variable('a'), DF.literal('a2') ],
                  [ DF.variable('b'), DF.literal('b2') ],
                ]),
              ]),
              metadata: () => Promise.resolve(
                {
                  state: new MetadataValidationState(),
                  cardinality: { type: 'estimate', value: 2 },
                  pageSize: 100,
                  requestTime: 30,

                  variables: [
                    { variable: DF.variable('a'), canBeUndef: false },
                    { variable: DF.variable('b'), canBeUndef: false },
                  ],
                },
              ),
              type: 'bindings',
            },
            operation: <any> {},
          },
          {
            output: {
              bindingsStream: new ArrayIterator<RDF.Bindings>([
                BF.bindings([
                  [ DF.variable('a'), DF.literal('a1') ],
                  [ DF.variable('d'), DF.literal('d1') ],
                ]),
                BF.bindings([
                  [ DF.variable('a'), DF.literal('a2') ],
                  [ DF.variable('d'), DF.literal('d2') ],
                ]),
              ]),
              metadata: () => Promise.resolve(
                {
                  state: new MetadataValidationState(),
                  cardinality: { type: 'estimate', value: 2 },
                  pageSize: 100,
                  requestTime: 40,

                  variables: [
                    { variable: DF.variable('a'), canBeUndef: false },
                    { variable: DF.variable('d'), canBeUndef: false },
                  ],
                },
              ),
              type: 'bindings',
            },
            operation: <any> {},
          },
        ],
        context,
      });
    });

    async function getSideData(action: IActionRdfJoin): Promise<IActorRdfJoinMultiSmallestTestSideData> {
      return (await actor.test(action)).getSideData();
    }

    it('should not test on 0 streams', async() => {
      await expect(actor.test({ type: 'inner', entries: [], context })).resolves
        .toFailTest('actor requires at least two join entries.');
    });

    it('should not test on 1 stream', async() => {
      await expect(actor.test({ type: 'inner', entries: [ <any> null ], context })).resolves
        .toFailTest('actor requires at least two join entries.');
    });

    it('should not test on 2 streams', async() => {
      await expect(actor.test({ type: 'inner', entries: [ <any> null, <any> null ], context })).resolves
        .toFailTest('actor requires 3 join entries at least. The input contained 2.');
    });

    it('should test on 3 streams', async() => {
      const action = action3();
      await expect(actor.test(action)).resolves.toPassTest({
        iterations: 40,
        persistedItems: 0,
        blockingItems: 0,
        requestTime: 2,
      });
      for (const entry of action.entries) {
        entry.output.bindingsStream.destroy();
      }
    });

    it('should test on 4 streams', async() => {
      const action = action4();
      await expect(actor.test(action)).resolves.toPassTest({
        iterations: 80,
        persistedItems: 0,
        blockingItems: 0,
        requestTime: 2.8,
      });
      for (const entry of action.entries) {
        entry.output.bindingsStream.destroy();
      }
    });

    it('should run on 3 streams', async() => {
      const action = action3();
      const output = await actor.run(action, await getSideData(action));
      expect(output.type).toBe('bindings');
      await expect((<any> output).metadata()).resolves.toEqual({
        state: expect.any(MetadataValidationState),
        cardinality: { type: 'estimate', value: 40 },

        variables: [
          { variable: DF.variable('a'), canBeUndef: false },
          { variable: DF.variable('c'), canBeUndef: false },
          { variable: DF.variable('b'), canBeUndef: false },
        ],
      });
      await expect(output.bindingsStream).toEqualBindingsStream([
        BF.bindings([
          [ DF.variable('a'), DF.literal('a1') ],
          [ DF.variable('b'), DF.literal('b1') ],
          [ DF.variable('c'), DF.literal('c1') ],
        ]),
        BF.bindings([
          [ DF.variable('a'), DF.literal('a2') ],
          [ DF.variable('b'), DF.literal('b2') ],
          [ DF.variable('c'), DF.literal('c2') ],
        ]),
      ]);

      // Check join order
      expect((<any> action.entries[0]).operation.called).toBe(0);
      expect((<any> action.entries[1]).operation.called).toBe(1);
      expect((<any> action.entries[2]).operation.called).toBe(0);
    });

    it('should run on 4 streams', async() => {
      const action = action4();
      const output = await actor.run(action, await getSideData(action));
      expect(output.type).toBe('bindings');
      await expect((<any> output).metadata()).resolves.toEqual({
        state: expect.any(MetadataValidationState),
        cardinality: { type: 'estimate', value: 80 },

        variables: [
          { variable: DF.variable('a'), canBeUndef: false },
          { variable: DF.variable('c'), canBeUndef: false },
          { variable: DF.variable('b'), canBeUndef: false },
          { variable: DF.variable('d'), canBeUndef: false },
        ],
      });
      await expect(output.bindingsStream).toEqualBindingsStream([
        BF.bindings([
          [ DF.variable('a'), DF.literal('a1') ],
          [ DF.variable('b'), DF.literal('b1') ],
          [ DF.variable('c'), DF.literal('c1') ],
          [ DF.variable('d'), DF.literal('d1') ],
        ]),
        BF.bindings([
          [ DF.variable('a'), DF.literal('a2') ],
          [ DF.variable('b'), DF.literal('b2') ],
          [ DF.variable('c'), DF.literal('c2') ],
          [ DF.variable('d'), DF.literal('d2') ],
        ]),
      ]);

      // Check join order
      expect((<any> action.entries[0]).operation.called).toBe(1);
      expect((<any> action.entries[1]).operation.called).toBe(2);
      expect((<any> action.entries[2]).operation.called).toBe(0);
      expect((<any> action.entries[3]).operation.called).toBe(0);
    });
  });
});
