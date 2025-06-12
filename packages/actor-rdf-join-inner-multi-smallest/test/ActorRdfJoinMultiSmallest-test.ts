import { ActorRdfJoinNestedLoop } from '@comunica/actor-rdf-join-inner-nestedloop';
import { BindingsFactory } from '@comunica/bindings-factory';
import type { IActionRdfJoin } from '@comunica/bus-rdf-join';
import { ActorRdfJoin } from '@comunica/bus-rdf-join';
import type { IActionRdfJoinEntriesSort, MediatorRdfJoinEntriesSort } from '@comunica/bus-rdf-join-entries-sort';
import type { IActionRdfJoinSelectivity, IActorRdfJoinSelectivityOutput } from '@comunica/bus-rdf-join-selectivity';
import type { Actor, IActorTest, Mediator } from '@comunica/core';
import { ActionContext, Bus } from '@comunica/core';
import { MetadataValidationState } from '@comunica/metadata';
import type { IActionContext } from '@comunica/types';
import { ArrayIterator } from 'asynciterator';
import { DataFactory } from 'rdf-data-factory';
import { ActorRdfJoinMultiSmallest } from '../lib/ActorRdfJoinMultiSmallest';
import '@comunica/jest';

const DF = new DataFactory();
const BF = new BindingsFactory();

describe('ActorRdfJoinMultiSmallest', () => {
  let bus: any;
  let context: IActionContext;

  beforeEach(() => {
    bus = new Bus({ name: 'bus' });
    context = new ActionContext();
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
      expect(() => { (<any> ActorRdfJoinMultiSmallest)(); }).toThrow();
    });
  });

  describe('An ActorRdfJoinMultiSmallest instance', () => {
    let mediatorJoinSelectivity: Mediator<
    Actor<IActionRdfJoinSelectivity, IActorTest, IActorRdfJoinSelectivityOutput>,
    IActionRdfJoinSelectivity, IActorTest, IActorRdfJoinSelectivityOutput>;
    let mediatorJoinEntriesSort: MediatorRdfJoinEntriesSort;
    let mediatorJoin: any;
    let minimizeCardinalityProduct: boolean;
    let actor: ActorRdfJoinMultiSmallest;
    let action4: () => IActionRdfJoin;
    let invocationCounter: any;

    beforeEach(() => {
      minimizeCardinalityProduct = true;
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
        mediate(a: any) {
          if (a.entries.length === 2) {
            a.entries[0].operation.called = invocationCounter;
            a.entries[1].operation.called = invocationCounter;
            invocationCounter++;
            return new ActorRdfJoinNestedLoop({ name: 'actor', bus, mediatorJoinSelectivity })
              .run(a);
          }
          return actor.run(a);
        },
      };
      actor = new ActorRdfJoinMultiSmallest(
        { name: 'actor', bus, mediatorJoin, mediatorJoinSelectivity, mediatorJoinEntriesSort , minimizeCardinalityProduct},
      );
      action4 = () => ({
        type: 'inner',
        entries: [
          {
            output: {
              bindingsStream: new ArrayIterator([
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
                  requestTime: 10,
                  canContainUndefs: false,
                  variables: [ DF.variable('a'), DF.variable('b') ],
                },
              ),
              type: 'bindings',
            },
            operation: <any> {},
          },
          {
            output: {
              bindingsStream: new ArrayIterator([
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
                  canContainUndefs: false,
                  variables: [ DF.variable('a'), DF.variable('c') ],
                },
              ),
              type: 'bindings',
            },
            operation: <any> {},
          },
          {
            output: {
              bindingsStream: new ArrayIterator([
                BF.bindings([
                  [ DF.variable('b'), DF.literal('b1') ],
                  [ DF.variable('d'), DF.literal('d1') ],
                ]),
                BF.bindings([
                  [ DF.variable('b'), DF.literal('b2') ],
                  [ DF.variable('d'), DF.literal('d2') ],
                ]),
              ]),
              metadata: () => Promise.resolve(
                {
                  state: new MetadataValidationState(),
                  cardinality: { type: 'estimate', value: 20 },
                  pageSize: 100,
                  requestTime: 30,
                  canContainUndefs: false,
                  variables: [ DF.variable('b'), DF.variable('d') ],
                },
              ),
              type: 'bindings',
            },
            operation: <any> {},
          },
          {
            output: {
              bindingsStream: new ArrayIterator([
                BF.bindings([
                  [ DF.variable('d'), DF.literal('d1') ],
                  [ DF.variable('f'), DF.literal('f1') ],
                ]),
                BF.bindings([
                  [ DF.variable('d'), DF.literal('d2') ],
                  [ DF.variable('f'), DF.literal('f2') ],
                ]),
              ]),
              metadata: () => Promise.resolve(
                {
                  state: new MetadataValidationState(),
                  cardinality: { type: 'estimate', value: 1 },
                  pageSize: 100,
                  requestTime: 30,
                  canContainUndefs: false,
                  variables: [ DF.variable('d'), DF.variable('f') ],
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

    it('should test on 4 streams smallest cardinality product', async() => {
      const action = action4();
      await expect(actor.test(action)).resolves.toEqual({
        iterations: 200,
        persistedItems: 0,
        blockingItems: 0,
        requestTime: 7.5,
      });
      action.entries.forEach(entry => entry.output.bindingsStream.destroy());
    });

    it('should run on 4 streams smallest cardinality product', async() => {
      const action = action4();
      const output = await actor.run(action);
      expect(output.type).toEqual('bindings');
      expect(await (<any> output).metadata()).toEqual({
        state: expect.any(MetadataValidationState),
        cardinality: { type: 'estimate', value: 80 },
        canContainUndefs: false,
        variables: [ DF.variable('a'), DF.variable('b'), DF.variable('c'), DF.variable('d'), DF.variable('f')],
      });
      await expect(output.bindingsStream).toEqualBindingsStream([
        BF.bindings([
          [ DF.variable('a'), DF.literal('a1') ],
          [ DF.variable('b'), DF.literal('b1') ],
          [ DF.variable('c'), DF.literal('c1') ],
          [ DF.variable('d'), DF.literal('d1') ],
          [ DF.variable('f'), DF.literal('f1') ],

        ]),
        BF.bindings([
          [ DF.variable('a'), DF.literal('a2') ],
          [ DF.variable('b'), DF.literal('b2') ],
          [ DF.variable('c'), DF.literal('c2') ],
          [ DF.variable('d'), DF.literal('d2') ],
          [ DF.variable('f'), DF.literal('f2') ],
        ]),
      ]);

      // Check join order
      expect((<any> action.entries[0]).operation.called).toBe(0);
      expect((<any> action.entries[1]).operation.called).toBe(0);
      expect((<any> action.entries[2]).operation.called).toBe(0);
      expect((<any> action.entries[3]).operation.called).toBe(0);
      expect((<any> action.entries[4]).operation.called).toBe(0);

    });

  });

  describe('An ActorRdfJoinMultiSmallest instance', () => {
    let mediatorJoinSelectivity: Mediator<
    Actor<IActionRdfJoinSelectivity, IActorTest, IActorRdfJoinSelectivityOutput>,
    IActionRdfJoinSelectivity, IActorTest, IActorRdfJoinSelectivityOutput>;
    let mediatorJoinEntriesSort: MediatorRdfJoinEntriesSort;
    let mediatorJoin: any;
    let minimizeCardinalityProduct: boolean;
    let actor: ActorRdfJoinMultiSmallest;
    let action3: () => IActionRdfJoin;
    let action4: () => IActionRdfJoin;
    let invocationCounter: any;

    beforeEach(() => {
      minimizeCardinalityProduct = false;
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
        mediate(a: any) {
          if (a.entries.length === 2) {
            a.entries[0].operation.called = invocationCounter;
            a.entries[1].operation.called = invocationCounter;
            invocationCounter++;
            return new ActorRdfJoinNestedLoop({ name: 'actor', bus, mediatorJoinSelectivity })
              .run(a);
          }
          return actor.run(a);
        },
      };
      actor = new ActorRdfJoinMultiSmallest(
        { name: 'actor', bus, mediatorJoin, mediatorJoinSelectivity, mediatorJoinEntriesSort , minimizeCardinalityProduct},
      );
      action3 = () => ({
        type: 'inner',
        entries: [
          {
            output: {
              bindingsStream: new ArrayIterator([
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
                  canContainUndefs: false,
                  variables: [ DF.variable('a'), DF.variable('b') ],
                },
              ),
              type: 'bindings',
            },
            operation: <any> {},
          },
          {
            output: {
              bindingsStream: new ArrayIterator([
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
                  canContainUndefs: false,
                  variables: [ DF.variable('a'), DF.variable('c') ],
                },
              ),
              type: 'bindings',
            },
            operation: <any> {},
          },
          {
            output: {
              bindingsStream: new ArrayIterator([
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
                  canContainUndefs: false,
                  variables: [ DF.variable('a'), DF.variable('b') ],
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
              bindingsStream: new ArrayIterator([
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
                  canContainUndefs: false,
                  variables: [ DF.variable('a'), DF.variable('b') ],
                },
              ),
              type: 'bindings',
            },
            operation: <any> {},
          },
          {
            output: {
              bindingsStream: new ArrayIterator([
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
                  canContainUndefs: false,
                  variables: [ DF.variable('a'), DF.variable('c') ],
                },
              ),
              type: 'bindings',
            },
            operation: <any> {},
          },
          {
            output: {
              bindingsStream: new ArrayIterator([
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
                  canContainUndefs: false,
                  variables: [ DF.variable('a'), DF.variable('b') ],
                },
              ),
              type: 'bindings',
            },
            operation: <any> {},
          },
          {
            output: {
              bindingsStream: new ArrayIterator([
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
                  canContainUndefs: false,
                  variables: [ DF.variable('a'), DF.variable('d') ],
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

    it('should not test on 0 streams', () => {
      return expect(actor.test({ type: 'inner', entries: [], context })).rejects
        .toThrow(new Error('actor requires at least two join entries.'));
    });

    it('should not test on 1 stream', () => {
      return expect(actor.test({ type: 'inner', entries: [ <any> null ], context })).rejects
        .toThrow(new Error('actor requires at least two join entries.'));
    });

    it('should not test on 2 streams', () => {
      return expect(actor.test({ type: 'inner', entries: [ <any> null, <any> null ], context })).rejects
        .toThrow(new Error('actor requires 3 join entries at least. The input contained 2.'));
    });

    it('should test on 3 streams', async() => {
      const action = action3();
      await expect(actor.test(action)).resolves.toEqual({
        iterations: 40,
        persistedItems: 0,
        blockingItems: 0,
        requestTime: 2,
      });
      action.entries.forEach(entry => entry.output.bindingsStream.destroy());
    });

    it('should test on 4 streams', async() => {
      const action = action4();
      await expect(actor.test(action)).resolves.toEqual({
        iterations: 80,
        persistedItems: 0,
        blockingItems: 0,
        requestTime: 2.8,
      });
      action.entries.forEach(entry => entry.output.bindingsStream.destroy());
    });

    it('should run on 3 streams', async() => {
      const action = action3();
      const output = await actor.run(action);
      expect(output.type).toEqual('bindings');
      expect(await (<any> output).metadata()).toEqual({
        state: expect.any(MetadataValidationState),
        cardinality: { type: 'estimate', value: 40 },
        canContainUndefs: false,
        variables: [ DF.variable('a'), DF.variable('c'), DF.variable('b') ],
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
      const output = await actor.run(action);
      expect(output.type).toEqual('bindings');
      expect(await (<any> output).metadata()).toEqual({
        state: expect.any(MetadataValidationState),
        cardinality: { type: 'estimate', value: 80 },
        canContainUndefs: false,
        variables: [ DF.variable('a'), DF.variable('c'), DF.variable('b'), DF.variable('d') ],
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

  // describe('An ActorRdfJoinMultiSmallest instance', () => {
  //   let mediatorJoinSelectivity: Mediator<
  //   Actor<IActionRdfJoinSelectivity, IActorTest, IActorRdfJoinSelectivityOutput>,
  //   IActionRdfJoinSelectivity, IActorTest, IActorRdfJoinSelectivityOutput>;
  //   let mediatorJoinEntriesSort: MediatorRdfJoinEntriesSort;
  //   let mediatorJoin: any;
  //   let minimizeCardinalityProduct: boolean;
  //   let actor: ActorRdfJoinMultiSmallest;
  //   let action23: () => IActionRdfJoin;
  //   let invocationCounter: any;

  //   beforeEach(() => {
  //     minimizeCardinalityProduct = true;
  //     mediatorJoinSelectivity = <any> {
  //       mediate: async() => ({ selectivity: 1 }),
  //     };
  //     mediatorJoinEntriesSort = <any> {
  //       async mediate(action: IActionRdfJoinEntriesSort) {
  //         const entries = [ ...action.entries ]
  //           .sort((left, right) => left.metadata.cardinality.value - right.metadata.cardinality.value);
  //         return { entries };
  //       },
  //     };
  //     invocationCounter = 0;
  //     mediatorJoin = {
  //       mediate(a: any) {
  //         if (a.entries.length === 2) {
  //           a.entries[0].operation.called = invocationCounter;
  //           a.entries[1].operation.called = invocationCounter;
  //           invocationCounter++;
  //           return new ActorRdfJoinNestedLoop({ name: 'actor', bus, mediatorJoinSelectivity })
  //             .run(a);
  //         }
  //         return actor.run(a);
  //       },
  //     };
  //     actor = new ActorRdfJoinMultiSmallest(
  //       { name: 'actor', bus, mediatorJoin, mediatorJoinSelectivity, mediatorJoinEntriesSort , minimizeCardinalityProduct},
  //     );
  //     action23 = () => ({
  //       type: 'inner',
  //       entries: [
  //         {
  //           output: {
  //             bindingsStream: new ArrayIterator([
  //               BF.bindings([
  //                 [ DF.variable('a'), DF.literal('a1') ],
  //                 [ DF.variable('b'), DF.literal('b1') ],
  //               ]),
  //               BF.bindings([
  //                 [ DF.variable('a'), DF.literal('a2') ],
  //                 [ DF.variable('b'), DF.literal('b2') ],
  //               ]),
  //             ]),
  //             metadata: () => Promise.resolve(
  //               {
  //                 state: new MetadataValidationState(),
  //                 cardinality: { type: 'estimate', value: 4 },
  //                 pageSize: 100,
  //                 requestTime: 10,
  //                 canContainUndefs: false,
  //                 variables: [ DF.variable('a'), DF.variable('b') ],
  //               },
  //             ),
  //             type: 'bindings',
  //           },
  //           operation: <any> {},
  //         },
  //         {
  //           output: {
  //             bindingsStream: new ArrayIterator([
  //               BF.bindings([
  //                 [ DF.variable('a'), DF.literal('a1') ],
  //                 [ DF.variable('c'), DF.literal('c1') ],
  //               ]),
  //               BF.bindings([
  //                 [ DF.variable('a'), DF.literal('a2') ],
  //                 [ DF.variable('c'), DF.literal('c2') ],
  //               ]),
  //             ]),
  //             metadata: () => Promise.resolve(
  //               {
  //                 state: new MetadataValidationState(),
  //                 cardinality: { type: 'estimate', value: 5 },
  //                 pageSize: 100,
  //                 requestTime: 20,
  //                 canContainUndefs: false,
  //                 variables: [ DF.variable('a'), DF.variable('c') ],
  //               },
  //             ),
  //             type: 'bindings',
  //           },
  //           operation: <any> {},
  //         },
  //         {
  //           output: {
  //             bindingsStream: new ArrayIterator([
  //               BF.bindings([
  //                 [ DF.variable('a'), DF.literal('a1') ],
  //                 [ DF.variable('b'), DF.literal('b1') ],
  //               ]),
  //               BF.bindings([
  //                 [ DF.variable('a'), DF.literal('a2') ],
  //                 [ DF.variable('b'), DF.literal('b2') ],
  //               ]),
  //             ]),
  //             metadata: () => Promise.resolve(
  //               {
  //                 state: new MetadataValidationState(),
  //                 cardinality: { type: 'estimate', value: 2 },
  //                 pageSize: 100,
  //                 requestTime: 30,
  //                 canContainUndefs: false,
  //                 variables: [ DF.variable('a'), DF.variable('b') ],
  //               },
  //             ),
  //             type: 'bindings',
  //           },
  //           operation: <any> {},
  //         },
  //         {
  //           output: {
  //             bindingsStream: new ArrayIterator([
  //               BF.bindings([
  //                 [ DF.variable('d'), DF.literal('d1') ],
  //                 [ DF.variable('e'), DF.literal('e1') ],
  //               ]),
  //               BF.bindings([
  //                 [ DF.variable('d'), DF.literal('d2') ],
  //                 [ DF.variable('e'), DF.literal('e2') ],
  //               ]),
  //             ]),
  //             metadata: () => Promise.resolve(
  //               {
  //                 state: new MetadataValidationState(),
  //                 cardinality: { type: 'estimate', value: 5 },
  //                 pageSize: 100,
  //                 requestTime: 30,
  //                 canContainUndefs: false,
  //                 variables: [ DF.variable('d'), DF.variable('e') ],
  //               },
  //             ),
  //             type: 'bindings',
  //           },
  //           operation: <any> {},
  //         },
  //         {
  //           output: {
  //             bindingsStream: new ArrayIterator([
  //               BF.bindings([
  //                 [ DF.variable('d'), DF.literal('d1') ],
  //                 [ DF.variable('f'), DF.literal('f1') ],
  //               ]),
  //               BF.bindings([
  //                 [ DF.variable('d'), DF.literal('d2') ],
  //                 [ DF.variable('f'), DF.literal('f2') ],
  //               ]),
  //             ]),
  //             metadata: () => Promise.resolve(
  //               {
  //                 state: new MetadataValidationState(),
  //                 cardinality: { type: 'estimate', value: 1 },
  //                 pageSize: 100,
  //                 requestTime: 30,
  //                 canContainUndefs: false,
  //                 variables: [ DF.variable('d'), DF.variable('f') ],
  //               },
  //             ),
  //             type: 'bindings',
  //           },
  //           operation: <any> {},
  //         },
  //         {
  //           output: {
  //             bindingsStream: new ArrayIterator([
  //               BF.bindings([
  //                 [ DF.variable('e'), DF.literal('e1') ],
  //                 [ DF.variable('f'), DF.literal('f1') ],
  //               ]),
  //               BF.bindings([
  //                 [ DF.variable('e'), DF.literal('e2') ],
  //                 [ DF.variable('f'), DF.literal('f2') ],
  //               ]),
  //             ]),
  //             metadata: () => Promise.resolve(
  //               {
  //                 state: new MetadataValidationState(),
  //                 cardinality: { type: 'estimate', value: 3 },
  //                 pageSize: 100,
  //                 requestTime: 30,
  //                 canContainUndefs: false,
  //                 variables: [ DF.variable('e'), DF.variable('f') ],
  //               },
  //             ),
  //             type: 'bindings',
  //           },
  //           operation: <any> {},
  //         },

  //       ],
  //       context,
  //     });
  //   });
  // it('should run on 2 disjoint size 3 streams', async() => {
  //   const action = action23();
  //   const output = await actor.run(action);
  //   expect(output.type).toEqual('bindings');
  //   // expect(await (<any> output).metadata()).toEqual({
  //   //   state: expect.any(MetadataValidationState),
  //   //   cardinality: { type: 'estimate', value: 40 },
  //   //   canContainUndefs: false,
  //   //   variables: [ DF.variable('a'), DF.variable('c'), DF.variable('b') ],
  //   // });
  //   await expect(output.bindingsStream).toEqualBindingsStream([
  //     BF.bindings([
  //       [ DF.variable('a'), DF.literal('a1') ],
  //       [ DF.variable('b'), DF.literal('b1') ],
  //       [ DF.variable('c'), DF.literal('c1') ],
  //     ]),
  //     BF.bindings([
  //       [ DF.variable('a'), DF.literal('a2') ],
  //       [ DF.variable('b'), DF.literal('b2') ],
  //       [ DF.variable('c'), DF.literal('c2') ],
  //     ]),
  //     BF.bindings([
  //       [ DF.variable('d'), DF.literal('d1') ],
  //       [ DF.variable('e'), DF.literal('e1') ],
  //       [ DF.variable('f'), DF.literal('f1') ],
  //     ]),
  //     BF.bindings([
  //       [ DF.variable('d'), DF.literal('d2') ],
  //       [ DF.variable('e'), DF.literal('e2') ],
  //       [ DF.variable('f'), DF.literal('f2') ],
  //     ]),
  //   ]);

  //   // Check join order

  //   expect((<any> action.entries[0]).operation.called).toBe(1);
  //   expect((<any> action.entries[1]).operation.called).toBe(1);
  //   expect((<any> action.entries[2]).operation.called).toBe(1);
  //   expect((<any> action.entries[3]).operation.called).toBe(1);
  //   expect((<any> action.entries[4]).operation.called).toBe(0);
  //   expect((<any> action.entries[5]).operation.called).toBe(0);

  // });
  // });
});
