import { ActorQueryOperation } from '@comunica/bus-query-operation';
import { KeysInitQuery } from '@comunica/context-entries';
import { ActionContext, Bus } from '@comunica/core';
import type { IJoinEntry } from '@comunica/types';
import { TypesComunica } from '@comunica/utils-algebra';
import { BindingsFactory } from '@comunica/utils-bindings-factory';
import { MetadataValidationState } from '@comunica/utils-metadata';
import { getSafeBindings } from '@comunica/utils-query-operation';
import { ArrayIterator, UnionIterator } from 'asynciterator';
import { DataFactory } from 'rdf-data-factory';
import { ActorQueryOperationHintedGroup } from '../lib/ActorQueryOperationHintedGroup';
import '@comunica/utils-jest';

const DF = new DataFactory();
const BF = new BindingsFactory(DF);

describe('ActorQueryOperationHintedGroup', () => {
  let bus: any;
  let mediatorQueryOperation: any;
  let mediatorJoin: any;

  beforeEach(() => {
    bus = new Bus({ name: 'bus' });
    mediatorQueryOperation = {
      mediate: (arg: any) => Promise.resolve({
        bindingsStream: new ArrayIterator([
          BF.bindings([[ DF.variable('a'), DF.literal('1') ]]),
          BF.bindings([[ DF.variable('a'), DF.literal('2') ]]),
        ], { autoStart: false }),
        metadata: () => Promise.resolve({
          cardinality: { type: 'exact', value: 2 },
          variables: [{ variable: DF.variable('a'), canBeUndef: false }],
        }),
        operated: arg,
        type: 'bindings',
      }),
    };
    mediatorJoin = {
      mediate: (arg: any) => Promise.resolve({
        bindingsStream: new UnionIterator(
          arg.entries.map((entry: IJoinEntry) => entry.output.bindingsStream),
          { autoStart: false },
        ),
        metadata: () => Promise.resolve({
          cardinality: { type: 'exact', value: 100 },
          variables: [
            { variable: DF.variable('a'), canBeUndef: false },
            { variable: DF.variable('b'), canBeUndef: false },
          ],
        }),
        operated: arg,
        type: 'bindings',
      }),
    };
  });

  describe('The ActorQueryOperationHintedGroup module', () => {
    it('should be a function', () => {
      expect(ActorQueryOperationHintedGroup).toBeInstanceOf(Function);
    });

    it('should be a ActorQueryOperationHintedGroup constructor', () => {
      expect(new (<any> ActorQueryOperationHintedGroup)(
        { name: 'actor', bus, mediatorQueryOperation, mediatorJoin },
      )).toBeInstanceOf(ActorQueryOperationHintedGroup);
      expect(new (<any> ActorQueryOperationHintedGroup)(
        { name: 'actor', bus, mediatorQueryOperation, mediatorJoin },
      )).toBeInstanceOf(ActorQueryOperation);
    });
  });

  describe('An ActorQueryOperationHintedGroup instance', () => {
    let actor: ActorQueryOperationHintedGroup;

    beforeEach(() => {
      actor = new ActorQueryOperationHintedGroup(
        { name: 'actor', bus, mediatorQueryOperation, mediatorJoin },
      );
    });

    it('should test on hintedgroup', async() => {
      const op: any = { operation: { type: TypesComunica.HINTED_GROUP }};
      await expect(actor.test(op)).resolves.toPassTestVoid();
    });

    it('should not test on non-hintedgroup', async() => {
      const op: any = { operation: { type: 'join' }};
      await expect(actor.test(op)).resolves
        .toFailTest(`Actor actor only supports ${TypesComunica.HINTED_GROUP} operations, but got join`);
    });

    it('should run with two inputs', async() => {
      const op: any = {
        operation: { type: TypesComunica.HINTED_GROUP, input: [{}, {}]},
        context: new ActionContext({ [KeysInitQuery.dataFactory.name]: DF }),
      };
      const output = getSafeBindings(await actor.run(op, undefined));
      expect(output.type).toBe('bindings');
      await expect(output.metadata()).resolves.toEqual({
        cardinality: { type: 'exact', value: 100 },
        variables: [
          { variable: DF.variable('a'), canBeUndef: false },
          { variable: DF.variable('b'), canBeUndef: false },
        ],
      });
    });

    it('should run with three inputs and preserve order', async() => {
      const joinCalls: any[] = [];
      mediatorJoin.mediate = (arg: any) => {
        joinCalls.push(arg);
        return Promise.resolve({
          bindingsStream: new ArrayIterator([
            BF.bindings([[ DF.variable('a'), DF.literal('joined') ]]),
          ], { autoStart: false }),
          metadata: () => Promise.resolve({
            cardinality: { type: 'exact', value: 1 },
            variables: [{ variable: DF.variable('a'), canBeUndef: false }],
          }),
          type: 'bindings',
        });
      };

      const op: any = {
        operation: { type: TypesComunica.HINTED_GROUP, input: [{}, {}, {}]},
        context: new ActionContext({ [KeysInitQuery.dataFactory.name]: DF }),
      };
      const output = getSafeBindings(await actor.run(op, undefined));
      expect(output.type).toBe('bindings');
      // Should make 2 join calls: first two entries, then result + third
      expect(joinCalls).toHaveLength(2);
      // Both should be inner joins
      expect(joinCalls[0].type).toBe('inner');
      expect(joinCalls[1].type).toBe('inner');
      // First call joins entries 0 and 1
      expect(joinCalls[0].entries).toHaveLength(2);
      // Second call joins result with entry 2
      expect(joinCalls[1].entries).toHaveLength(2);
    });

    it('should run with a single input', async() => {
      const op: any = {
        operation: { type: TypesComunica.HINTED_GROUP, input: [{}]},
        context: new ActionContext({ [KeysInitQuery.dataFactory.name]: DF }),
      };
      const output = getSafeBindings(await actor.run(op, undefined));
      expect(output.type).toBe('bindings');
      await expect(output.bindingsStream).toEqualBindingsStream([
        BF.bindings([[ DF.variable('a'), DF.literal('1') ]]),
        BF.bindings([[ DF.variable('a'), DF.literal('2') ]]),
      ]);
    });

    it('should return empty when one entry has exact zero cardinality', async() => {
      mediatorQueryOperation.mediate = (arg: any) => Promise.resolve({
        bindingsStream: new ArrayIterator([], { autoStart: false }),
        metadata: () => Promise.resolve({
          cardinality: { type: 'exact', value: 0 },
          variables: [{ variable: DF.variable('a'), canBeUndef: false }],
        }),
        operated: arg,
        type: 'bindings',
      });

      const op: any = {
        operation: { type: TypesComunica.HINTED_GROUP, input: [{}, {}]},
        context: new ActionContext({ [KeysInitQuery.dataFactory.name]: DF }),
      };
      const output = getSafeBindings(await actor.run(op, undefined));
      expect(output.type).toBe('bindings');
      await expect(output.metadata()).resolves.toEqual({
        state: expect.any(MetadataValidationState),
        cardinality: { type: 'exact', value: 0 },
        variables: [
          { variable: DF.variable('a'), canBeUndef: false },
        ],
      });
      await expect(output.bindingsStream).toEqualBindingsStream([]);
    });
  });
});
