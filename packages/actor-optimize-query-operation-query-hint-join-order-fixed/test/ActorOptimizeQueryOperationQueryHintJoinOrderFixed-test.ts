import { KeysInitQuery, KeysQueryOperation } from '@comunica/context-entries';
import { ActionContext, Bus } from '@comunica/core';
import type { IActionContext } from '@comunica/types';
import { AlgebraFactory } from '@comunica/utils-algebra';
import { DataFactory } from 'rdf-data-factory';
import {
  ActorOptimizeQueryOperationQueryHintJoinOrderFixed,
} from '../lib/ActorOptimizeQueryOperationQueryHintJoinOrderFixed';
import '@comunica/utils-jest';

const DF = new DataFactory();

describe('ActorOptimizeQueryOperationQueryHintJoinOrderFixed', () => {
  let bus: any;
  let factory: AlgebraFactory;

  beforeEach(() => {
    bus = new Bus({ name: 'bus' });
    factory = new AlgebraFactory(DF);
  });

  describe('An ActorOptimizeQueryOperationQueryHintJoinOrderFixed instance', () => {
    let actor: ActorOptimizeQueryOperationQueryHintJoinOrderFixed;
    let context: IActionContext;

    beforeEach(() => {
      actor = new ActorOptimizeQueryOperationQueryHintJoinOrderFixed({ name: 'actor', bus });
      context = new ActionContext({ [KeysInitQuery.dataFactory.name]: DF });
    });

    it('should test', async() => {
      await expect(actor.test({ operation: <any> undefined, context })).resolves.toPassTestVoid();
    });

    it('should not modify an operation without hint triple', async() => {
      const operation = factory.createJoin([
        factory.createPattern(DF.namedNode('s1'), DF.namedNode('p1'), DF.namedNode('o1')),
        factory.createPattern(DF.namedNode('s2'), DF.namedNode('p2'), DF.namedNode('o2')),
      ]);
      const result = await actor.run({ operation, context });
      expect(result.operation).toMatchObject(operation);
      expect(result.context.get(KeysQueryOperation.isJoinOrderFixed)).toBeUndefined();
    });

    it('should remove the hint triple and set context when hint is present', async() => {
      const hintPattern = factory.createPattern(
        DF.namedNode('https://comunica.dev/hint'),
        DF.namedNode('https://comunica.dev/optimizer'),
        DF.literal('None'),
      );
      const regularPattern = factory.createPattern(
        DF.namedNode('s1'),
        DF.namedNode('p1'),
        DF.namedNode('o1'),
      );
      const operation = factory.createJoin([
        hintPattern,
        regularPattern,
      ]);
      const result = await actor.run({ operation, context });
      expect(result.context.get(KeysQueryOperation.isJoinOrderFixed)).toBe(true);
      // The hint pattern should be replaced with a NOP
      expect(result.operation.type).toBe('join');
      expect((<any> result.operation).input[0].type).toBe('nop');
      expect((<any> result.operation).input[1].type).toBe('pattern');
    });

    it('should handle hint triple in a nested operation', async() => {
      const hintPattern = factory.createPattern(
        DF.namedNode('https://comunica.dev/hint'),
        DF.namedNode('https://comunica.dev/optimizer'),
        DF.literal('None'),
      );
      const regularPattern = factory.createPattern(
        DF.namedNode('s1'),
        DF.namedNode('p1'),
        DF.namedNode('o1'),
      );
      const operation = factory.createProject(
        factory.createJoin([
          hintPattern,
          regularPattern,
        ]),
        [],
      );
      const result = await actor.run({ operation, context });
      expect(result.context.get(KeysQueryOperation.isJoinOrderFixed)).toBe(true);
      expect(result.operation.type).toBe('project');
      const join = (<any> result.operation).input;
      expect(join.type).toBe('join');
      expect(join.input[0].type).toBe('nop');
      expect(join.input[1].type).toBe('pattern');
    });

    it('should not set context when pattern has wrong subject', async() => {
      const operation = factory.createPattern(
        DF.namedNode('https://wrong.dev/hint'),
        DF.namedNode('https://comunica.dev/optimizer'),
        DF.literal('None'),
      );
      const result = await actor.run({ operation, context });
      expect(result.context.get(KeysQueryOperation.isJoinOrderFixed)).toBeUndefined();
    });

    it('should not set context when pattern has wrong predicate', async() => {
      const operation = factory.createPattern(
        DF.namedNode('https://comunica.dev/hint'),
        DF.namedNode('https://wrong.dev/optimizer'),
        DF.literal('None'),
      );
      const result = await actor.run({ operation, context });
      expect(result.context.get(KeysQueryOperation.isJoinOrderFixed)).toBeUndefined();
    });

    it('should not set context when pattern has wrong object', async() => {
      const operation = factory.createPattern(
        DF.namedNode('https://comunica.dev/hint'),
        DF.namedNode('https://comunica.dev/optimizer'),
        DF.literal('Something'),
      );
      const result = await actor.run({ operation, context });
      expect(result.context.get(KeysQueryOperation.isJoinOrderFixed)).toBeUndefined();
    });

    it('should not set context when object is a named node instead of a literal', async() => {
      const operation = factory.createPattern(
        DF.namedNode('https://comunica.dev/hint'),
        DF.namedNode('https://comunica.dev/optimizer'),
        DF.namedNode('None'),
      );
      const result = await actor.run({ operation, context });
      expect(result.context.get(KeysQueryOperation.isJoinOrderFixed)).toBeUndefined();
    });

    it('should not set context when subject is a variable', async() => {
      const operation = factory.createPattern(
        DF.variable('hint'),
        DF.namedNode('https://comunica.dev/optimizer'),
        DF.literal('None'),
      );
      const result = await actor.run({ operation, context });
      expect(result.context.get(KeysQueryOperation.isJoinOrderFixed)).toBeUndefined();
    });
  });
});
