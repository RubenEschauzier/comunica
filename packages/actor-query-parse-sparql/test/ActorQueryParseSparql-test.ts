import { ActorQueryParse } from '@comunica/bus-query-parse';
import { KeysInitQuery } from '@comunica/context-entries';
import { ActionContext, Bus } from '@comunica/core';
import type { IActionContext } from '@comunica/types';
import { AlgebraFactory, TypesComunica } from '@comunica/utils-algebra';
import { DataFactory } from 'rdf-data-factory';
import { ActorQueryParseSparql } from '..';
import '@comunica/utils-jest';

const DF = new DataFactory();
const AF = new AlgebraFactory(DF);

describe('ActorQueryParseSparql', () => {
  let bus: any;

  beforeEach(() => {
    bus = new Bus({ name: 'bus' });
  });

  describe('The ActorQueryParseSparql module', () => {
    it('should be a function', () => {
      expect(ActorQueryParseSparql).toBeInstanceOf(Function);
    });

    it('should be a ActorQueryParseSparql constructor', () => {
      expect(new (<any> ActorQueryParseSparql)({ name: 'actor', bus })).toBeInstanceOf(ActorQueryParseSparql);
      expect(new (<any> ActorQueryParseSparql)({ name: 'actor', bus })).toBeInstanceOf(ActorQueryParse);
    });

    it('should not be able to create new ActorQueryParseSparql objects without \'new\'', () => {
      expect(() => {
        (<any> ActorQueryParseSparql)();
      }).toThrow(`Class constructor ActorQueryParseSparql cannot be invoked without 'new'`);
    });
  });

  describe('An ActorQueryParseSparql instance', () => {
    let actor: ActorQueryParseSparql;
    let context: IActionContext;

    beforeEach(() => {
      actor = new ActorQueryParseSparql({ name: 'actor', bus });
      context = new ActionContext({ [KeysInitQuery.dataFactory.name]: DF });
    });

    it('should not test on the graphql format', async() => {
      await expect(actor.test({
        query: 'a',
        queryFormat: { language: 'graphql', version: '1.0' },
        context,
      }))
        .resolves.toFailTest(`This actor can only parse SPARQL queries`);
    });

    it('should test on no format', async() => {
      await expect(actor.test({
        query: 'a',
        context,
      })).resolves.toPassTestVoid();
    });

    it('should test on the sparql format', async() => {
      await expect(actor.test({
        query: 'a',
        queryFormat: { language: 'sparql', version: '1.1' },
        context,
      }))
        .resolves.toPassTestVoid();
    });

    it('should run', async() => {
      const result = await actor.run({ query: 'SELECT * WHERE { ?a a ?b }', context });
      expect(result).toMatchObject({
        operation: AF.createProject(AF.createBgp([ AF.createPattern(
          DF.variable('a'),
          DF.namedNode('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'),
          DF.variable('b'),
        ) ]), [ DF.variable('a'), DF.variable('b') ]),
      });
    });

    it('should run for a SPARQL-star query', async() => {
      const result = await actor.run({ query: 'SELECT * WHERE { << ?a a ?q >> a ?b }', context });
      expect(result).toBeTruthy();
    });

    it('should run for an update query', async() => {
      const result = await actor.run({
        query: 'INSERT { <http://example/egbook> <http://ex.org/p> "A" } WHERE {}',
        context,
      });
      expect(result).toMatchObject({
        operation: AF.createDeleteInsert(undefined, [ AF.createPattern(
          DF.namedNode('http://example/egbook'),
          DF.namedNode('http://ex.org/p'),
          DF.literal('A'),
        ) ]),
      });
    });

    it('should run with an overridden baseIRI', async() => {
      const result = await actor.run({
        query: 'BASE <http://example.org/book/> SELECT * WHERE { ?a a ?b }',
        context,
      });
      expect(result).toMatchObject({
        baseIRI: 'http://example.org/book/',
        operation: AF.createProject(AF.createBgp([ AF.createPattern(
          DF.variable('a'),
          DF.namedNode('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'),
          DF.variable('b'),
        ) ]), [ DF.variable('a'), DF.variable('b') ]),
      });
    });

    it('should produce hinted-group when query hint triple is present', async() => {
      const actorWithPrefix = new ActorQueryParseSparql({
        name: 'actor',
        bus,
        prefixes: { comunica: 'http://comunica-internal/' },
      });
      const query = `SELECT * WHERE {
        comunica:hint comunica:optimizer "None" .
        { ?a a ?b . }
        { ?c a ?d . }
      }`;
      const result = await actorWithPrefix.run({ query, context });
      expect(result.operation.type).toBe('project');
      const inner = (<any> result.operation).input;
      expect(inner.type).toBe(TypesComunica.HINTED_GROUP);
      expect(inner.input).toHaveLength(2);
      // Both sub-operations should be BGPs (one per group)
      expect(inner.input[0].type).toBe('bgp');
      expect(inner.input[1].type).toBe('bgp');
    });

    it('should produce nested hinted-group for nested braces', async() => {
      const actorWithPrefix = new ActorQueryParseSparql({
        name: 'actor',
        bus,
        prefixes: { comunica: 'http://comunica-internal/' },
      });
      const query = `SELECT * WHERE {
        comunica:hint comunica:optimizer "None" .
        {
          { ?a a ?b . }
          { ?c a ?d . }
        }
        ?e a ?f .
      }`;
      const result = await actorWithPrefix.run({ query, context });
      expect(result.operation.type).toBe('project');
      const outer = (<any> result.operation).input;
      expect(outer.type).toBe(TypesComunica.HINTED_GROUP);
      expect(outer.input).toHaveLength(2);
      // First entry is a nested hinted-group
      expect(outer.input[0].type).toBe(TypesComunica.HINTED_GROUP);
      expect(outer.input[0].input).toHaveLength(2);
      // Second entry is a BGP for ?e a ?f
      expect(outer.input[1].type).toBe('bgp');
    });

    it('should not produce hinted-group when hint is absent', async() => {
      const query = `SELECT * WHERE {
        { ?a a ?b . }
        { ?c a ?d . }
      }`;
      const result = await actor.run({ query, context });
      // Without hint, nested groups should be flattened into a single BGP
      expect(result.operation.type).toBe('project');
      const inner = (<any> result.operation).input;
      expect(inner.type).toBe('bgp');
    });
  });
});
