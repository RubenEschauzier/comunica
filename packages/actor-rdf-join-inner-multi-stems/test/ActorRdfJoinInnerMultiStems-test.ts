import { Bus } from '@comunica/core';
import type { IJoinEntryWithMetadata, MetadataVariable, QueryResultCardinality } from '@comunica/types';
import { BindingsFactory } from '@comunica/utils-bindings-factory';
import { MetadataValidationState } from '@comunica/utils-metadata';
import type * as RDF from '@rdfjs/types';
import { ArrayIterator } from 'asynciterator';
import { DataFactory } from 'rdf-data-factory';
import { toSparql } from 'sparqlalgebrajs';
import { ActorRdfJoinMultiStems } from '../lib/ActorRdfJoinMultiStems';
import { RouterFixedMinimalIndex } from '../lib/routers/FixedRouter';

const DF = new DataFactory();
const BF = new BindingsFactory(DF);

describe('ActorRdfJoinInnerMultiStems', () => {
  let bus: any;

  beforeEach(() => {
    bus = new Bus({ name: 'bus' });
  });

  describe('An ActorRdfJoinInnerMultiStems instance', () => {
    it('should correctly find connected components within join graph', () => {
      const entries = createEntriesWithDifferentJoinVars(
        [
          [ 'a', 'b' ],
          [ 'd', 'c' ],
          [ 'c', 'f' ],
          [ 'f', 'b' ],
          [ 'g', 'h' ],
          [ 'h', 'i' ],
          [ 'i', 'j' ],
        ],
      );
      const disjointEntries = ActorRdfJoinMultiStems.findConnectedComponentsInJoinGraph(entries);
      expect(disjointEntries.entries).toHaveLength(2);
      expect(disjointEntries.entries).toEqual([ entries.slice(0, 4), entries.slice(4, 7) ]);
      expect(disjointEntries.indexes).toEqual([[ 0, 1, 2, 3 ], [ 4, 5, 6 ]]);
    });

    it('should return each entry as a separate group when all are disjoint', () => {
      const entries = createEntriesWithDifferentJoinVars([
        [ 'a', 'b' ],
        [ 'c', 'd' ],
        [ 'e', 'f' ],
        [ 'g', 'h' ],
      ]);
      const disjointEntries = ActorRdfJoinMultiStems.findConnectedComponentsInJoinGraph(entries);
      expect(disjointEntries.entries).toHaveLength(4);
      for (const [ i, group ] of disjointEntries.entries.entries()) {
        expect(group).toEqual([ entries[i] ]);
      }
      for (const [ i, group ] of disjointEntries.indexes.entries()) {
        expect(group).toEqual([ i ]);
      }
    });

    it('should return a single group when all entries are connected transitively', () => {
      const entries = createEntriesWithDifferentJoinVars([
        [ 'a', 'b' ],
        [ 'b', 'c' ],
        [ 'c', 'd' ],
        [ 'd', 'e' ],
      ]);
      const disjointEntries = ActorRdfJoinMultiStems.findConnectedComponentsInJoinGraph(entries);
      expect(disjointEntries.entries).toHaveLength(1);
      expect(disjointEntries.entries).toEqual([ entries ]);
    });

    it('should treat entries with one variable as disjoint if no overlaps', () => {
      const entries = createEntriesWithDifferentJoinVars([
        [ 'a' ],
        [ 'b' ],
        [ 'c' ],
      ]);
      const disjointEntries = ActorRdfJoinMultiStems.findConnectedComponentsInJoinGraph(entries);
      expect(disjointEntries.entries).toHaveLength(3);
      expect(disjointEntries.entries).toEqual([[ entries[0] ], [ entries[1] ], [ entries[2] ]]);
      expect(disjointEntries.indexes).toEqual([[ 0 ], [ 1 ], [ 2 ]]);
    });

    it('should group all entries connected via a common variable (star shape)', () => {
      const entries = createEntriesWithDifferentJoinVars([
        [ 'x', 'a' ],
        [ 'x', 'b' ],
        [ 'x', 'c' ],
        [ 'x', 'd' ],
      ]);
      const disjointEntries = ActorRdfJoinMultiStems.findConnectedComponentsInJoinGraph(entries);
      expect(disjointEntries.entries).toHaveLength(1);
      expect(disjointEntries.entries).toEqual([ entries ]);
      expect(disjointEntries.indexes).toEqual([[ 0, 1, 2, 3 ]]);
    });

    it('should find multiple disjoint groups in a complex graph with cycles', () => {
      const entries = createEntriesWithDifferentJoinVars([
        [ 'a', 'b' ], // 0
        [ 'b', 'c' ], // 1
        [ 'c', 'a' ], // 2 (forms cycle with 0,1)
        [ 'x', 'y' ], // 3
        [ 'y', 'z' ], // 4
        [ 'p', 'q' ], // 5
        [ 'q', 'r' ], // 6
        [ 's', 't' ], // 7
      ]);
      const disjointEntries = ActorRdfJoinMultiStems.findConnectedComponentsInJoinGraph(entries);
      expect(disjointEntries.entries).toHaveLength(4);
      expect(disjointEntries.entries).toEqual(
        [ entries.slice(0, 3), entries.slice(3, 5), entries.slice(5, 7), entries.slice(7) ],
      );
      expect(disjointEntries.indexes).toEqual([[ 0, 1, 2 ], [ 3, 4 ], [ 5, 6 ], [ 7 ]]);
    });
    it('should group entries that share the same constant subject (subject star-join)', () => {
      const entry1 = createEntryWithPattern([ 'name' ], DF.namedNode('http://example.org/Alice'), DF.namedNode('http://example.org/name'), DF.variable('name'));
      const entry2 = createEntryWithPattern([ 'age' ], DF.namedNode('http://example.org/Alice'), DF.namedNode('http://example.org/age'), DF.variable('age'));
      const entry3 = createEntryWithPattern([ 'city' ], DF.namedNode('http://example.org/Alice'), DF.namedNode('http://example.org/livesIn'), DF.variable('city'));
      const entry4 = createEntryWithPattern([ 'unrelated' ], DF.namedNode('http://example.org/Bob'), DF.namedNode('http://example.org/name'), DF.variable('unrelated'));

      const result = ActorRdfJoinMultiStems.findConnectedComponentsInJoinGraph([ entry1, entry2, entry3, entry4 ]);
      expect(result.entries).toHaveLength(2);
      expect(result.indexes).toEqual([[ 0, 1, 2 ], [ 3 ]]);
    });

    it('should NOT group entries that only share an object constant (e.g. rdf:type)', () => {
      const entry1 = createEntryWithPattern([ 'person' ], DF.variable('person'), DF.namedNode('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'), DF.namedNode('http://example.org/LivingBeing'));
      const entry2 = createEntryWithPattern([ 'dog' ], DF.variable('dog'), DF.namedNode('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'), DF.namedNode('http://example.org/LivingBeing'));

      const result = ActorRdfJoinMultiStems.findConnectedComponentsInJoinGraph([ entry1, entry2 ]);
      expect(result.entries).toHaveLength(2);
      expect(result.indexes).toEqual([[ 0 ], [ 1 ]]);
    });
  });

  describe('RouterBase operations mapping and route table', () => {
    it('should create route table with matching operations and resolve done masks', () => {
      const op1: any = {
        type: 'pattern',
        subject: DF.variable('x'),
        predicate: DF.namedNode('http://example.org/p1'),
        object: DF.variable('y'),
        graph: DF.defaultGraph(),
      };
      const op2: any = {
        type: 'pattern',
        subject: DF.variable('y'),
        predicate: DF.namedNode('http://example.org/p2'),
        object: DF.variable('z'),
        graph: DF.defaultGraph(),
      };
      const op3: any = {
        type: 'pattern',
        subject: DF.variable('z'),
        predicate: DF.namedNode('http://example.org/p3'),
        object: DF.variable('w'),
        graph: DF.defaultGraph(),
      };

      const router = new RouterFixedMinimalIndex();
      const routeOperations = [
        { operation: op1, doneBitMask: 1, variables: [ DF.variable('x'), DF.variable('y') ], namedNodes: [] },
        { operation: op2, doneBitMask: 2, variables: [ DF.variable('y'), DF.variable('z') ], namedNodes: [] },
        { operation: op3, doneBitMask: 4, variables: [ DF.variable('z'), DF.variable('w') ], namedNodes: [] },
      ];

      const routeTable = router.createRouteTable(routeOperations);
      // State 1 = only op1 done (bit 0)
      expect(routeTable[1]).toBeDefined();
      expect(routeTable[1][0][0].next).toBe(1);
      expect(routeTable[1][0][0].operation).toBe(op2);

      // Calling createRouteTable again with identical operations is allowed
      expect(() => router.createRouteTable(routeOperations)).not.toThrow();

      // Calling createRouteTable with different operations throws
      const differentOps = [
        { operation: op1, doneBitMask: 1, variables: [ DF.variable('x') ], namedNodes: [] },
        { operation: op3, doneBitMask: 4, variables: [ DF.variable('z') ], namedNodes: [] },
      ];
      expect(() => router.createRouteTable(differentOps)).toThrow(/Router state error/);
    });

    it('should correctly build routes for composite derived resource operations', () => {
      const op1: any = { type: 'pattern', subject: DF.variable('x'), predicate: DF.namedNode('http://example.org/p1'), object: DF.variable('y'), graph: DF.defaultGraph() };
      const op2: any = { type: 'pattern', subject: DF.variable('y'), predicate: DF.namedNode('http://example.org/p2'), object: DF.variable('z'), graph: DF.defaultGraph() };
      const op3: any = { type: 'pattern', subject: DF.variable('z'), predicate: DF.namedNode('http://example.org/p3'), object: DF.variable('w'), graph: DF.defaultGraph() };
      const opDerived12: any = { type: 'join', left: op1, right: op2 };

      const router = new RouterFixedMinimalIndex();
      const routeOperations = [
        // Base op1 (bit 0 = 1)
        { operation: op1, doneBitMask: 1, variables: [ DF.variable('x'), DF.variable('y') ], namedNodes: [] },
        // Base op2 (bit 1 = 2)
        { operation: op2, doneBitMask: 2, variables: [ DF.variable('y'), DF.variable('z') ], namedNodes: [] },
        // Base op3 (bit 2 = 4)
        { operation: op3, doneBitMask: 4, variables: [ DF.variable('z'), DF.variable('w') ], namedNodes: [] },
        // Composite derived resource covering op1 and op2 (bitmask 1 | 2 = 3)
        { operation: opDerived12, doneBitMask: 3, variables: [ DF.variable('x'), DF.variable('y'), DF.variable('z') ], namedNodes: [] },
      ];

      const routeTable = router.createRouteTable(routeOperations);
      // At state 1 (op1 done), opDerived12 has doneBitMask 3 (state & 3 != 0), so it is excluded
      const nextFromState1 = routeTable[1][0].map(r => r.next);
      expect(nextFromState1).toContain(1); // op2
      expect(nextFromState1).not.toContain(3); // derived resource not allowed since op1 is already done

      // At state 3 (bits 1 and 2 done, e.g. produced directly by opDerived12), next route should be op3 (index 2)
      expect(routeTable[3]).toBeDefined();
      expect(routeTable[3][0][0].next).toBe(2);
      expect(routeTable[3][0][0].operation).toBe(op3);
    });
  });
});

function createEntryWithPattern(
  variableValues: string[],
  subject: RDF.Term,
  predicate: RDF.Term,
  object: RDF.Term,
): IJoinEntryWithMetadata {
  const variables: MetadataVariable[] = variableValues.map(value => ({
    variable: DF.variable(value),
    canBeUndef: false,
  }));
  const operation: any = {
    type: 'pattern',
    subject,
    predicate,
    object,
    graph: DF.defaultGraph(),
  };
  return {
    output: {
      bindingsStream: new ArrayIterator<RDF.Bindings>([]),
      metadata: () => Promise.resolve({
        state: new MetadataValidationState(),
        cardinality: <QueryResultCardinality> { type: 'estimate', value: 4 },
        pageSize: 100,
        requestTime: 10,
        variables,
      }),
      type: <any> 'bindings',
    },
    operation,
    metadata: {
      state: new MetadataValidationState(),
      cardinality: <QueryResultCardinality> { type: 'estimate', value: 4 },
      pageSize: 100,
      requestTime: 10,
      variables,
    },
  };
}

function createEntriesWithDifferentJoinVars(variableValues: string[][]): IJoinEntryWithMetadata[] {
  const entriesWithVariablesSet = variableValues.map((values) => {
    const variables: MetadataVariable[] = values.map((value) => {
      return { variable: DF.variable(value), canBeUndef: false };
    });
    return {
      output: {
        bindingsStream: new ArrayIterator<RDF.Bindings>([]),
        metadata: () => Promise.resolve({
          state: new MetadataValidationState(),
          cardinality: <QueryResultCardinality> { type: 'estimate', value: 4 },
          pageSize: 100,
          requestTime: 10,
          variables,
        }),
        type: <any> 'bindings',
      },
      operation: <any> { type: 'pattern', subject: DF.variable('s'), predicate: DF.variable('p'), object: DF.variable('o'), graph: DF.defaultGraph() },
      metadata: {
        state: new MetadataValidationState(),
        cardinality: <QueryResultCardinality> { type: 'estimate', value: 4 },
        pageSize: 100,
        requestTime: 10,
        variables,
      },
    };
  });

  return entriesWithVariablesSet;
}
