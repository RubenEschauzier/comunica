import { ActionContext, Bus } from '@comunica/core';
import { ActorHashBindingsMurmur } from '@comunica/actor-hash-bindings-murmur';
import type { BindingsStream, IJoinEntryWithMetadata, MetadataVariable, QueryResultCardinality } from '@comunica/types';
import { BindingsFactory, type Bindings } from '@comunica/utils-bindings-factory';
import { MetadataValidationState } from '@comunica/utils-metadata';
import type * as RDF from '@rdfjs/types';
import { ArrayIterator } from 'asynciterator';
import { DataFactory } from 'rdf-data-factory';
import { toSparql } from 'sparqlalgebrajs';
import { KeysMergeBindingsContext } from '@comunica/context-entries';
import { SetUnionBindingsContextMergeHandler } from '@comunica/actor-merge-bindings-context-union';
import { ActorRdfJoinMultiStems } from '../lib/ActorRdfJoinMultiStems';
import { RouterFixedMinimalIndex } from '../lib/routers/FixedRouter';
import { StemsControllerStream, TimestampGenerator, stemsContextKeys, type ITimestampGenerator } from '../lib/StemsControllerStream';
import { StemsOperatorStream, type HashFunction, type JoinFunction } from '../lib/StemsOperatorStream';

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
        { operations: [ op1 ], doneBitMask: 1, variables: [ DF.variable('x'), DF.variable('y') ], namedNodes: [] },
        { operations: [ op2 ], doneBitMask: 2, variables: [ DF.variable('y'), DF.variable('z') ], namedNodes: [] },
        { operations: [ op3 ], doneBitMask: 4, variables: [ DF.variable('z'), DF.variable('w') ], namedNodes: [] },
      ];

      const routeTable = router.createRouteTable(routeOperations);
      // State 1 = only op1 done (bit 0)
      expect(routeTable[1]).toBeDefined();
      expect(routeTable[1][0][0].next).toBe(1);
      expect(routeTable[1][0][0].operations).toEqual([ op2 ]);

      // Calling createRouteTable again with identical operations is allowed
      expect(() => router.createRouteTable(routeOperations)).not.toThrow();

      // Calling createRouteTable with different operations throws
      const differentOps = [
        { operations: [ op1 ], doneBitMask: 1, variables: [ DF.variable('x') ], namedNodes: [] },
        { operations: [ op3 ], doneBitMask: 4, variables: [ DF.variable('z') ], namedNodes: [] },
      ];
      expect(() => router.createRouteTable(differentOps)).toThrow(/Router state error/);
    });

    it('should correctly build routes for composite derived resource operations', () => {
      const op1: any = { type: 'pattern', subject: DF.variable('x'), predicate: DF.namedNode('http://example.org/p1'), object: DF.variable('y'), graph: DF.defaultGraph() };
      const op2: any = { type: 'pattern', subject: DF.variable('y'), predicate: DF.namedNode('http://example.org/p2'), object: DF.variable('z'), graph: DF.defaultGraph() };
      const op3: any = { type: 'pattern', subject: DF.variable('z'), predicate: DF.namedNode('http://example.org/p3'), object: DF.variable('w'), graph: DF.defaultGraph() };

      const router = new RouterFixedMinimalIndex();
      const routeOperations = [
        // Base op1 (bit 0 = 1)
        { operations: [ op1 ], doneBitMask: 1, variables: [ DF.variable('x'), DF.variable('y') ], namedNodes: [] },
        // Base op2 (bit 1 = 2)
        { operations: [ op2 ], doneBitMask: 2, variables: [ DF.variable('y'), DF.variable('z') ], namedNodes: [] },
        // Base op3 (bit 2 = 4)
        { operations: [ op3 ], doneBitMask: 4, variables: [ DF.variable('z'), DF.variable('w') ], namedNodes: [] },
        // Composite derived resource covering op1 and op2 (bitmask 1 | 2 = 3)
        { operations: [ op1, op2 ], doneBitMask: 3, variables: [ DF.variable('x'), DF.variable('y'), DF.variable('z') ], namedNodes: [] },
      ];

      const routeTable = router.createRouteTable(routeOperations);
      // At state 1 (op1 done), opDerived12 has doneBitMask 3 (state & 3 != 0), so it is excluded
      const nextFromState1 = routeTable[1][0].map(r => r.next);
      expect(nextFromState1).toContain(1); // op2
      expect(nextFromState1).not.toContain(3); // derived resource not allowed since op1 is already done

      // At state 3 (bits 1 and 2 done, e.g. produced directly by opDerived12), next route should be op3 (index 2)
      expect(routeTable[3]).toBeDefined();
      expect(routeTable[3][0][0].next).toBe(2);
      expect(routeTable[3][0][0].operations).toEqual([ op3 ]);
    });

    describe('5-TP Query with Composite Resources (Ground Truth Verification)', () => {
      // 5 Triple Patterns:
      // TP0: ?x :p1 ?y
      // TP1: ?y :p2 ?z
      // TP2: ?z :p3 ?w
      // TP3: ?w :p4 ?v
      // TP4: ?v :p5 ?u
      const op0: any = { type: 'pattern', subject: DF.variable('x'), predicate: DF.namedNode('http://example.org/p1'), object: DF.variable('y'), graph: DF.defaultGraph() };
      const op1: any = { type: 'pattern', subject: DF.variable('y'), predicate: DF.namedNode('http://example.org/p2'), object: DF.variable('z'), graph: DF.defaultGraph() };
      const op2: any = { type: 'pattern', subject: DF.variable('z'), predicate: DF.namedNode('http://example.org/p3'), object: DF.variable('w'), graph: DF.defaultGraph() };
      const op3: any = { type: 'pattern', subject: DF.variable('w'), predicate: DF.namedNode('http://example.org/p4'), object: DF.variable('v'), graph: DF.defaultGraph() };
      const op4: any = { type: 'pattern', subject: DF.variable('v'), predicate: DF.namedNode('http://example.org/p5'), object: DF.variable('u'), graph: DF.defaultGraph() };

      const baseOps = [ op0, op1, op2, op3, op4 ];

      // Synthetic data tuples
      // Item 1: x1 -> y1 -> z1 -> w1 -> v1 -> u1
      // Item 2: x2 -> y2 -> z2 -> w2 -> v2 -> u2
      const dataTP0 = [
        BF.bindings([[ DF.variable('x'), DF.literal('x1') ], [ DF.variable('y'), DF.literal('y1') ]]),
        BF.bindings([[ DF.variable('x'), DF.literal('x2') ], [ DF.variable('y'), DF.literal('y2') ]]),
      ];
      const dataTP1 = [
        BF.bindings([[ DF.variable('y'), DF.literal('y1') ], [ DF.variable('z'), DF.literal('z1') ]]),
        BF.bindings([[ DF.variable('y'), DF.literal('y2') ], [ DF.variable('z'), DF.literal('z2') ]]),
      ];
      const dataTP2 = [
        BF.bindings([[ DF.variable('z'), DF.literal('z1') ], [ DF.variable('w'), DF.literal('w1') ]]),
        BF.bindings([[ DF.variable('z'), DF.literal('z2') ], [ DF.variable('w'), DF.literal('w2') ]]),
      ];
      const dataTP3 = [
        BF.bindings([[ DF.variable('w'), DF.literal('w1') ], [ DF.variable('v'), DF.literal('v1') ]]),
        BF.bindings([[ DF.variable('w'), DF.literal('w2') ], [ DF.variable('v'), DF.literal('v2') ]]),
      ];
      const dataTP4 = [
        BF.bindings([[ DF.variable('v'), DF.literal('v1') ], [ DF.variable('u'), DF.literal('u1') ]]),
        BF.bindings([[ DF.variable('v'), DF.literal('v2') ], [ DF.variable('u'), DF.literal('u2') ]]),
      ];

      let hashFn: HashFunction;
      const joinFn: JoinFunction = <JoinFunction> ActorRdfJoinMultiStems.joinBindings;

      function sortBindings(bindingsList: RDF.Bindings[]): string[] {
        return bindingsList.map(b => {
          const entries: string[] = [];
          for (const [ v, val ] of b) {
            entries.push(`${v.value}=${val.value}`);
          }
          return entries.sort().join(';');
        }).sort();
      }

      function buildBaseStems(
        data0: RDF.Bindings[],
        data1: RDF.Bindings[],
        data2: RDF.Bindings[],
        data3: RDF.Bindings[],
        data4: RDF.Bindings[],
        timestampGen: ITimestampGenerator,
      ): StemsOperatorStream[] {
        return [
          new StemsOperatorStream(
            <BindingsStream> <unknown> new ArrayIterator(data0),
            timestampGen,
            hashFn,
            joinFn,
            0,
            1 << 0, // 1
            [ op0 ],
            [ DF.variable('x'), DF.variable('y') ],
            [],
            [[ DF.variable('y') ]],
            false,
          ),
          new StemsOperatorStream(
            <BindingsStream> <unknown> new ArrayIterator(data1),
            timestampGen,
            hashFn,
            joinFn,
            1,
            1 << 1, // 2
            [ op1 ],
            [ DF.variable('y'), DF.variable('z') ],
            [],
            [[ DF.variable('y') ], [ DF.variable('z') ]],
            false,
          ),
          new StemsOperatorStream(
            <BindingsStream> <unknown> new ArrayIterator(data2),
            timestampGen,
            hashFn,
            joinFn,
            2,
            1 << 2, // 4
            [ op2 ],
            [ DF.variable('z'), DF.variable('w') ],
            [],
            [[ DF.variable('z') ], [ DF.variable('w') ]],
            false,
          ),
          new StemsOperatorStream(
            <BindingsStream> <unknown> new ArrayIterator(data3),
            timestampGen,
            hashFn,
            joinFn,
            3,
            1 << 3, // 8
            [ op3 ],
            [ DF.variable('w'), DF.variable('v') ],
            [],
            [[ DF.variable('w') ], [ DF.variable('v') ]],
            false,
          ),
          new StemsOperatorStream(
            <BindingsStream> <unknown> new ArrayIterator(data4),
            timestampGen,
            hashFn,
            joinFn,
            4,
            1 << 4, // 16
            [ op4 ],
            [ DF.variable('v'), DF.variable('u') ],
            [],
            [[ DF.variable('v') ]],
            false,
          ),
        ];
      }

      async function collectControllerResults(controller: StemsControllerStream): Promise<RDF.Bindings[]> {
        const results: RDF.Bindings[] = [];
        return new Promise((resolve, reject) => {
          controller.on('data', (binding: RDF.Bindings) => {
            results.push(binding);
          });
          controller.on('end', () => {
            resolve(results);
          });
          controller.on('error', reject);
        });
      }

      let expectedGroundTruth: string[];

      beforeAll(async () => {
        const hashActor = new ActorHashBindingsMurmur({ name: 'actor', bus: new Bus({ name: 'bus' }) });
        const hashResult = await hashActor.run({ context: new ActionContext() });
        hashFn = hashResult.hashFunction;

        // Calculate ground truth by running standard multi-stems with only base operators
        const tsGen = new TimestampGenerator();
        const baseStems = buildBaseStems(dataTP0, dataTP1, dataTP2, dataTP3, dataTP4, tsGen);
        const router = new RouterFixedMinimalIndex();
        const controller = new StemsControllerStream(baseStems, router, 100);
        const results = await collectControllerResults(controller);
        expectedGroundTruth = sortBindings(results);
        expect(expectedGroundTruth).toHaveLength(2);
      });

      it('Case 1: 1 Composite Resource (CR1 covers TP1 and TP2)', async () => {
        // CR1 = TP1 Join TP2: (?y :p2 ?z) and (?z :p3 ?w)
        const cr1Data = [
          BF.bindings([
            [ DF.variable('y'), DF.literal('y1') ],
            [ DF.variable('z'), DF.literal('z1') ],
            [ DF.variable('w'), DF.literal('w1') ],
          ]),
          BF.bindings([
            [ DF.variable('y'), DF.literal('y2') ],
            [ DF.variable('z'), DF.literal('z2') ],
            [ DF.variable('w'), DF.literal('w2') ],
          ]),
        ];

        const tsGen = new TimestampGenerator();
        const stems = buildBaseStems(dataTP0, dataTP1, dataTP2, dataTP3, dataTP4, tsGen);
        const router = new RouterFixedMinimalIndex();
        const controller = new StemsControllerStream(stems, router, 100);

        const cr1Stream = new StemsOperatorStream(
          <BindingsStream> <unknown> new ArrayIterator(cr1Data),
          tsGen,
          hashFn,
          joinFn,
          5, // Operator index 5
          (1 << 1) | (1 << 2), // doneBitMask: 2 | 4 = 6
          [ op1, op2 ],
          [ DF.variable('y'), DF.variable('z'), DF.variable('w') ],
          [],
          [[ DF.variable('y') ], [ DF.variable('w') ]],
          false,
          true, // isCompositeResource
        );

        controller.addOperator(cr1Stream);

        const results = await collectControllerResults(controller);
        const sortedActual = sortBindings(results);
        // Note: Deduplication between Base and CR plans is not yet considered,
        // so multiple exclusive plans may concurrently produce valid results.
        // We verify that the set of results exactly matches expectedGroundTruth.
        expect(Array.from(new Set(sortedActual)).sort()).toEqual(expectedGroundTruth);
      });

      it('Case 2: 2 Composite Resources with Overlap (CR1 = TP1+TP2, CR2 = TP2+TP3)', async () => {
        // CR1 = TP1 + TP2 (mask = 2 | 4 = 6, index 5)
        const cr1Data = [
          BF.bindings([
            [ DF.variable('y'), DF.literal('y1') ],
            [ DF.variable('z'), DF.literal('z1') ],
            [ DF.variable('w'), DF.literal('w1') ],
          ]),
          BF.bindings([
            [ DF.variable('y'), DF.literal('y2') ],
            [ DF.variable('z'), DF.literal('z2') ],
            [ DF.variable('w'), DF.literal('w2') ],
          ]),
        ];
        // CR2 = TP2 + TP3 (mask = 4 | 8 = 12, index 6)
        const cr2Data = [
          BF.bindings([
            [ DF.variable('z'), DF.literal('z1') ],
            [ DF.variable('w'), DF.literal('w1') ],
            [ DF.variable('v'), DF.literal('v1') ],
          ]),
          BF.bindings([
            [ DF.variable('z'), DF.literal('z2') ],
            [ DF.variable('w'), DF.literal('w2') ],
            [ DF.variable('v'), DF.literal('v2') ],
          ]),
        ];

        const tsGen = new TimestampGenerator();
        const stems = buildBaseStems(dataTP0, dataTP1, dataTP2, dataTP3, dataTP4, tsGen);
        const router = new RouterFixedMinimalIndex();
        const controller = new StemsControllerStream(stems, router, 100);

        const cr1Stream = new StemsOperatorStream(
          <BindingsStream> <unknown> new ArrayIterator(cr1Data),
          tsGen,
          hashFn,
          joinFn,
          5,
          (1 << 1) | (1 << 2), // 6
          [ op1, op2 ],
          [ DF.variable('y'), DF.variable('z'), DF.variable('w') ],
          [],
          [[ DF.variable('y') ], [ DF.variable('w') ]],
          false,
          true,
        );

        const cr2Stream = new StemsOperatorStream(
          <BindingsStream> <unknown> new ArrayIterator(cr2Data),
          tsGen,
          hashFn,
          joinFn,
          6,
          (1 << 2) | (1 << 3), // 12
          [ op2, op3 ],
          [ DF.variable('z'), DF.variable('w'), DF.variable('v') ],
          [],
          [[ DF.variable('z') ], [ DF.variable('v') ]],
          false,
          true,
        );

        controller.addOperator(cr1Stream);
        controller.addOperator(cr2Stream);

        const results = await collectControllerResults(controller);
        const sortedActual = sortBindings(results);
        // Note: Deduplication between Base and CR plans is not yet considered,
        // so multiple exclusive plans may concurrently produce valid results.
        expect(Array.from(new Set(sortedActual)).sort()).toEqual(expectedGroundTruth);
      });

      it('Case 3: 2 Composite Resources without Overlap (CR1 = TP1+TP2, CR2 = TP3+TP4, joined with TP0)', async () => {
        // CR1 = TP1 + TP2 (mask = 2 | 4 = 6, index 5)
        const cr1Data = [
          BF.bindings([
            [ DF.variable('y'), DF.literal('y1') ],
            [ DF.variable('z'), DF.literal('z1') ],
            [ DF.variable('w'), DF.literal('w1') ],
          ]),
          BF.bindings([
            [ DF.variable('y'), DF.literal('y2') ],
            [ DF.variable('z'), DF.literal('z2') ],
            [ DF.variable('w'), DF.literal('w2') ],
          ]),
        ];
        // CR2 = TP3 + TP4 (mask = 8 | 16 = 24, index 6)
        const cr2Data = [
          BF.bindings([
            [ DF.variable('w'), DF.literal('w1') ],
            [ DF.variable('v'), DF.literal('v1') ],
            [ DF.variable('u'), DF.literal('u1') ],
          ]),
          BF.bindings([
            [ DF.variable('w'), DF.literal('w2') ],
            [ DF.variable('v'), DF.literal('v2') ],
            [ DF.variable('u'), DF.literal('u2') ],
          ]),
        ];

        const tsGen = new TimestampGenerator();
        const stems = buildBaseStems(dataTP0, dataTP1, dataTP2, dataTP3, dataTP4, tsGen);
        const router = new RouterFixedMinimalIndex();
        const controller = new StemsControllerStream(stems, router, 100);

        const cr1Stream = new StemsOperatorStream(
          <BindingsStream> <unknown> new ArrayIterator(cr1Data),
          tsGen,
          hashFn,
          joinFn,
          5,
          (1 << 1) | (1 << 2), // 6
          [ op1, op2 ],
          [ DF.variable('y'), DF.variable('z'), DF.variable('w') ],
          [],
          [[ DF.variable('y') ], [ DF.variable('w') ]],
          false,
          true,
        );

        const cr2Stream = new StemsOperatorStream(
          <BindingsStream> <unknown> new ArrayIterator(cr2Data),
          tsGen,
          hashFn,
          joinFn,
          6,
          (1 << 3) | (1 << 4), // 24
          [ op3, op4 ],
          [ DF.variable('w'), DF.variable('v'), DF.variable('u') ],
          [],
          [[ DF.variable('w') ], [ DF.variable('u') ]],
          false,
          true,
        );

        controller.addOperator(cr1Stream);
        controller.addOperator(cr2Stream);

        const results = await collectControllerResults(controller);
        const sortedActual = sortBindings(results);
        // Note: Deduplication between Base and CR plans is not yet considered,
        // so multiple exclusive plans may concurrently produce valid results.
        expect(Array.from(new Set(sortedActual)).sort()).toEqual(expectedGroundTruth);
      });

      it('retains and merges sourcesBinding context across STeMs joins', async () => {
        const BFWithSources = new BindingsFactory(DF, {
          [KeysMergeBindingsContext.sourcesBinding.name]: new SetUnionBindingsContextMergeHandler(),
        });

        const data0 = [
          BFWithSources.bindings([[ DF.variable('x'), DF.literal('x1') ], [ DF.variable('y'), DF.literal('y1') ]])
            .setContextEntry(KeysMergeBindingsContext.sourcesBinding, [ 'http://example.org/doc1' ]),
        ];
        const data1 = [
          BFWithSources.bindings([[ DF.variable('y'), DF.literal('y1') ], [ DF.variable('z'), DF.literal('z1') ]])
            .setContextEntry(KeysMergeBindingsContext.sourcesBinding, [ 'http://example.org/doc2' ]),
        ];

        const tsGen = new TimestampGenerator();
        const opStream0 = new StemsOperatorStream(
          <BindingsStream> <unknown> new ArrayIterator(data0),
          tsGen,
          hashFn,
          joinFn,
          0,
          1 << 0,
          [ op0 ],
          [ DF.variable('x'), DF.variable('y') ],
          [],
          [[ DF.variable('y') ]],
          false,
        );
        const opStream1 = new StemsOperatorStream(
          <BindingsStream> <unknown> new ArrayIterator(data1),
          tsGen,
          hashFn,
          joinFn,
          1,
          1 << 1,
          [ op1 ],
          [ DF.variable('y'), DF.variable('z') ],
          [],
          [[ DF.variable('y') ]],
          false,
        );

        const router = new RouterFixedMinimalIndex();
        const controller = new StemsControllerStream([ opStream0, opStream1 ], router, 100);
        const results = await collectControllerResults(controller);

        expect(results).toHaveLength(1);
        const finalSources = (<Bindings> results[0]).getContextEntry(KeysMergeBindingsContext.sourcesBinding);
        expect(finalSources).toEqual([ 'http://example.org/doc1', 'http://example.org/doc2' ]);
      });
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
