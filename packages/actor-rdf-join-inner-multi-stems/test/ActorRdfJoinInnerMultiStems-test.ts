import { Bus } from '@comunica/core';
import { ActorRdfJoinMultiStems } from '../lib/ActorRdfJoinMultiStems';
import { DataFactory } from 'rdf-data-factory';
import { BindingsFactory } from '@comunica/utils-bindings-factory';
import { ArrayIterator } from 'asynciterator';
import { MetadataValidationState } from '@comunica/utils-metadata';
import type * as RDF from '@rdfjs/types';
import { IJoinEntry, IJoinEntryWithMetadata, MetadataVariable, QueryResultCardinality } from '@comunica/types';

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
          ['a', 'b'],
          ['d', 'c'],
          ['c', 'f'],
          ['f', 'b'],
          ['g', 'h'],
          ['h', 'i'],
          ['i', 'j']
        ]
      )
      const disjointEntries = ActorRdfJoinMultiStems.findConnectedComponentsInJoinGraph(entries);
      expect(disjointEntries.entries.length).toEqual(2);
      expect(disjointEntries.entries).toEqual([entries.slice(0, 4),entries.slice(4, 7)]);
      expect(disjointEntries.indexes).toEqual([[0,1,2,3],[4,5,6]]);
    });

    it('should return each entry as a separate group when all are disjoint', () => {
      const entries = createEntriesWithDifferentJoinVars([
        ['a', 'b'],
        ['c', 'd'],
        ['e', 'f'],
        ['g', 'h']
      ]);
      const disjointEntries = ActorRdfJoinMultiStems.findConnectedComponentsInJoinGraph(entries);
      expect(disjointEntries.entries.length).toBe(4);
      disjointEntries.entries.forEach((group, i) => {
        expect(group).toEqual([entries[i]]);
      });
      disjointEntries.indexes.forEach((group, i) => {
        expect(group).toEqual([i]);
      });
    });

    it('should return a single group when all entries are connected transitively', () => {
      const entries = createEntriesWithDifferentJoinVars([
        ['a', 'b'],
        ['b', 'c'],
        ['c', 'd'],
        ['d', 'e']
      ]);
      const disjointEntries = ActorRdfJoinMultiStems.findConnectedComponentsInJoinGraph(entries);
      expect(disjointEntries.entries.length).toBe(1);
      expect(disjointEntries.entries).toEqual([entries]);
    });

    it('should treat entries with one variable as disjoint if no overlaps', () => {
      const entries = createEntriesWithDifferentJoinVars([
        ['a'],
        ['b'],
        ['c']
      ]);
      const disjointEntries = ActorRdfJoinMultiStems.findConnectedComponentsInJoinGraph(entries);
      expect(disjointEntries.entries.length).toBe(3);
      expect(disjointEntries.entries).toEqual([[entries[0]], [entries[1]], [entries[2]]]);
      expect(disjointEntries.indexes).toEqual([[0],[1],[2]]);
    });

    it('should group all entries connected via a common variable (star shape)', () => {
      const entries = createEntriesWithDifferentJoinVars([
        ['x', 'a'],
        ['x', 'b'],
        ['x', 'c'],
        ['x', 'd']
      ]);
      const disjointEntries = ActorRdfJoinMultiStems.findConnectedComponentsInJoinGraph(entries);
      expect(disjointEntries.entries.length).toBe(1);
      expect(disjointEntries.entries).toEqual([entries]);
      expect(disjointEntries.indexes).toEqual([[0,1,2,3]]);
    });

    it('should find multiple disjoint groups in a complex graph with cycles', () => {
      const entries = createEntriesWithDifferentJoinVars([
        ['a', 'b'], // 0
        ['b', 'c'], // 1
        ['c', 'a'], // 2 (forms cycle with 0,1)
        ['x', 'y'], // 3
        ['y', 'z'], // 4
        ['p', 'q'], // 5
        ['q', 'r'], // 6
        ['s', 't'], // 7
      ]);
      const disjointEntries = ActorRdfJoinMultiStems.findConnectedComponentsInJoinGraph(entries);
      expect(disjointEntries.entries.length).toBe(4);
      expect(disjointEntries.entries).toEqual(
        [entries.slice(0,3),entries.slice(3,5),entries.slice(5,7),entries.slice(7)]
      );
      expect(disjointEntries.indexes).toEqual([[0,1,2],[3,4], [5,6], [7]]);
    });
  });
});


function createEntriesWithDifferentJoinVars(variableValues: string[][]): IJoinEntryWithMetadata[]{
  const entriesWithVariablesSet = variableValues.map((values) => {
    const variables: MetadataVariable[] = values.map(value => {return { variable: DF.variable(value), canBeUndef: false}});
    return {
      output: {
        bindingsStream: new ArrayIterator<RDF.Bindings>([]),
        metadata: () => Promise.resolve({
          state: new MetadataValidationState(),
          cardinality: <QueryResultCardinality> { type: 'estimate', value: 4 },
          pageSize: 100,
          requestTime: 10,
          variables: variables,
        }),
        type: <any> 'bindings'
      },
      operation: <any> {},
      metadata: {
        state: new MetadataValidationState(),
        cardinality: <QueryResultCardinality> { type: 'estimate', value: 4 },
        pageSize: 100,
        requestTime: 10,
        variables: variables,
      }
    };
  });

  return entriesWithVariablesSet;
}