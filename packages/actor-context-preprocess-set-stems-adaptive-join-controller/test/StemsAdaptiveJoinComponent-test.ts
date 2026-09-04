import type { IJoinEntryWithMetadata, MetadataVariable, QueryResultCardinality } from '@comunica/types';
import { MetadataValidationState } from '@comunica/utils-metadata';
import type * as RDF from '@rdfjs/types';
import { ArrayIterator } from 'asynciterator';
import { DataFactory } from 'rdf-data-factory';
import { StemsAdaptiveJoinComponent } from '../lib/StemsAdaptiveJoinComponent';

const DF = new DataFactory();

/**
 * Builds join entries whose (synchronous) metadata exposes the given variables - the same shape
 * StemsAdaptiveJoinComponent#computeJoinVariablesForSubset reads via `entry.metadata.variables`.
 */
function createEntries(variableValues: string[][]): IJoinEntryWithMetadata[] {
  return variableValues.map((values) => {
    const variables: MetadataVariable[] = values.map(value => ({
      variable: DF.variable(value),
      canBeUndef: false,
    }));
    const metadata = {
      state: new MetadataValidationState(),
      cardinality: <QueryResultCardinality> { type: 'estimate', value: 4 },
      pageSize: 100,
      requestTime: 10,
      variables,
    };
    return <IJoinEntryWithMetadata> <any> {
      output: {
        bindingsStream: new ArrayIterator<RDF.Bindings>([]),
        metadata: () => Promise.resolve(metadata),
        type: 'bindings',
      },
      operation: { type: 'pattern' },
      metadata,
    };
  });
}

/**
 * Runs the real, current computeJoinVariablesForSubset: the keys a composite resource covering
 * `covered` will hash its tuples under.
 */
function joinVariablesFor(variableValues: string[][], covered: number[]): string[][] {
  const joinEntries = createEntries(variableValues);
  const component = new StemsAdaptiveJoinComponent(<any> {
    id: 'test',
    joinEntries,
    stemsControllerStream: undefined,
    router: undefined,
    timestampGenerator: undefined,
    hashFn: undefined,
    joinFn: undefined,
    dataFactory: DF,
  });

  return (<any> component).computeJoinVariablesForSubset(covered)
    .map((variables: RDF.Variable[]) => variables.map(variable => variable.value));
}

describe('StemsAdaptiveJoinComponent', () => {
  describe('computeJoinVariablesForSubset', () => {
    // The composite resource's join keys must be exactly its own variables (the union of the
    // entries it covers) intersected with each entry it does not cover - the same pairwise
    // intersection ActorRdfJoinMultiStems#getJoinVariables computes for base operators.

    it('handles a star with the composite resource covering one pattern', () => {
      // ?s ex:n ?n . ?s ex:k ?k . ?s ex:m ?m -- every entry joins on ?s only.
      // Both uncovered entries intersect the composite resource on the same variable, so they
      // collapse into a single key rather than being repeated per uncovered entry.
      expect(joinVariablesFor([[ 's', 'n' ], [ 's', 'k' ], [ 's', 'm' ]], [ 0 ]))
        .toEqual([[ 's' ]]);
    });

    it('handles a star with the composite resource covering two patterns', () => {
      expect(joinVariablesFor([[ 's', 'n' ], [ 's', 'k' ], [ 's', 'm' ]], [ 0, 1 ]))
        .toEqual([[ 's' ]]);
    });

    it('handles a 3-chain with the composite resource covering the head', () => {
      // ?a ex:p ?b . ?b ex:p ?c . ?c ex:p ?d, composite resource covers only the first pattern,
      // so it binds ?a and ?b. It does not bind ?c, so it must not hash on it even though the
      // two uncovered entries join each other on ?c.
      expect(joinVariablesFor([[ 'a', 'b' ], [ 'b', 'c' ], [ 'c', 'd' ]], [ 0 ]))
        .toEqual([[ 'b' ]]);
    });

    it('handles a 3-chain with the composite resource covering the first two patterns', () => {
      // With only one uncovered entry left, every other entry is inside the composite resource
      expect(joinVariablesFor([[ 'a', 'b' ], [ 'b', 'c' ], [ 'c', 'd' ]], [ 0, 1 ]))
        .toEqual([[ 'c' ]]);
    });

    it('handles a 4-chain with the composite resource covering the first two patterns', () => {
      // ?a ex:p ?b . ?b ex:p ?c . ?c ex:p ?d . ?d ex:p ?e, composite resource binds ?a, ?b, ?c.
      // It must not hash on ?d, even though the two uncovered entries join each other on it.
      expect(joinVariablesFor([[ 'a', 'b' ], [ 'b', 'c' ], [ 'c', 'd' ], [ 'd', 'e' ]], [ 0, 1 ]))
        .toEqual([[ 'c' ]]);
    });

    it('handles a cycle with two overlapping variables', () => {
      // ?a ex:p ?b . ?b ex:p ?c . ?c ex:p ?d . ?d ex:p ?e, composite resource binds ?a, ?b, ?c.
      // It must not hash on ?d, even though the two uncovered entries join each other on it.
      expect(joinVariablesFor([[ 'a', 'b' ], [ 'b', 'c' ], [ 'c', 'a' ],  ['c', 'e' ]], [ 0, 1 ]))
        .toEqual([[ 'a' , 'c'], [ 'c' ]]);
    });

    it('omits an uncovered entry that shares nothing with the composite resource', () => {
      // ?a ex:p ?b . ?c ex:p ?d, disjoint from the composite resource covering the first pattern
      expect(joinVariablesFor([[ 'a', 'b' ], [ 'c', 'd' ]], [ 0 ]))
        .toEqual([]);
    });
  });
});
