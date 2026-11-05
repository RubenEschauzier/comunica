import { BindingsFactory } from '@comunica/utils-bindings-factory';
import type { Bindings } from '@comunica/utils-bindings-factory';
import type * as RDF from '@rdfjs/types';
import { DataFactory } from 'rdf-data-factory';
import { DelegatedPatternsFilter } from '../lib/filters/DelegatedPatternsFilter';
import { DelegatedPatternsFilterIndex } from '../lib/filters/DelegatedPatternsFilterIndex';
import { bitForIndex } from '../lib/utils/BitUtils';

const DF = new DataFactory();
const BF = new BindingsFactory(DF);

// Source documents per mapping, standing in for the sources context entry
const sources = new WeakMap<Bindings, string[]>();
const sourceExtractor = (binding: Bindings): string[] => sources.get(binding) ?? [];

function mapping(entries: Record<string, RDF.Term>, documents: string[] = []): Bindings {
  const binding = BF.bindings(Object.entries(entries).map(([ name, term ]) => [ DF.variable(name), term ]));
  sources.set(binding, documents);
  return binding;
}

function filter(blockMask: number, owner: number, anchors: string[], domain: string): DelegatedPatternsFilter {
  return new DelegatedPatternsFilter(blockMask, owner, anchors, sourceExtractor, domain);
}

function linear(filters: DelegatedPatternsFilter[], binding: Bindings, doneMask: number, crMask: number): boolean {
  return filters.some(f => f.shouldFilter(binding, doneMask, crMask));
}

function indexOf(filters: DelegatedPatternsFilter[]): DelegatedPatternsFilterIndex {
  const index = new DelegatedPatternsFilterIndex();
  for (const f of filters) {
    index.add(f);
  }
  return index;
}

// Deterministic PRNG, so a failing case can be reproduced from its seed
function mulberry32(seed: number): () => number {
  let state = seed;
  return () => {
    state = state + 0x6D2B79F5 | 0;
    let t = Math.imul(state ^ state >>> 15, 1 | state);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4_294_967_296;
  };
}

const POD = (i: number): string => `http://ex.org/pods/${String(i).padStart(6, '0')}/`;

describe('DelegatedPatternsFilterIndex', () => {
  const block = bitForIndex(0) | bitForIndex(1);
  const owner = 5;

  it('is empty when nothing was added', () => {
    const index = new DelegatedPatternsFilterIndex();
    expect(index.size).toBe(0);
    expect(index.shouldFilter(mapping({ s: DF.namedNode(`${POD(1)}card#me`) }), bitForIndex(0), 0)).toBe(false);
  });

  it('counts every filter added, also for a domain added before', () => {
    const index = indexOf([ filter(block, owner, [ 's' ], POD(1)), filter(block, owner, [ 's' ], POD(1)) ]);
    expect(index.size).toBe(2);
  });

  it('scans a group with a single filter', () => {
    const index = indexOf([ filter(block, owner, [ 's' ], POD(1)) ]);
    expect(index.shouldFilter(mapping({ s: DF.namedNode(`${POD(1)}card#me`) }), bitForIndex(0), 0)).toBe(true);
    expect(index.shouldFilter(mapping({ s: DF.namedNode(`${POD(2)}card#me`) }), bitForIndex(0), 0)).toBe(false);
  });

  it('looks up by domain once a group has several filters', () => {
    const index = indexOf([ 1, 2, 3 ].map(i => filter(block, owner, [ 's' ], POD(i))));
    expect(index.shouldFilter(mapping({ s: DF.namedNode(`${POD(2)}card#me`) }), bitForIndex(0), 0)).toBe(true);
    expect(index.shouldFilter(mapping({ s: DF.namedNode(`${POD(4)}card#me`) }), bitForIndex(0), 0)).toBe(false);
  });

  it('does not filter mappings of the owning composite or outside the block', () => {
    const index = indexOf([ 1, 2 ].map(i => filter(block, owner, [ 's' ], POD(i))));
    const inside = mapping({ s: DF.namedNode(`${POD(1)}card#me`) });
    expect(index.shouldFilter(inside, bitForIndex(0), bitForIndex(owner))).toBe(false);
    expect(index.shouldFilter(inside, bitForIndex(3), 0)).toBe(false);
  });

  it('does not filter a mapping with an unbound anchor', () => {
    const index = indexOf([ 1, 2 ].map(i => filter(block, owner, [ 's', 'o' ], POD(i))));
    expect(index.shouldFilter(mapping({ s: DF.namedNode(`${POD(1)}card#me`) }), bitForIndex(0), 0)).toBe(false);
  });

  it('decides blank node anchors by the documents of the mapping', () => {
    const index = indexOf([ 1, 2 ].map(i => filter(block, owner, [ 'k' ], POD(i))));
    expect(index.shouldFilter(mapping({ k: DF.blankNode('b1') }, [ `${POD(2)}card` ]), bitForIndex(0), 0)).toBe(true);
    expect(index.shouldFilter(mapping({ k: DF.blankNode('b1') }, [ `${POD(2)}card`, `${POD(3)}x` ]), bitForIndex(0), 0))
      .toBe(false);
    expect(index.shouldFilter(mapping({ k: DF.blankNode('b1') }, []), bitForIndex(0), 0)).toBe(false);
  });

  it('filters everything the masks let through for a block without variable anchors', () => {
    const index = indexOf([ 1, 2 ].map(i => filter(block, owner, [], POD(i))));
    expect(index.shouldFilter(mapping({ s: DF.namedNode('http://elsewhere.org/x') }), bitForIndex(0), 0)).toBe(true);
  });

  it('respects the boundary of a domain that does not end on a separator', () => {
    const index = indexOf([ 'http://ex.org/a', 'http://ex.org/b' ].map(d => filter(block, owner, [ 's' ], d)));
    const decide = (iri: string): boolean => index.shouldFilter(mapping({ s: DF.namedNode(iri) }), bitForIndex(0), 0);
    expect(decide('http://ex.org/a')).toBe(true);
    expect(decide('http://ex.org/a/x')).toBe(true);
    expect(decide('http://ex.org/a#x')).toBe(true);
    expect(decide('http://ex.org/a?x')).toBe(true);
    expect(decide('http://ex.org/ab')).toBe(false);
  });

  describe('decides like a linear scan over the same filters', () => {
    // Domain shapes: pod roots of one length, roots of a few lengths, of many lengths (which makes the
    // index look up by separator), domains without a trailing separator, and domains ending on a hash
    const domainShapes: Record<string, (i: number) => string> = {
      'same length': POD,
      'few lengths': i => `http://ex.org/pods/${'x'.repeat(i % 3)}${i}/`,
      'many lengths': i => `http://ex.org/pods/${'x'.repeat(i % 9)}${i}/`,
      undelimited: i => `http://ex.org/pods/${i}`,
      'hash delimited': i => `http://ex.org/pods/${i}/doc#`,
    };
    const valuesFor = (shape: (i: number) => string, i: number): string[] => {
      const domain = shape(i);
      return [
        domain,
        `${domain}card#me`,
        `${domain.replace(/[/#]$/u, '')}#me`,
        `${domain.replace(/[/#]$/u, '')}?q=1`,
        `${domain.replace(/[/#]$/u, '')}9/card`,
        'http://elsewhere.org/x',
      ];
    };

    for (const [ shapeName, shape ] of Object.entries(domainShapes)) {
      for (const filterCount of [ 1, 2, 5, 40 ]) {
        it(`${shapeName}, ${filterCount} filter(s) per group`, () => {
          const rng = mulberry32(filterCount * 7 + shapeName.length);
          const pick = <T>(items: T[]): T => items[Math.floor(rng() * items.length)];

          // Two groups on the operator: a ?s anchored block and a ?s ?k anchored one of another composite
          const filters: DelegatedPatternsFilter[] = [];
          for (let i = 0; i < filterCount; i++) {
            filters.push(filter(block, owner, [ 's' ], shape(i)));
            filters.push(filter(bitForIndex(0) | bitForIndex(2), owner + 1, [ 'k', 's' ], shape(i * 2)));
          }
          // A duplicate domain must not change anything
          filters.push(filter(block, owner, [ 's' ], shape(0)));
          const index = indexOf(filters);

          const decisions = { filtered: 0, kept: 0 };
          for (let n = 0; n < 2_000; n++) {
            const s = rng() < 0.15 ?
              DF.blankNode(`b${n}`) :
              DF.namedNode(pick(valuesFor(shape, Math.floor(rng() * filterCount * 2.5))));
            const k = rng() < 0.5 ? DF.blankNode(`k${n}`) : DF.namedNode(pick(valuesFor(shape, Math.floor(rng() * filterCount * 2.5))));
            const entries: Record<string, RDF.Term> = rng() < 0.1 ? { s } : { s, k };
            const documents = rng() < 0.1 ? [] : [ pick(valuesFor(shape, Math.floor(rng() * filterCount * 2.5))) ];
            if (rng() < 0.3) {
              documents.push(pick(valuesFor(shape, Math.floor(rng() * filterCount * 2.5))));
            }
            const binding = mapping(entries, documents);
            const doneMask = pick([ bitForIndex(0), bitForIndex(2), bitForIndex(3) ]);
            const crMask = pick([ 0, 0, bitForIndex(owner), bitForIndex(owner + 1) ]);
            const expected = linear(filters, binding, doneMask, crMask);
            expect(index.shouldFilter(binding, doneMask, crMask)).toBe(expected);
            decisions[expected ? 'filtered' : 'kept']++;
          }
          // Both outcomes have to occur, or the comparison says little
          expect(decisions.filtered).toBeGreaterThan(50);
          expect(decisions.kept).toBeGreaterThan(50);
        });
      }
    }
  });
});
