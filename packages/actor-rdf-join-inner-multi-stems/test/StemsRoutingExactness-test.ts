import { ActorHashBindingsMurmur } from '@comunica/actor-hash-bindings-murmur';
import { ActionContext, Bus } from '@comunica/core';
import type { BindingsStream } from '@comunica/types';
import { BindingsFactory } from '@comunica/utils-bindings-factory';
import type * as RDF from '@rdfjs/types';
import { ArrayIterator } from 'asynciterator';
import { DataFactory } from 'rdf-data-factory';
import { ActorRdfJoinMultiStems } from '../lib/ActorRdfJoinMultiStems';
import type { IStemsRouter } from '../lib/routers/BaseRouter';
import { RouterFixedMinimalIndex } from '../lib/routers/FixedRouter';
import { RouterLotteryScheduling } from '../lib/routers/LotteryRouter';
import { RouterLotterySchedulingSignature } from '../lib/routers/LotterySignatureRouter';
import { StemsControllerStream, TimestampGenerator } from '../lib/StemsControllerStream';
import { StemsOperatorStream, type HashFunction, type JoinFunction } from '../lib/StemsOperatorStream';
import { computePairwiseJoinVariables } from '../lib/utils/JoinVariables';

const DF = new DataFactory();
const BF = new BindingsFactory(DF);

type Row = Record<string, string>;

/**
 * Exactness of routing with composite resources: every combination of pairwise disjoint
 * composites must contribute its join exactly once, so the produced multiset of results has to
 * equal the sum over those combinations. Composite tuples bind their internal variables to values
 * no base tuple uses, so no composite tuple is pruned as already covered and every combination
 * produces distinguishable results.
 */
describe('SteMs routing with composite resources', () => {
  let hashFn: HashFunction;
  const joinFn = <JoinFunction> ActorRdfJoinMultiStems.joinBindings;

  beforeAll(async() => {
    const hashActor = new ActorHashBindingsMurmur({ name: 'actor', bus: new Bus({ name: 'bus' }) });
    hashFn = (await hashActor.run({ context: new ActionContext() })).hashFunction;
  });

  // Deterministic PRNG, so a failing case can be reproduced from its seed
  function mulberry32(seed: number): () => number {
    let state = seed;
    return () => {
      // Wraps to a 32-bit integer, which Math.trunc does not
      // eslint-disable-next-line unicorn/prefer-math-trunc
      state = (state + 0x6D2B79F5) | 0;
      let t = Math.imul(state ^ (state >>> 15), 1 | state);
      t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  interface ICase {
    nPatterns: number;
    base: Row[][];
    composites: { from: number; to: number; rows: Row[] }[];
  }

  // Chain query t_i = (?v_i, p_i, ?v_{i+1}); a composite answers the patterns from..to
  function generateCase(seed: number): ICase {
    const random = mulberry32(seed);
    const pick = (n: number): number => Math.floor(random() * n);
    const nPatterns = 3 + pick(3);
    const value = (): string => `${pick(2)}`;
    const base: Row[][] = [];
    for (let i = 0; i < nPatterns; i++) {
      const rows: Row[] = [];
      for (let k = pick(3); k < 3; k++) {
        rows.push({ [`v${i}`]: value(), [`v${i + 1}`]: value() });
      }
      base.push(rows);
    }
    const composites: ICase['composites'] = [];
    const nComposites = 1 + pick(3);
    for (let c = 0; c < nComposites; c++) {
      const from = pick(nPatterns - 1);
      const to = from + 1 + pick(Math.min(2, nPatterns - from - 1));
      const rows: Row[] = [];
      for (let k = pick(2); k < 3; k++) {
        const row: Row = { [`v${from}`]: value(), [`v${to + 1}`]: value() };
        for (let v = from + 1; v <= to; v++) {
          row[`v${v}`] = `c${c}_${pick(2)}`;
        }
        rows.push(row);
      }
      composites.push({ from, to, rows });
    }
    return { nPatterns, base, composites };
  }

  function join(relations: Row[][]): Row[] {
    let result: Row[] = [{}];
    for (const relation of relations) {
      const next: Row[] = [];
      for (const left of result) {
        for (const right of relation) {
          if (Object.keys(right).every(key => left[key] === undefined || left[key] === right[key])) {
            next.push({ ...left, ...right });
          }
        }
      }
      result = next;
    }
    return result;
  }

  const serialize = (row: Row): string => Object.keys(row).sort().map(key => `${key}=${row[key]}`).join(';');

  // Sum over every set of pairwise disjoint composites of its join with the remaining patterns
  function expectedResults(testCase: ICase): string[] {
    const results: string[] = [];
    for (let subset = 0; subset < 1 << testCase.composites.length; subset++) {
      const chosen = testCase.composites.filter((_, c) => (subset & (1 << c)) !== 0);
      const covered = new Set<number>();
      let disjoint = true;
      for (const composite of chosen) {
        for (let i = composite.from; i <= composite.to; i++) {
          disjoint &&= !covered.has(i);
          covered.add(i);
        }
      }
      if (!disjoint) {
        continue;
      }
      const relations = [
        ...chosen.map(composite => composite.rows),
        ...testCase.base.filter((_, i) => !covered.has(i)),
      ];
      results.push(...join(relations).map(serialize));
    }
    return results.sort();
  }

  async function actualResults(testCase: ICase, router: IStemsRouter): Promise<string[]> {
    const variable = (i: number): RDF.Variable => DF.variable(`v${i}`);
    const pattern = (i: number): any => ({
      type: 'pattern',
      subject: variable(i),
      predicate: DF.namedNode(`http://example.org/p${i}`),
      object: variable(i + 1),
      graph: DF.defaultGraph(),
    });
    const toBindings = (rows: Row[]): RDF.Bindings[] => rows.map(row =>
      BF.bindings(Object.entries(row).map(([ key, val ]) => [ DF.variable(key), DF.literal(val) ])));

    const baseVariables = testCase.base.map((_, i) => [ variable(i), variable(i + 1) ]);
    const compositeVariables = testCase.composites.map(composite =>
      Array.from({ length: composite.to - composite.from + 2 }, (_, k) => variable(composite.from + k)));
    const joinVariables = computePairwiseJoinVariables([ ...baseVariables, ...compositeVariables ]);

    const timestamps = new TimestampGenerator();
    const operators = testCase.base.map((rows, i) => new StemsOperatorStream(
      <BindingsStream> <unknown> new ArrayIterator(toBindings(rows), { autoStart: false }),
      timestamps,
      hashFn,
      joinFn,
      i,
      1 << i,
      [ pattern(i) ],
      baseVariables[i],
      [],
      joinVariables[i],
      false,
    ));
    const controller = new StemsControllerStream(operators, router, 1);
    for (const [ c, composite ] of testCase.composites.entries()) {
      const covered = Array.from({ length: composite.to - composite.from + 1 }, (_, k) => composite.from + k);
      controller.addOperator(new StemsOperatorStream(
        <BindingsStream> <unknown> new ArrayIterator(toBindings(composite.rows), { autoStart: false }),
        timestamps,
        hashFn,
        joinFn,
        testCase.nPatterns + c,
        covered.reduce((mask, i) => mask | (1 << i), 0),
        covered.map(pattern),
        compositeVariables[c],
        [],
        joinVariables[testCase.nPatterns + c],
        false,
      ));
    }

    const results: string[] = [];
    await new Promise<void>((resolve, reject) => {
      controller.on('data', (binding: RDF.Bindings) => {
        const row: Row = {};
        for (const [ key, val ] of binding) {
          row[key.value] = val.value;
        }
        results.push(serialize(row));
      });
      controller.on('end', resolve);
      controller.on('error', reject);
    });
    return results.sort();
  }

  const routers: [string, () => IStemsRouter][] = [
    [ 'fixed', () => new RouterFixedMinimalIndex() ],
    [ 'lottery', () => new RouterLotteryScheduling() ],
    [ 'lottery-signature', () => new RouterLotterySchedulingSignature() ],
  ];
  const seeds = Array.from({ length: 150 }, (_, i) => i + 1);

  it('computes the plain join without composites', async() => {
    for (const seed of seeds) {
      const testCase = { ...generateCase(seed), composites: []};
      await expect(actualResults(testCase, new RouterFixedMinimalIndex())).resolves.toEqual(expectedResults(testCase));
    }
  });

  it.each(routers)('produces every combination of composites exactly once (%s router)', async(_, createRouter) => {
    const failures: string[] = [];
    for (const seed of seeds) {
      const testCase = generateCase(seed);
      const expected = expectedResults(testCase);
      const actual = await actualResults(testCase, createRouter());
      if (actual.join('\n') !== expected.join('\n')) {
        failures.push(`seed ${seed}: expected ${expected.length}, got ${actual.length}`);
      }
    }
    expect(failures).toEqual([]);
  });

  it('does not repeat a composite on a copy that already passed it by', async() => {
    // Chain t0..t4 where composite c answers t2 and t3. A t1 tuple splits into a base copy going to
    // t0 and a copy joining c; the base copy must not join c later as well
    const testCase: ICase = {
      nPatterns: 5,
      base: [
        [{ v0: 'x1', v1: 'y1' }, { v0: 'x2', v1: 'y2' }],
        [{ v1: 'y1', v2: 'z1' }, { v1: 'y2', v2: 'z2' }],
        [{ v2: 'z1', v3: 'w1' }],
        [{ v3: 'w1', v4: 'u1' }],
        [{ v4: 'u1', v5: 'r1' }, { v4: 'u2', v5: 'r2' }],
      ],
      composites: [{ from: 2, to: 3, rows: [{ v2: 'z2', v3: 'c0_w2', v4: 'u2' }]}],
    };
    const actual = await actualResults(testCase, new RouterFixedMinimalIndex());
    expect(actual).toEqual(expectedResults(testCase));
    expect(actual).toHaveLength(2);
  });
});
