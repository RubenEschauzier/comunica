import type { Operation } from 'sparqlalgebrajs/lib/algebra';
import type { ISampleResult } from './IndexBasedJoinSampler';
import { JoinTree } from './JoinTree';

export class JoinOrderEnumerator {
  private readonly adjacencyList: Map<number, number[]>;
  private readonly estimates: Map<string, ISampleResult>;
  private readonly entries: Operation[];

  public constructor(
    adjacencyList: Map<number, number[]>,
    estimates: Map<string, ISampleResult>,
    entries: Operation[],
  ) {
    this.adjacencyList = adjacencyList;
    this.estimates = estimates;
    this.entries = entries;
  }

  public search(): JoinTree {
    const bestPlan: Map<string, JoinTree> = new Map();
    for (let i = 0; i < this.entries.length; i++) {
      const singleton = new Set([ i ]);
      const singletonKey = JSON.stringify(JoinOrderEnumerator.sortArrayAsc([ ...singleton ]));
      bestPlan.set(singletonKey, new JoinTree(undefined, undefined, singleton, this.estimates.get(JSON.stringify([ i ]))!.estimatedCardinality));
    }

    const csgCmpPairs = this.enumerateCsgCmpPairs(this.entries.length);
    for (const csgCmpPair of csgCmpPairs) {
      const tree1 = bestPlan.get(JSON.stringify(JoinOrderEnumerator.sortArrayAsc([ ...csgCmpPair[0] ])))!;
      const tree2 = bestPlan.get(JSON.stringify(JoinOrderEnumerator.sortArrayAsc([ ...csgCmpPair[1] ])))!;
      const newEntries = new Set([ ...tree1.entries, ...tree2.entries ]);

      const estimatedSize = this.estimates.get(JSON.stringify(JoinOrderEnumerator.sortArrayAsc([ ...newEntries ])))!.estimatedCardinality;
      // The order in which joins happen matters for the cost, so evaluate both
      const currPlanLeft = new JoinTree(tree1, tree2, newEntries, estimatedSize);
      const currPlanRight = new JoinTree(tree2, tree1, newEntries, estimatedSize);
      const currPlan = currPlanLeft.cost > currPlanRight.cost ? currPlanRight : currPlanLeft;

      const currPlanKey = JSON.stringify(JoinOrderEnumerator.sortArrayAsc([ ...currPlan.entries ]));

      if (!bestPlan.get(currPlanKey) || bestPlan.get(currPlanKey)!.cost > currPlan.cost) {
        bestPlan.set(currPlanKey, currPlan);
      }
    }
    const allEntriesKey = JSON.stringify([ ...Array.from({ length: this.entries.length }).keys() ]);
    return bestPlan.get(allEntriesKey)!;
  }

  public enumerateCsgCmpPairs(nTps: number) {
    const allTps = new Set(new Array(nTps).keys());
    const csgs = this.enumerateCsg(nTps);
    const csgCmpPairs: [Set<number>, Set<number>][] = [];

    for (const csg of csgs) {
      const cmps = this.enumerateCmp(csg, allTps);
      for (const cmp of cmps) {
        csgCmpPairs.push([ csg, cmp ]);
      }
    }
    return csgCmpPairs;
  }

  public enumerateCsg(nTps: number) {
    const csgs: Set<number>[] = [];
    for (let i = nTps - 1; i >= 0; i--) {
      const vI = new Set([ i ]);
      csgs.push(vI);
      this.enumerateCsgRecursive(csgs, vI, new Set(new Array(i + 1).keys()));
    }
    return csgs;
  }

  public enumerateCmp(tpsSubset: Set<number>, allTps: Set<number>) {
    const cmps: Set<number>[] = [];
    const X = new Set([ ...this.reduceSet(this.setMinimum(tpsSubset), allTps), ...tpsSubset ]);
    const neighbours = this.getNeighbours(tpsSubset);
    for (const vertex of X) {
      neighbours.delete(vertex);
    }

    for (const vertex of JoinOrderEnumerator.sortSetDesc(neighbours)) {
      cmps.push(new Set([ vertex ]));
      this.enumerateCsgRecursive(
        cmps,
        new Set([ vertex ]),
        new Set([ ...X, ...this.reduceSet(vertex, neighbours) ]),
      );
    }
    return cmps;
  }

  public enumerateCsgRecursive(csgs: Set<number>[], S: Set<number>, X: Set<number>) {
    // Difference of neighbours and X
    const neighbours = this.getNeighbours(S);
    for (const vertex of X) {
      neighbours.delete(vertex);
    }

    const subsets = this.getAllSubsets(neighbours);
    for (const subset of subsets) {
      // Add union of two sets as connected subgraph
      csgs.push(new Set([ ...S, ...subset ]));
    }
    for (const subset of subsets) {
      this.enumerateCsgRecursive(csgs, new Set([ ...S, ...subset ]), new Set([ ...X, ...neighbours ]));
    }
  }

  public getNeighbours(S: Set<number>) {
    const neighbours: Set<number> = new Set();
    S.forEach((vertex: number) => {
      this.adjacencyList.get(vertex)!.forEach((neighbour: number) => {
        neighbours.add(neighbour);
      });
    });
    return neighbours;
  }

  public getAllSubsets(set: Set<number>): Set<number>[] {
    const subsets: Set<number>[] = [ new Set([]) ];
    for (const el of set) {
      const last = subsets.length - 1;
      for (let i = 0; i <= last; i++) {
        subsets.push(new Set([ ...subsets[i], el ]));
      }
    }
    // Remove empty set
    return subsets.slice(1);
  }

  public reduceSet(i: number, set: Set<number>) {
    const reducedSet: Set<number> = new Set();
    for (const vertex of set) {
      if (vertex <= i) {
        reducedSet.add(vertex);
      }
    }
    return reducedSet;
  }

  public setMinimum(set: Set<number>): number {
    return Math.min(...set);
  }

  public static sortSetDesc(set: Set<number>): number[] {
    return [ ...set ].sort((a, b) => b - a);
  }

  public static sortArrayAsc(arr: number[]) {
    return [ ...arr ].sort((n1, n2) => n1 - n2);
  }
}
