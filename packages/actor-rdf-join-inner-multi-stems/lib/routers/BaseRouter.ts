import type { Bindings } from '@comunica/utils-bindings-factory';

import type * as RDF from '@rdfjs/types';
import { eddiesContextKeys } from '../EddieControllerStream';
import type { EddieOperatorStream } from '../EddieOperatorStream';

export abstract class RouterBase implements IEddieRouter {
  /**
   * Connected components of join graph. Represented by their bitmask index
   */
  protected connectedComponents: number[][];

  public createRouteTable(variables: RDF.Variable[][], namedNodes: RDF.NamedNode[][]): Record<number, IEddieRoutingEntry[]> {
    const n = variables.length;
    const routeTable: Record<number, IEddieRoutingEntry[]> = {};
    const variableValues = variables.map(vArr => vArr.map(x => x.value));
    const namedNodeValues = namedNodes.map(nArr => nArr.map(x => x.value));

    const totalStates = 1 << n;
    for (let state = 1; state < totalStates; state++) {
      // Build doneVector array from bits of state
      const doneIndexes = this.getSetBitIndexes(state);

      // // If all done, there is no next entry
      if (doneIndexes.length === n) {
        continue;
      }

      const doneVars = new Set(doneIndexes.flatMap(i => variableValues[i]));
      const doneNamedNodes = new Set(doneIndexes.flatMap(i => namedNodeValues[i]));

      const possibleNext: IEddieRoutingEntry[] = [];

      for (let nextIdx = 0; nextIdx < n; nextIdx++) {
        if ((state & (1 << nextIdx)) === 0) {
          // First check for overlapping variables between triple patterns
          const varsNext = variables[nextIdx];
          const joinVars = varsNext.filter(v => doneVars.has(v.value));
          // When variables overlap we add it to possible next routing decision
          if (joinVars.length > 0) {
            possibleNext.push({ next: nextIdx, joinVars });
            continue;
          }
          // Same approach but for IRIs, if a IRI matches between triple patterns
          // it is a valid routing decision
          const namedNodesNext = namedNodes[nextIdx];
          const joinNamedNodes = namedNodesNext.filter(n => doneNamedNodes.has(n.value));
          if (joinNamedNodes.length > 0) {
            // JoinVars is used to determine matches using the hash function. We
            // set join vars to empty for a join with no variables. This
            // means the hash for these joins always match
            possibleNext.push({ next: nextIdx, joinVars });
          }
        }
      }

      routeTable[state] = possibleNext;
    }
    return routeTable;
  };

  protected getSetBitIndexes(mask: number): number[] {
    const indexes: number[] = [];
    let position = 0;
    while (mask !== 0) {
      if ((mask & 1) === 1) {
        indexes.push(position);
      }
      mask >>>= 1;
      position++;
    }
    return indexes;
  }

  protected getUnSetBitIndexes(mask: number, n: number): number[] {
    const indexes: number[] = [];
    for (let i = 0; i < n; i++) {
      if ((mask & (1 << i)) === 0) {
        indexes.push(i);
      }
    }
    return indexes;
  }

  public routeBinding(binding: Bindings, n: number): number | undefined {
    const done = binding.getContextEntry(eddiesContextKeys.eddiesMetadata)!.done;
    if (done === (1 << n) - 1) {
      return undefined;
    }
    const indexesReadyEddies = this.getUnSetBitIndexes(done, n);

    return Math.min(...indexesReadyEddies);
  }

  public abstract updateRouteTable(
    operators: EddieOperatorStream[],
    routeTable: Record<string, IEddieRoutingEntry[]>
  ): Record<number, IEddieRoutingEntry[]>;
}

export interface IEddieRouter {
  routeBinding: (binding: Bindings, n: number) => number | undefined;
  createRouteTable: (variables: RDF.Variable[][], namedNodes: RDF.NamedNode[][]) => Record<number, IEddieRoutingEntry[]>;
  updateRouteTable: (
    operators: EddieOperatorStream[], routeTable: Record<string, IEddieRoutingEntry[]>
  ) => Record<number, IEddieRoutingEntry[]>;
}

export interface IEddieRoutingEntry {
  next: number;
  joinVars: RDF.Variable[];
}

export interface IEddieRouterFactory {
  /**
   * Creates a new router instance for a specific query execution.
   * @param variables - The variables involved in this specific query.
   */
  createRouter: () => IEddieRouter;
}
