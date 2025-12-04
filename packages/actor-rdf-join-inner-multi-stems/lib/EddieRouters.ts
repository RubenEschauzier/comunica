import type { Bindings } from '@comunica/utils-bindings-factory';

import type * as RDF from '@rdfjs/types';
import { eddiesContextKeys } from './EddieControllerStream';
import { EddieOperatorStream, ISelectivityData } from './EddieOperatorStream';

export abstract class RouteBase implements IEddieRouter {
  /**
   * Route table mapping the done vector to a routing order
   */
  protected routeTable: Record<number, IEddieRoutingEntry[]>;
  /**
   * Connected components of join graph. Represented by their bitmask index
   */
  protected connectedComponents: number[][];
  /**
   * Number of triple patterns
   */
  protected n: number;

  public constructor() {
    this.routeTable = {};
  }
  
  public init(n: number){
    this.n = n;
  }
  public createRouteTable(variables: RDF.Variable[][]): Record<number, IEddieRoutingEntry[]> {
    const n = variables.length;
    const routeTable: Record<number, IEddieRoutingEntry[]> = {};
    const variableValues = variables.map(vArr => vArr.map(x => x.value));

    const totalStates = 1 << n;
    for (let state = 1; state < totalStates; state++) {
      // Build doneVector array from bits of state
      const doneIndexes = this.getSetBitIndexes(state);

      // // If all done, there is no next entry
      if (doneIndexes.length === n) {
        continue;
      }

      const doneVars = new Set(doneIndexes.flatMap(i => variableValues[i]));

      const possibleNext: IEddieRoutingEntry[] = [];
      
      for (let nextIdx = 0; nextIdx < n; nextIdx++) {
        if ((state & (1 << nextIdx)) === 0) {
          const varsNext = variables[nextIdx];
          const joinVars = varsNext.filter(v => doneVars.has(v.value));
          if (joinVars.length > 0) {
            possibleNext.push({ next: nextIdx, joinVars });
          }
        }
      }

      routeTable[state] = possibleNext;
    }
    this.routeTable = routeTable;
    return routeTable;
  };

  protected getSetBitIndexes(mask: number): number[] {
    const indexes: number[] = [];
    let position = 0;
    while (mask !== 0) {
      if ((mask & 1) === 1) {
        indexes.push(position);
      }
      mask >>>= 1; // Unsigned right shift
      position++;
    }
    return indexes;
  }

  protected getUnSetBitIndexes(mask: number): number[] {
    const indexes: number[] = [];
    for (let i = 0; i < this.n; i++) {
      if ((mask & (1 << i)) === 0) {
        indexes.push(i);
      }
    }
    return indexes;
  }

  public routeBinding(binding: Bindings): number | undefined {
    const done = binding.getContextEntry(eddiesContextKeys.eddiesMetadata)!.done;
    if (done === (1 << this.n) - 1) {
      return undefined;
    }
    const indexesReadyEddies = this.getUnSetBitIndexes(done);

    return Math.min(...indexesReadyEddies);
  }

  abstract updateRouteTable(operators: EddieOperatorStream[]): Record<number, IEddieRoutingEntry[]>;
}

export class RouteRandom extends RouteBase {
  public override routeBinding(binding: Bindings): number | undefined {
    const done = binding.getContextEntry(eddiesContextKeys.eddiesMetadata)!.done;
    if (done === (1 << this.n) - 1) {
      return undefined;
    }
    const indexesReadyEddies = this.getUnSetBitIndexes(done);
    const nextEntry = this.sampleRandom(indexesReadyEddies);
    return nextEntry;
  }

  public updateRouteTable(operators: EddieOperatorStream[]): Record<number, IEddieRoutingEntry[]> {
    return this.routeTable;
  };

  private sampleRandom<T>(arr: T[]) {
    return arr[Math.floor(Math.random() * arr.length)];
  }
}

export class RouteFixedMinimalIndex extends RouteBase {
  public updateRouteTable(operators: EddieOperatorStream[]): Record<number, IEddieRoutingEntry[]> {
    return this.routeTable;
  };
}

export class RouteLotteryScheduling extends RouteBase {

  public override updateRouteTable(operators: EddieOperatorStream[]): Record<number, IEddieRoutingEntry[]> {
    const ticketMetadata: Record<string, number>[] = operators.map(op => {
      return {tickets: op.tickets}
    });

    const updatedRoutingTable: Record<string, IEddieRoutingEntry[]> = {};
    for (const [doneKey, routing] of Object.entries(this.routeTable)){
        const key = parseInt(doneKey, 10);
        // If size 0 or 1 no choice to be made
        if (routing.length < 2){
          updatedRoutingTable[key] = routing;
          continue;
        }
        const ticketWeights: number[] = routing.map(x => x.next).map(idx => ticketMetadata[idx].tickets);

        const minScore = Math.min(...ticketWeights);
        const offset = minScore < 0 ? -minScore : 0;
        const ticketWeightsNonNegative = ticketWeights.map(w => w + offset + 1); 
        updatedRoutingTable[key] = this.reorderWeightedChoice(routing, ticketWeightsNonNegative);
    }

    return updatedRoutingTable;
  }

  protected reorderWeightedChoice<T>(items: T[], weights: number[]): T[] {
    const reordered = [...items];
    const total = weights.reduce((a, b) => a + b, 0);
    const r = Math.random() * total;
    let acc = 0;
    for (let i = 0; i < items.length; i++) {
      acc += weights[i];
      if (r < acc) {
        const temp = reordered[0];
        reordered[0] = reordered[i];
        reordered[i] = temp;
        return reordered;
      };
    }
    const temp = reordered[0];
    reordered[0] = reordered[items.length - 1];
    reordered[items.length - 1] = temp;
    return reordered; 
  }
}

export class RouteLotterySchedulingSignature extends RouteLotteryScheduling {
  // public override createRouteTable(variables: RDF.Variable[][]): Record<number, IEddieRoutingEntry[]> {
  //   const routeTable = super.createRouteTable(variables);
  //   for (const key of Object.keys(routeTable)){
  //     const keyInt = parseInt(key, 10);
  //     routeTable[keyInt] = this.shuffle(routeTable[keyInt]);
  //   }
  //   this.routeTable = routeTable;
  //   return routeTable;
  // }
  
  public override updateRouteTable(operators: EddieOperatorStream[]): Record<number, IEddieRoutingEntry[]> {
    const selectivityMetadata: Record<string, Record<number, ISelectivityData>>[] = this.collectMetadata(operators);

    const updatedRoutingTable: Record<string, IEddieRoutingEntry[]> = {};
    for (const [doneKey, routing] of Object.entries(this.routeTable)){
        const key = parseInt(doneKey, 10);
        // If size 0 or 1 no choice to be made
        if (routing.length < 2){
          updatedRoutingTable[key] = routing;
          continue;
        }
        const ticketWeights: number[] = [];
        for (const idx of routing.map(x => x.next)){
          let selectivityData = selectivityMetadata[idx].selectivity[key];
          if (!selectivityData){
            ticketWeights.push(0);
          }
          else{
            ticketWeights.push(selectivityData.in - selectivityData.out);
          }
        }

        const minScore = Math.min(...ticketWeights);
        const offset = minScore < 0 ? -minScore : 0;
        const ticketWeightsNonNegative = ticketWeights.map(w => w + offset + 1); 
        updatedRoutingTable[key] = this.reorderWeightedChoice(routing, ticketWeightsNonNegative);
    }
    return updatedRoutingTable;
  }

  protected collectMetadata(operators: EddieOperatorStream[]){
    return operators.map(op => {
      return {selectivity: op.selectivities}
    });
  }

  protected shuffle<T>(array: T[]): T[] {
    const result = array.slice(); // create a copy to avoid mutating original array
    for (let i = result.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [result[i], result[j]] = [result[j], result[i]];
    }
    return result;
  }
}

// Eddie routers should support batched routing with certain batchsize
export interface IEddieRouter {
  init: (n: number) => void;
  routeBinding: (binding: Bindings) => number | undefined;
  createRouteTable: (variables: RDF.Variable[][]) => Record<number, IEddieRoutingEntry[]>;
  updateRouteTable: (operators: EddieOperatorStream[]) => Record<number, IEddieRoutingEntry[]>;
}

export interface IEddieRoutingEntry {
  next: number;
  joinVars: RDF.Variable[];
}
