import { Bindings } from '@comunica/utils-bindings-factory';

import { eddiesContextKeys } from "./EddieControllerStream";
import type * as RDF from '@rdfjs/types';


export abstract class RouteBase implements IEddieRouter{
    protected routeTable: Record<number, IEddieRoutingEntry[]>;
    /**
     * Number of triple patterns
     */
    protected n: number;
    public constructor(n: number){
        this.routeTable = {};
        this.n = n;
    }

    public createRouteTable(variables: RDF.Variable[][]): Record<number, IEddieRoutingEntry[]>{
        const n = variables.length;
        const routeTable: Record<number, IEddieRoutingEntry[]> = {};
        const variableValues = variables.map(vArr => vArr.map(x => x.value));

        const totalStates = 1 << n;
        for (let state = 1; state < totalStates; state++) {
            // Build doneVector array from bits of state
            const doneIndexes = this.getSetBitIndexes(state);

            // TODO: Add this back when we fix support for multiple join variables
            // // If all done, there is no next entry
            if (doneIndexes.length === n) {
                continue;
            }

            const doneVars = new Set(doneIndexes.flatMap(i => variableValues[i]));

            const possibleNext: IEddieRoutingEntry[] = [];

            for (let nextIdx = 0; nextIdx < n; nextIdx++) {
                if ((state & (1 << nextIdx)) === 0){
                    const varsNext = variables[nextIdx];
                    const joinVars = varsNext.filter(v => doneVars.has(v.value));
                    if (joinVars.length > 0) {
                        possibleNext.push({ next: nextIdx, joinVars: joinVars });
                    }         
               }
            }

            routeTable[state] = possibleNext;
        }
        this.routeTable = routeTable;
        return routeTable
    };

    protected getSetBitIndexes(mask: number): number[] {
        const indexes: number[] = [];
        let position = 0;
        while (mask !== 0) {
            if ((mask & 1) === 1) {
                indexes.push(position);
            }
            mask = mask >>> 1;  // unsigned right shift
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
    abstract routeBinding(binding: Bindings): number | undefined;
    abstract updateRouteTable(operatorMetadata: Record<string, any>[]): Record<number, IEddieRoutingEntry[]>;
}


export class RouteRandom extends RouteBase{
    public routeBinding(binding: Bindings): number | undefined{
        const done = binding.getContextEntry(eddiesContextKeys.eddiesMetadata)!.done;
        if (done === (1 << this.n) - 1){
            return undefined
        }
        const indexesReadyEddies = this.getUnSetBitIndexes(done);
        const nextEntry = this.sampleRandom(indexesReadyEddies);
        return nextEntry;
    }

    public updateRouteTable(operatorMetadata: Record<string, any>[]): Record<number, IEddieRoutingEntry[]>{
        return this.routeTable
    };

    private sampleRandom<T>(arr: T[]){
        return arr[Math.floor(Math.random() * arr.length)];
    }
}


export class RouteFixedMinimalIndex extends RouteBase{
    public routeBinding(binding: Bindings): number | undefined {
        const done = binding.getContextEntry(eddiesContextKeys.eddiesMetadata)!.done;
        if (done === (1 << this.n) - 1){
            return undefined
        }
        const indexesReadyEddies = this.getUnSetBitIndexes(done);

        return Math.min(...indexesReadyEddies);
    }
    public updateRouteTable(operatorMetadata: Record<string, any>[]): Record<number, IEddieRoutingEntry[]>{
        return this.routeTable
    };

}


// Eddie routers should support batched routing with certain batchsize
export interface IEddieRouter{
    routeBinding: (binding: Bindings) => number | undefined;
    createRouteTable: (variables: RDF.Variable[][]) => Record<number, IEddieRoutingEntry[]>;
    updateRouteTable: (operatorMetadata: Record<string, any>[]) => Record<number, IEddieRoutingEntry[]>;
}

export interface IEddieRoutingEntry {
    next: number;
    joinVars: RDF.Variable[];
}