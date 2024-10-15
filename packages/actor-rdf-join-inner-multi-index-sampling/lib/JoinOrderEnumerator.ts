import { Operation } from "sparqlalgebrajs/lib/algebra";
import { JoinTree } from "./JoinTree";
import { ISampleResult } from "./IndexBasedJoinSampler";

export class JoinOrderEnumerator{
    private adjacencyList: Map<number, number[]>;
    private readonly estimates: Record<string, ISampleResult>;
    private readonly entries: Operation[];

    public constructor(
        adjacencyList: Map<number, number[]>, 
        estimates: Record<string, ISampleResult>,
        entries: Operation[]
    ){
        this.adjacencyList = adjacencyList;
        this.estimates = estimates;
        this.entries = entries
    }

    public search(){
        const bestPlan: Map<Set<number>, JoinTree> = new Map();
        for (let i = 0; i < this.entries.length; i++){
            const singleton = new Set([i]);
            bestPlan.set(singleton, new JoinTree([], singleton));
            bestPlan.get(singleton)?.setCost(this.estimates[JSON.stringify([...singleton])].estimatedCardinality);
        }
        const csgCmpPairs = this.enumerateCsgCmpPairs(this.entries.length);
        for (const csgCmpPair of csgCmpPairs){
            const tree1 = bestPlan.get(csgCmpPair[0])!;
            const tree2 = bestPlan.get(csgCmpPair[1])!;

            const currPlan = JoinTree.mergeTree(tree1, tree2);
            //TODO This will fail due to the order being important for the estimated cardinalities. Convert this record<> estimates to Map instead 
            const cost = tree1.cost + tree2.cost + 
                this.estimates[JSON.stringify([...currPlan.entries])].estimatedCardinality!;
            if (!bestPlan.get(currPlan.entries) || bestPlan.get(currPlan.entries)!.cost > cost){
                bestPlan.set(currPlan.entries, currPlan);
            }
        }
        console.log(bestPlan);
    }

    public enumerateCsgCmpPairs(nTps: number){
        const allTps = new Set([...Array(nTps).keys()]) ;
        const csgs = this.enumerateCsg(nTps);
        const csgCmpPairs: [Set<number>, Set<number>][] = []

        for (const csg of csgs){
            const cmps = this.enumerateCmp(csg, allTps);
            for (const cmp of cmps){
                csgCmpPairs.push([csg, cmp]);
            }
        }
        return csgCmpPairs
    }

    public enumerateCsg(nTps: number){
        const csgs: Set<number>[]= [];
        for (let i = nTps-1; i>=0; i--){
            const vI = new Set([i]);
            csgs.push(vI);
            this.enumerateCsgRecursive(csgs, vI, new Set([...Array(i+1).keys()]))
        }
       return csgs;
    }

    public enumerateCmp(tpsSubset: Set<number>, allTps: Set<number>){
        const cmps: Set<number>[] = [];
        const X = new Set([...this.reduceSet(this.setMinimum(tpsSubset), allTps), ...tpsSubset]);
        const neighbours = this.getNeighbours(tpsSubset);
        X.forEach((vertex) => neighbours.delete(vertex));

        for (const vertex of this.elementsInOrder(neighbours)){
            cmps.push(new Set([vertex]))
            this.enumerateCsgRecursive(
                cmps, 
                new Set([vertex]), 
                new Set([...X, ...this.reduceSet(vertex, neighbours)])
            );
        }
        return cmps;
    }

    public enumerateCsgRecursive(csgs: Set<number>[], S: Set<number>, X: Set<number>){
        // Difference of neighbours and X
        const neighbours = this.getNeighbours(S);
        X.forEach((vertex) => neighbours.delete(vertex));

        const subsets = this.getAllSubsets(neighbours);
        for (const subset of subsets){
            // Add union of two sets as connected subgraph
            csgs.push(new Set([...S, ...subset]));
        }
        for (const subset of subsets){
            this.enumerateCsgRecursive(csgs, new Set([...S, ...subset]), new Set([...X, ...neighbours]));
        }
    }

    public getNeighbours(S: Set<number>){
        const neighbours: Set<number> = new Set() 
        S.forEach((vertex: number)=>{
            this.adjacencyList.get(vertex)!.forEach((neighbour: number)=>{
                neighbours.add(neighbour);
            })
        })
        return neighbours
    }

    public getAllSubsets(set: Set<number>): Set<number>[] {
        const subsets: Set<number>[] = [new Set([])];
        for (const el of set) {
            const last = subsets.length-1;
            for (let i = 0; i <= last; i++) {
                subsets.push( new Set([...subsets[i], el] ));
            }
        }
        // Remove empty set
        return subsets.slice(1);
    }

    public reduceSet(i: number, set: Set<number>){
        const reducedSet: Set<number> = new Set();
        for (const vertex of set){
            if (vertex <= i){
                reducedSet.add(vertex);
            }
        }
        return reducedSet
    }

    public setMinimum(set: Set<number>): number {
        return Math.min(...set);
    }

    public elementsInOrder(set: Set<number>): number[]{
        return [...set].sort((a, b) => b - a);
    }
}