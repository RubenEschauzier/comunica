import { IJoinEntry } from "@comunica/types";
import { Operation } from "sparqlalgebrajs/lib/algebra";
import type * as RDF from '@rdfjs/types';
import { timeStamp } from "console";
import { FifoQueue } from "./FifoQueue";

export class JoinGraph {
    private adjacencyList: Map<number, number[]>;
    private entries: Operation[];
    public indexToOrder: Map<number, number>;
    public orderToIndex: Map<number, number>;

    constructor(entries: Operation[]) {
        this.entries = entries;
        this.adjacencyList = new Map();
    }

    public constructJoinGraphBFS(start: Operation){
        // First extract all variables in each of the joined triple patterns
        const variablesTriplePatterns: Set<string>[] = [];
        this.entries.forEach((x, i) => {
            this.adjacencyList.set(i, []);
            variablesTriplePatterns.push(this.extractVariables(x));
        })
        const operationToIndex: Map<Operation, number> = new Map();
        const queue: FifoQueue<Operation> = new FifoQueue();
        const visited: Map<Operation, boolean> = new Map();

        let currentVertex = 0;
        let nextAvailableIndex = 1;
        // Start BFS from the startVertex
        queue.enqueue(start);
        visited.set(start, true);

        while (queue.size() > 0) {
            const vertex = queue.dequeue()!;
            operationToIndex.set(vertex, currentVertex);
            // Get all the adjacent vertices
            const neighbors = this.getJoinConnections(vertex, variablesTriplePatterns);
            if (neighbors) {
                for (const neighbor of neighbors) {
                    if (!operationToIndex.has(neighbor)){
                        operationToIndex.set(neighbor,nextAvailableIndex);
                        nextAvailableIndex += 1
                    }
                    this.adjacencyList.get(currentVertex)!.push(operationToIndex.get(neighbor)!)
                    // If the neighbor hasn't been visited, visit it
                    if (!visited.has(neighbor)) {
                        queue.enqueue(neighbor);
                        visited.set(neighbor, true);
                    }

                }
            }
            currentVertex += 1
        }

        // Reorder operations according to BFS order
        for (const op of operationToIndex.keys()){
            this.entries[operationToIndex.get(op)!] = op;
        }
        console.log(this.adjacencyList)
        console.log(this.entries)
    }

    public getJoinConnections(op: Operation, variablesTriplePatterns: Set<string>[]): Operation[]{
        const connections: Operation[] = [];
        for (let i = 0; i < this.entries.length; i++){
            for (const variable of this.extractVariables(op)){
                if (variablesTriplePatterns[i].has(variable) && this.entries[i] !== op){
                    connections.push(this.entries[i]);
                }
            }
        }
        return connections;
    }

    /**
     * Extracts all variables in triple pattern, currently does not take graph into account,
     * extend by adding 'graph' to the forEached array
     * @param op 
     */
    public extractVariables(op: Operation){
        const variables: Set<string> = new Set();
        [ 'subject', 'predicate', 'object' ].forEach((type)=>{
            if (op[type].termType == 'Variable') {
                variables.add(op[type].value);
            }
        });
        return variables;
    }

    public getEntries(){
        return this.entries
    }
        
    // // BFS function to label/number vertices
    // bfsNumbering(startVertex: number){
    //     const queue: FifoQueue<number> = new FifoQueue();
    //     const visited: Map<number, boolean> = new Map();
    //     const indexToOrder: Map<number, number> = new Map();

    //     let currentNumber = 0;

    //     // Start BFS from the startVertex
    //     queue.enqueue(startVertex);
    //     visited.set(startVertex, true);
    //     indexToOrder.set(startVertex, currentNumber++);

    //     while (queue.size() > 0) {
    //         const vertex = queue.dequeue()!;
            
    //         // Get all the adjacent vertices
    //         const neighbors = this.adjacencyList.get(vertex);
    //         if (neighbors) {
    //             for (const neighbor of neighbors) {
    //                 // If the neighbor hasn't been visited, visit it
    //                 if (!visited.has(neighbor)) {
    //                     queue.enqueue(neighbor);
    //                     visited.set(neighbor, true);
    //                     indexToOrder.set(neighbor, currentNumber++);
    //                 }
    //             }
    //         }
    //     }
    //     return indexToOrder
    // }

        // public constructOrderedJoinGraph(){
    //     // First extract all variables in each of the joined triple patterns
    //     const variablesTriplePatterns: Set<string>[] = [];
    //     this.entries.forEach((x, i) => {
    //         this.adjacencyList.set(i, []);
    //         variablesTriplePatterns.push(this.extractVariables(x));
    //     })
    //     // Determine which triple patterns have overlapping variables
    //     for (let i = 0; i < this.entries.length; i++){
    //         for (let j = i + 1; j < this.entries.length; j++){
    //             for (const variable of variablesTriplePatterns[i]){
    //                 if (variablesTriplePatterns[j].has(variable)){
    //                     this.adjacencyList.get(i)!.push(j);
    //                     this.adjacencyList.get(j)!.push(i);
    //                     continue;
    //                 }
    //             }
    //         }
    //     }
    //     const indexToOrder = this.bfsNumbering(0);
    //     const orderedEntries = Array.from(indexToOrder.values()).map(x=>this.entries[x]);
        
    // }


    // public constructJoinGraph(){
    //     // First extract all variables in each of the joined triple patterns
    //     const variablesTriplePatterns: Set<string>[] = [];
    //     this.entries.forEach((x, i) => {
    //         this.adjacencyList.set(i, []);
    //         variablesTriplePatterns.push(this.extractVariables(x));
    //     })
    //     // Determine which triple patterns have overlapping variables
    //     for (let i = 0; i < this.entries.length; i++){
    //         for (let j = i + 1; j < this.entries.length; j++){
    //             for (const variable of variablesTriplePatterns[i]){
    //                 if (variablesTriplePatterns[j].has(variable)){
    //                     this.adjacencyList.get(i)!.push(j);
    //                     this.adjacencyList.get(j)!.push(i);
    //                     continue;
    //                 }
    //             }
    //         }
    //     }
    //     return this.adjacencyList;
    // }

}