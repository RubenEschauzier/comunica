import { IJoinEntry } from "@comunica/types";
import { Operation } from "sparqlalgebrajs/lib/algebra";
import type * as RDF from '@rdfjs/types';
import { timeStamp } from "console";
import { FifoQueue } from "./FifoQueue";

export class JoinGraph {
    private adjacencyList: Map<number, number[]>;
    private entries: Operation[];
    public size: number;

    constructor(entries: Operation[]) {
        this.entries = entries;
        this.adjacencyList = new Map();
        this.size = entries.length
    }

    public constructJoinGraphBFS(start: Operation){
        // First extract all variables in each of the joined triple patterns
        const variablesTriplePatterns: Set<string>[] = [];
        this.entries.forEach((x, i) => {
            this.adjacencyList.set(i, []);
            variablesTriplePatterns.push(this.extractVariables(x));
        });
        
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

    public getAdjencyList(){
        return this.adjacencyList
    }
}