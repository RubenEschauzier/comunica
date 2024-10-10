import { Operation } from "sparqlalgebrajs/lib/algebra";
import { JoinGraph } from "./JoinGraph";

export class JoinOrderEnumerator{
    private joinGraph: JoinGraph;

    public constructor(joinGraph: JoinGraph){
        this.joinGraph = joinGraph;
        joinGraph.constructJoinGraphBFS(this.joinGraph.getEntries()[0]);
    }

    public enumerateCsg(tps: Operation[]){
        const csgs: Set<number>[]= [];
        for (let i = tps.length-1; i>0; i--){
            const vI = new Set([i]);
            csgs.push(vI);
            this.enumerateCsgRecursive(csgs, vI, new Set([...Array(i).keys()]))
        }
    }

    public enumerateCsgRecursive(csgs: Set<number>[], S: Set<number>, X: Set<number>){

    }

}