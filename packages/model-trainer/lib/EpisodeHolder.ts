import { StateSpaceTree } from "@comunica/mediator-join-reinforcement-learning";
import { MCTSJoinInformation, MCTSJoinPredictionOutput, MCTSMasterTree } from "./MCTSMasterTree";

export class EpisodeLogger{
    joinState: StateSpaceTree;
    masterTree: MCTSMasterTree;
    node_idmapping: Map<number, number>;

    adjMatrix: number[][];
    features: number[];
    executionTime: number;
    testMap: Map<number[][], number>;


    public constructor(){
        this.node_idmapping = new Map<number, number>;
        
    }

    public setState(joinState: StateSpaceTree): void{
        // this.adjMatrix = adjMatrix;
        // this.features = features
        this.joinState = joinState;
    }

    public setTime(executionTime: number): void{
        this.executionTime = executionTime;
    }

    public addJoinIndexes(joinIndexes: number[]): void{
        this.joinState.addJoinIndexes(joinIndexes);
    }

    public setJoinState(joinState: MCTSJoinPredictionOutput){
        this.masterTree.updateMasterMap(joinState);
    }

    public getState(): StateSpaceTree{
        return this.joinState;
    }

    public getTime(): number{
        return this.executionTime;
    }

    public getNodeMapping(): Map<number, number>{
        return this.node_idmapping;
    }

    public setMasterTree(treeToSet: MCTSMasterTree): void{
        this.masterTree = treeToSet;
    }

    public getJoinStateMasterTree(joinIndexes: number[][]){
        return this.masterTree.getJoinInformation(joinIndexes);
    }

    public setNodeIdMapping(key: number, value: number){
        this.node_idmapping.set(key, value);
    }

    public getNodeIdMappingKey(key: number){
        return this.node_idmapping.get(key)!;
    }
}