import { StateSpaceTree } from "@comunica/mediator-join-reinforcement-learning";
import { MCTSJoinInformation, MCTSJoinPredictionOutput, MCTSMasterTree } from "./MCTSMasterTree";

export class EpisodeLogger{
    joinState: StateSpaceTree;
    masterTree: MCTSMasterTree;
    planHolder: Map<string, number>;
    node_idmapping: Map<number, number>;

    adjMatrix: number[][];
    features: number[];
    executionTime: number;
    testMap: Map<number[][], number>;


    public constructor(){
        this.node_idmapping = new Map<number, number>();
        
    }

    public setState(joinState: StateSpaceTree): void{
        // this.adjMatrix = adjMatrix;
        // this.features = features
        this.joinState = joinState;
    }

    public setTime(executionTime: number): void{
        this.executionTime = executionTime;
    }

    public setPlanHolder(planHolder: Map<string, number>){
        this.planHolder = planHolder;
    }
    
    public setPlan(joinPlan: string){
        this.planHolder.set(joinPlan, 0);
    }

    public addJoinIndexes(joinIndexes: number[]): void{
        this.joinState.addJoinIndexes(joinIndexes);
    }

    public setJoinState(joinState: MCTSJoinPredictionOutput, featureMatrix: number[][], adjacencyMatrix: number[][]){
        this.masterTree.updateMasterMap(joinState, featureMatrix, adjacencyMatrix);
    }

    public updateRunningMomentsMasterTree(newValue: number, index: number): void{
        this.masterTree.updateRunningMoments(newValue, index);
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

    public getRunningMomentsMasterTree(){
        return this.masterTree.getRunningMoments();
    }

    public getTotalEntriesMasterTree(){
        return this.masterTree.getTotalEntries();
    }

    public setTotalEntriesMasterTree(totalEntries: number){
        this.masterTree.setTotalEntries(totalEntries);
    }
    public setNodeIdMapping(key: number, value: number){
        this.node_idmapping.set(key, value);
    }

    public getNodeIdMappingKey(key: number){
        return this.node_idmapping.get(key)!;
    }
}