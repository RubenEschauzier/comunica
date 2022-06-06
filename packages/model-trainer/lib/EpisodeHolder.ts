import { StateSpaceTree } from "@comunica/mediator-join-reinforcement-learning";

export class EpisodeLogger{
    joinState: StateSpaceTree
    adjMatrix: number[][];
    features: number[]
    executionTime: number;
    public constructor(){
    }

    public setState(joinState: StateSpaceTree){
        // this.adjMatrix = adjMatrix;
        // this.features = features
        joinState = joinState
    }

    public setTime(executionTime: number){
        this.executionTime = executionTime;
    }
}