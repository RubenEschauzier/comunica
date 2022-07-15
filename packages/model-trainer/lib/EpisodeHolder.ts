import { StateSpaceTree } from "@comunica/mediator-join-reinforcement-learning";

export class EpisodeLogger{
    joinState: StateSpaceTree
    adjMatrix: number[][];
    features: number[];
    executionTime: number;
    public constructor(){
    }

    public setState(joinState: StateSpaceTree): void{
        // this.adjMatrix = adjMatrix;
        // this.features = features
        this.joinState = joinState;
    }

    public setTime(executionTime: number): void{
        this.executionTime = executionTime;
    }

    public getState(): StateSpaceTree{
        return this.joinState;
    }

    public getTime(): number{
        return this.executionTime;
    }
}