import { StateSpaceTree } from "@comunica/mediator-join-reinforcement-learning";

export class EpisodeLogger{
    stateTree: StateSpaceTree;
    executionTime: number;
    public constructor(){
    }

    public setState(data: StateSpaceTree){
        this.stateTree = data;
    }

    public setTime(executionTime: number){
        this.executionTime = executionTime;
    }
}