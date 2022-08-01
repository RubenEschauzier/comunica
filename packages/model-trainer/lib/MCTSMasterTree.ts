export class MCTSMasterTree{
    // changeQueue: any;
    masterMap: Map<string, MCTSJoinInformation>;
    
    public constructor() {
        this.masterMap = new Map();
        const async = require('async');
        // this.changeQueue = async.queue((task: MCTSJoinPredictionOutput, completed:any) => {
        //     console.log(task.N);
        //     if (task.N == 0){
        //         this.addUnexploredJoin({N:task.N, meanValueJoin: task.predictedValueJoin, totalValueJoin: task.predictedValueJoin, 
        //             priorProbability: task.predictedProbability}, task.state, task.predictedValueJoin, task.predictedProbability)
        //     }
        //     if (task.N > 0){
        //         const joinInformation = this.masterMap.get(task.state);
        //         if (joinInformation == undefined){
        //             throw("Join is undefined in masterMap");
        //         }
        //         /* Join information should never be undefined*/
        //         this.updateJoin(joinInformation!, task.predictedValueJoin, task.predictedProbability, task.state);
        //     }
        //     if (task.N < 0){
        //         throw("Passed item with negative N");
        //     }
        // }, 1);

    }

    public addUnexploredJoin(joinInformation: MCTSJoinInformation, joinIndexes: number[][], estimatedValue: number, estimatedProbability: number){
        // const joinInformation: MCTSJoinInformation = {N: 0, meanValueJoin: estimatedValue, TotalValueJoin: estimatedValue, 
        //     priorProbability: estimatedProbability};
        const key: string = joinIndexes.flat().toString().replaceAll(',', '');
        this.masterMap.set(key, joinInformation);
    }

    public updateJoin(joinInformation: MCTSJoinPredictionOutput){
        /* Get previous information from map */
        const previousStateInformation = this.getJoinInformation(joinInformation.state);
        const key: string = joinInformation.state.flat().toString().replaceAll(',', '');

        /* Update the join information*/
        const newStateInformation = {N: previousStateInformation.N += 1, totalValueJoin: previousStateInformation.totalValueJoin += joinInformation.predictedValueJoin,
            meanValueJoin: previousStateInformation.meanValueJoin = previousStateInformation.totalValueJoin / (joinInformation.N + 1),
            priorProbability: previousStateInformation.priorProbability = joinInformation.predictedProbability
        }

        this.masterMap.set(key, newStateInformation);
    }

    public updateMasterMap(operationInformation: MCTSJoinPredictionOutput){
        this.updateJoin(operationInformation);
        // if (operationInformation.N == 0){
        //     this.addUnexploredJoin({N:operationInformation.N, meanValueJoin: operationInformation.predictedValueJoin, totalValueJoin: operationInformation.predictedValueJoin, 
        //         priorProbability: operationInformation.predictedProbability}, operationInformation.state, operationInformation.predictedValueJoin, operationInformation.predictedProbability)
        // }
        // if (task.N > 0){
        //     const joinInformation = this.masterMap.get(task.state);
        //     if (joinInformation == undefined){
        //         throw("Join is undefined in masterMap");
        //     }
        //     /* Join information should never be undefined*/
        //     this.updateJoin(joinInformation!, task.predictedValueJoin, task.predictedProbability, task.state);
        // }
        // if (task.N < 0){
        //     throw("Passed item with negative N");
        // }
    }

    // public addOperationToQueue(operationInformation: MCTSJoinPredictionOutput){
    //     this.changeQueue.push(operationInformation);
    // }

    public getJoinInformation(joinIndexes: number[][]): MCTSJoinInformation{
        const key: string = joinIndexes.flat().toString().replaceAll(',', '');
        const value: MCTSJoinInformation|undefined = this.masterMap.get(key);
        if (value == undefined || null){
            const emptyResult: MCTSJoinInformation = {N:0, meanValueJoin: 0, totalValueJoin: 0, priorProbability: 0};
            return emptyResult
        }
        else{
            return value;
        }
         
    }
}

export interface MCTSJoinInformation{
    /* 
    * Number of times join state is visited 
    */
    N: number;
    /*
    * Average predicted value of the join
    */
    meanValueJoin: number;
    /*
    * Total predicted value of the join
    */
    totalValueJoin: number;
    /*
    * Predicted probability of choosing this join state
    */
    priorProbability: number;
}

export interface MCTSJoinPredictionOutput{
    /* 
    * Number of times join state is visited 
    */
    N: number;
    /* 
    * Current state
    */
    state: number[][];
    /* 
    * Predicted value of the join state 
    */
    predictedValueJoin: number;
    /* 
    * Predicted probability of choosing this join state 
    */
    predictedProbability: number;
}
