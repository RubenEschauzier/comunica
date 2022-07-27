export class MCTSMasterTree{
    changeQueue: any;
    masterMap: Map<number[][], MCTSJoinInformation>;
    
    public constructor() {
        // this.masterMap = new Map();
        const async = require('async');
        this.changeQueue = async.queue((task: MCTSJoinPredictionOutput, completed:any) => {

            if (task.N == 0){
                this.addUnexploredJoin({N:task.N, meanValueJoin: task.predictedValueJoin, totalValueJoin: task.predictedValueJoin, 
                    priorProbability: task.predictedProbability}, task.state, task.predictedValueJoin, task.predictedProbability)
            }
            if (task.N > 0){
                const joinInformation = this.masterMap.get(task.state);
                if (joinInformation == undefined){
                    throw("Join is undefined in masterMap")
                }
                /* Join information should never be undefined*/
                this.updateJoin(joinInformation!, task.predictedValueJoin, task.predictedProbability, task.state);
            }
            if (task.N < 0){
                throw("Passed item with negative N");
            }
        }, 1);

    }

    public addUnexploredJoin(joinInformation: MCTSJoinInformation, joinIndexes: number[][], estimatedValue: number, estimatedProbability: number){
        // const joinInformation: MCTSJoinInformation = {N: 0, meanValueJoin: estimatedValue, TotalValueJoin: estimatedValue, 
        //     priorProbability: estimatedProbability};
        this.masterMap.set(joinIndexes, joinInformation);
    }

    public updateJoin(joinInformation: MCTSJoinInformation, estimatedValue: number, estimatedProbability: number, joinIndexes: number[][]){
        joinInformation.N += 1;
        joinInformation.totalValueJoin += estimatedValue;
        joinInformation.meanValueJoin = joinInformation.totalValueJoin / joinInformation.N;

        this.masterMap.set(joinIndexes, joinInformation);
    }

    public addOperationToQueue(operationInformation: MCTSJoinInformation){
        this.changeQueue.push(operationInformation);
    }

    public getJoinInformation(joinIndexes: number[][]): MCTSJoinInformation{
        const value: MCTSJoinInformation|undefined = this.masterMap.get(joinIndexes);
        if (value == null){
            throw(`Key ${joinIndexes} not in masterMap`);
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
