export class MCTSMasterTree{    
    /* Map containing all exploration information of one query iteration */
    masterMap: Map<string, MCTSJoinInformation>;
    /* Keys of all join states reached in one query execution */
    joinsExploredOneExecution: string[];
    totalEntries: number;
    runningMeanStd: number[][]

    public constructor(runningMeanStd: number[][]) {
        this.masterMap = new Map();
        this.joinsExploredOneExecution = [];
        this.runningMeanStd = runningMeanStd
    }

    public addUnexploredJoin(joinInformation: MCTSJoinInformation, joinIndexes: number[][], estimatedValue: number, estimatedProbability: number){
        // const joinInformation: MCTSJoinInformation = {N: 0, meanValueJoin: estimatedValue, TotalValueJoin: estimatedValue, 
        //     priorProbability: estimatedProbability};
        const key: string = joinIndexes.flat().toString().replaceAll(',', '');
        this.masterMap.set(key, joinInformation);
    }

    public updateJoin(joinInformation: MCTSJoinPredictionOutput, joinFeatureMatrix: number[][], joinAdjacancyMatrix: number[][]){
        /* Get previous information from map */
        const previousStateInformation = this.getJoinInformation(joinInformation.state);
        const key: string = joinInformation.state.flat().toString().replaceAll(',', '');

        /* Calling this function means we explore this join, thus we add it to the explored joins array */
        this.joinsExploredOneExecution.push(key);

        /* Update the join information*/
        const newStateInformation: MCTSJoinInformation = {N: previousStateInformation.N += 1, totalValueJoin: previousStateInformation.totalValueJoin += joinInformation.predictedValueJoin,
            meanValueJoin: previousStateInformation.meanValueJoin = previousStateInformation.totalValueJoin / (joinInformation.N + 1),
            priorProbability: previousStateInformation.priorProbability = joinInformation.predictedProbability,
            featureMatrix: joinFeatureMatrix, adjencyMatrix: joinAdjacancyMatrix
            
        }

        this.masterMap.set(key, newStateInformation);
    }

    public updateMasterMap(operationInformation: MCTSJoinPredictionOutput, joinFeatureMatrix: number[][], joinAdjacancyMatrix: number[][]){
        this.updateJoin(operationInformation, joinFeatureMatrix, joinAdjacancyMatrix);
    }

    public setExecutionTimeJoin(key: string, executionTime: number){
        /* When we call this function we can be sure the key is in the map */
        /* Should prob do some error handling here */

        const stateInformation: MCTSJoinInformation = this.masterMap.get(key)!;
        const updatedInformation: MCTSJoinInformation = {...stateInformation, actualExecutionTime: executionTime};
        this.masterMap.set(key, updatedInformation);
        
    }

    public getJoinInformation(joinIndexes: number[][]): MCTSJoinInformation{
        const key: string = joinIndexes.flat().toString().replaceAll(',', '');
        const value: MCTSJoinInformation|undefined = this.masterMap.get(key);
        if (value == undefined || null){
            const emptyResult: MCTSJoinInformation = {N:0, meanValueJoin: 0, totalValueJoin: 0, priorProbability: 0, 
                featureMatrix: [], adjencyMatrix: [[]]};
            return emptyResult
        }
        else{
            return value;
        }
         
    }

    public getExploredJoins(){
        return this.joinsExploredOneExecution;
    }

    public resetExploredJoins(){
        this.joinsExploredOneExecution = [];
    }

    public setTotalEntries(totalEntries: number){
        this.totalEntries = totalEntries;
    }

    public getTotalEntries(){
        return this.totalEntries;
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
    /*
    * Feature matrix of the current state
    */
    featureMatrix: number[][];
    /* 
    * Adjency matrix of the made joins
    */
    adjencyMatrix: number[][];
    /*
    * 
    */
    
    /*
    * Actual recorded value
    */
    actualExecutionTime?: number;

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
