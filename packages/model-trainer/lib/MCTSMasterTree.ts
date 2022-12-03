import * as tf from '@tensorflow/tfjs-node';
import { fstat } from 'fs';
import path = require('path');

export class MCTSMasterTree{    
    /* Map containing all exploration information of one query iteration */
    masterMap: Map<string, MCTSJoinInformation>;
    /* Keys of all join states reached in one query execution */
    joinsExploredOneExecution: string[];
    totalEntries: number;
    runningMoments: runningMoments

    public constructor(runningMeanStd: runningMoments) {
        this.masterMap = new Map();
        this.joinsExploredOneExecution = [];
        this.runningMoments = runningMeanStd
    }

    public addUnexploredJoin(joinInformation: MCTSJoinInformation, joinIndexes: number[][], estimatedValue: number, estimatedProbability: number){
        // const joinInformation: MCTSJoinInformation = {N: 0, meanValueJoin: estimatedValue, TotalValueJoin: estimatedValue, 
        //     priorProbability: estimatedProbability};
        const key: string = joinIndexes.flat().toString().replaceAll(',', '');
        this.masterMap.set(key, joinInformation);
        console.trace()
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

    public updateJoinDirect(newJoinInformation: MCTSJoinInformation, joinIndexes: number[][]){
        /**
         * Directly updates join information in map, should only be used when specific values of the join information should be changed, like prior probability
         */
        const key: string = joinIndexes.flat().toString().replaceAll(',', '');
        this.masterMap.set(key, newJoinInformation);
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

    public getRunningMoments(): runningMoments{
        return this.runningMoments
    }

    public updateRunningMoments(newValue: number, index: number){
        // Update procedure from Welford's online algorithm for running mean and variance without numerical instabilities
        const toUpdateAggregate: aggregateValues | undefined = this.getRunningMoments().runningStats.get(index);
        if (toUpdateAggregate){
            toUpdateAggregate.N +=1;
            const delta = newValue - toUpdateAggregate.mean; 
            toUpdateAggregate.mean += delta / toUpdateAggregate.N;
            const newDelta = newValue - toUpdateAggregate.mean;
            toUpdateAggregate.M2 += delta * newDelta;
            toUpdateAggregate.std = Math.sqrt(toUpdateAggregate.M2 / toUpdateAggregate.N);
        }
        else{
            throw Error("Tried to get running moments index that does not exist");
        }
    }
    public loadRunningMoments(momentLocation: string){
        const fs = require('fs');
        // const data = fs.readFileSync(momentLocation);
        const test = path.join(__dirname, '..','..','actor-rdf-join-inner-multi-reinforcement-learning','model',momentLocation,'runningWeightsX');
        const data = fs.readFileSync(test, 'utf8');
        const dataParsed = JSON.parse(data);
        const indexes = dataParsed.indexes
        const runningIndexStats: Map<number, aggregateValues> = new Map();
        for (const index of indexes){
            const strIndex = index.toString()
            runningIndexStats.set(index, dataParsed[strIndex]);
        }
        const loadedRunningMoments: runningMoments = {indexes: indexes, runningStats: runningIndexStats}
        this.runningMoments = loadedRunningMoments;
        console.log("Loaded Running Moments")
        console.log(loadedRunningMoments);
        // console.log(data)
        // serialiseObject['indexes'] = runningMomentsX.indexes;
        // for (const [key, value] of runningMomentsX.runningStats){
        //   serialiseObject[key] = value;
        // }
        // fs.writeFile(fileLocationX, JSON.stringify(serialiseObject), function(err: any) {
        //     if(err) {
        //         return console.log(err);
        //     }
        // }); 
    
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
    * Estimated probability as tensor to retain gradients, very ugly solution.. NEED TO BE DISPOSED AFTER RESET OF MASTER TREE
    */
    estimatedProbabilityTensor?: tf.Tensor;
    /*
    * Probability vector of actualProbabilities
    */
    actualProbabilityVector?: number[];
    /*
    * Probability vector of estimated probabilities of children
    */
    predictedProbabilityVectorTensor?: tf.Tensor[];
    /* Probability vector of estimated probabilities of children, in array form */
    predictedProbabilityVector?: number[]

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

export interface runningMoments{
    /*
    * Indexes of features to normalise and track
    */
    indexes: number[];
    /*
    * The running values of mean, std, n, mn mapped to index
    * These are the quantities required for the Welfold's online algorithm
    */
    runningStats: Map<number, aggregateValues>;
}

export interface aggregateValues{
    N: number;
    /*
    * Aggregate mean
    */
    mean: number;
    /*
    * Aggregate standard deviation
    */
    std: number;
    /*
    * Aggregate M2
    */
    M2: number;
}
