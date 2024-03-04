import { IQueryGraphViews, IResultSetRepresentation } from '@comunica/mediator-join-reinforcement-learning';
import * as tf from '@tensorflow/tfjs-node';
import { squaredDifference } from '@tensorflow/tfjs-node';
import { ModelTreeLSTM } from './treeLSTM';
import { IQueryGraphEncodingModels } from './instanceModel';
import { DenseOwnImplementation } from './fullyConnectedLayers';

export class ModelTrainerOffline{
    optimizer: tf.Optimizer
    public constructor(modelTrainerArgs: IModelTrainerOfflineArgs){
        if (modelTrainerArgs.optimizer=='adam'){
            this.optimizer = tf.train.adam(modelTrainerArgs.learningRate)
        }
        else if (modelTrainerArgs.optimizer=='sgd'){
            this.optimizer = tf.train.sgd(modelTrainerArgs.learningRate);
        }
        else{
            throw new Error(`Invalid optimizer argument in ModelTrainerOffline, got: ${modelTrainerArgs.optimizer}, expected ${optimizerOptions}`);
        }
    }

    public trainOfflineBatched(partialJoinTree: number[][][], resultSetRepresentation: IResultSetRepresentation, 
        executionTimes: number[], model: ModelTreeLSTM, batchSize: number){
        return tf.tidy(()=>{
            const numBatches = Math.ceil(executionTimes.length/batchSize);
            let episodeLoss: number = 0;
            for (let b=0; b<numBatches;b++){
                const treesToExecute: number[][][] = partialJoinTree.slice(b*batchSize, Math.min((b+1)*batchSize, partialJoinTree.length));
                const loss: tf.Scalar|null = this.optimizer.minimize(()=>{
                    const qValuesRecursive: tf.Tensor[] = [];
                    for (const tree of treesToExecute){
                        qValuesRecursive.push(model.forwardPassRecursive(resultSetRepresentation, tree));
                    }
                    const executionTimesBatch: tf.Tensor[] = executionTimes.slice(b*batchSize, Math.min((b+1)*batchSize, executionTimes.length)).map(x=>tf.tensor(x));
                    const loss = tf.sum(squaredDifference(tf.concat(qValuesRecursive).squeeze(), tf.concat(executionTimesBatch).squeeze()));
                    return loss.squeeze();
                }, true);

                if (loss==null){
                    throw new Error("Received null loss");
                }
                const lossArray = loss.dataSync();
                if (lossArray.length > 1){
                    throw new Error("Loss is array with multiple elements");
                }
                episodeLoss += lossArray[0];
            }
            return episodeLoss/executionTimes.length;
        });
    };

    public trainOfflineExperienceReplayTemp(partialJoinTree: number[][][], leafFeaturesTensor: tf.Tensor[], 
        graphViews: IQueryGraphViews[], executionTimes: number[], modelLSTM: ModelTreeLSTM, 
        modelsGCN: IQueryGraphEncodingModels, batchSize: number){
        return tf.tidy(()=>{
            const numBatches = Math.ceil(executionTimes.length/batchSize);
            let episodeLoss: number = 0;
            for (let b=0; b<numBatches; b++){
                const treesToExecute: number[][][] = partialJoinTree.slice(b*batchSize, Math.min((b+1)*batchSize, partialJoinTree.length));
                const leafFeaturesBatch: tf.Tensor[] = leafFeaturesTensor.slice(b*batchSize, 
                    Math.min((b+1)*batchSize, leafFeaturesTensor.length));                
                const loss: tf.Scalar|null = this.optimizer.minimize(()=>{
                    const qValuesRecursive: tf.Tensor[] = [];
                    for (let i=0;i<treesToExecute.length;i++){
                        const subjSubjRepresentation = modelsGCN.modelSubjSubj.
                        forwardPass(leafFeaturesBatch[i].squeeze(), tf.tensor(graphViews[(b*batchSize)+i].subSubView));
                        const objObjRepresentation = modelsGCN.modelObjObj.
                        forwardPass(leafFeaturesBatch[i].squeeze(), tf.tensor(graphViews[(b*batchSize)+i].objObjView));
                        const objSubjRepresentation = modelsGCN.modelObjSubj.
                        forwardPass(leafFeaturesBatch[i].squeeze(), tf.tensor(graphViews[(b*batchSize)+i].objSubView));
                        
                        const learnedRepresentation = tf.concat([subjSubjRepresentation, objObjRepresentation, objSubjRepresentation], 1);
                        const learnedRepresentationList = tf.split(learnedRepresentation, learnedRepresentation.shape[0]);
                        const learnedRepresentationResultSet: IResultSetRepresentation = {
                          hiddenStates: learnedRepresentationList,
                          memoryCell: learnedRepresentationList.map(x=>tf.zeros(x.shape))
                        };
                        qValuesRecursive.push(modelLSTM.forwardPassRecursive(learnedRepresentationResultSet, treesToExecute[i]));
                    }
                    const executionTimesBatch: tf.Tensor[] = executionTimes.slice(b*batchSize, Math.min((b+1)*batchSize, executionTimes.length)).map(x=>tf.tensor(x));
                    const loss = tf.sum(squaredDifference(tf.concat(qValuesRecursive).squeeze(), tf.concat(executionTimesBatch).squeeze()));
                    return loss.squeeze();
                }, true);

                if (loss==null){
                    throw new Error("Received null loss");
                }
                const lossArray = loss.dataSync();
                if (lossArray.length > 1){
                    throw new Error("Loss is array with multiple elements");
                }
                episodeLoss += lossArray[0];
            }
            return episodeLoss/executionTimes.length;
        });
    };

    public trainOfflineExperienceReplay(
        partialJoinTree: number[][][],
        resultSetRepresentations: IResultSetRepresentation[], 
        executionTimes: number[], 
        model: ModelTreeLSTM, 
        batchSize: number
    ){
        return tf.tidy(()=>{
            const numBatches = Math.ceil(executionTimes.length/batchSize);
            let episodeLoss: number = 0;
            for (let b=0; b<numBatches; b++){
                const treesToExecute: number[][][] = partialJoinTree.slice(b*batchSize, Math.min((b+1)*batchSize, partialJoinTree.length));
                const resultSetsToUse: IResultSetRepresentation[] = resultSetRepresentations.slice(b*batchSize, 
                    Math.min((b+1)*batchSize, resultSetRepresentations.length));

                const loss: tf.Scalar|null = this.optimizer.minimize(()=>{
                    const qValuesRecursive: tf.Tensor[] = [];
                    for (let i=0;i<treesToExecute.length;i++){
                        qValuesRecursive.push(model.forwardPassRecursive(resultSetRepresentations[i], partialJoinTree[i]));
                    }
                    const executionTimesBatch: tf.Tensor[] = executionTimes.slice(b*batchSize, Math.min((b+1)*batchSize, executionTimes.length)).map(x=>tf.tensor(x));
                    const loss = tf.sum(squaredDifference(tf.concat(qValuesRecursive).squeeze(), tf.concat(executionTimesBatch).squeeze()));
                    return loss.squeeze();
                }, true);

                if (loss==null){
                    throw new Error("Received null loss");
                }
                const lossArray = loss.dataSync();
                if (lossArray.length > 1){
                    throw new Error("Loss is array with multiple elements");
                }
                episodeLoss += lossArray[0];
            }
            return episodeLoss/executionTimes.length;
        });
    }
    
    public pretrainOptimizerBatched(
        queryCardinalities: number[], 
        leafFeaturesTensor: tf.Tensor[], 
        graphViews: IQueryGraphViews[], 
        modelsGCN: IQueryGraphEncodingModels, 
        cardinalityPredictionLayers: IGraphCardinalityPredictionLayers
    ){
        return tf.tidy(() => {
            const loss: tf.Scalar|null = this.optimizer.minimize(()=>{
                const cardinalityPredictionsBatch: tf.Tensor[] = [];
                for (let i=0;i<leafFeaturesTensor.length;i++){

                    const cardPrediction = this.getCardinalityPrediction(
                        leafFeaturesTensor[i],
                        graphViews[i],
                        modelsGCN,
                        cardinalityPredictionLayers
                    )
                    cardinalityPredictionsBatch.push(cardPrediction);
                }
                
                const executionTimesBatch: tf.Tensor[] = queryCardinalities.map(x=>tf.tensor(x));
                const loss = tf.sum(squaredDifference(tf.concat(cardinalityPredictionsBatch).squeeze(), tf.concat(executionTimesBatch).squeeze()));
                return loss.squeeze();
            }, true);
            return loss?.arraySync()
        });
    }

    public forwardPassCardinalityPrediction(
        learnedRepresentation: tf.Tensor[], 
        cardinalityPredictionLayers: IGraphCardinalityPredictionLayers
    ){
        const averagedRepresentation = <tf.Tensor> cardinalityPredictionLayers.pooling.apply(learnedRepresentation);
        const hiddenRepresentation = <tf.Tensor> cardinalityPredictionLayers.hiddenLayer.apply(averagedRepresentation.transpose());
        const activationOutput = <tf.Tensor> cardinalityPredictionLayers.activationHiddenLayer.apply(hiddenRepresentation);
        const cardinalityPrediction = <tf.Tensor> cardinalityPredictionLayers.linearLayer.apply(activationOutput);

        return cardinalityPrediction
    }

    public getCardinalityPrediction(        
        leafFeatures: tf.Tensor, 
        graphView: IQueryGraphViews, 
        modelsGCN: IQueryGraphEncodingModels, 
        cardinalityPredictionLayers: IGraphCardinalityPredictionLayers
    ){
        // Obtain query representations from GCN forwardpasses
        const subjSubjRepresentation = modelsGCN.modelSubjSubj.
            forwardPass(leafFeatures.squeeze(), tf.tensor(graphView.subSubView));
        const objObjRepresentation = modelsGCN.modelObjObj.
            forwardPass(leafFeatures.squeeze(), tf.tensor(graphView.objObjView));
        const objSubjRepresentation = modelsGCN.modelObjSubj.
            forwardPass(leafFeatures.squeeze(), tf.tensor(graphView.objSubView));

        // Concatinate them into one representation
        const learnedRepresentation = tf.concat([subjSubjRepresentation, objObjRepresentation, objSubjRepresentation], 1);
        const learnedRepresentationList = tf.split(learnedRepresentation, learnedRepresentation.shape[0]);

        const cardinalityPrediction = this.forwardPassCardinalityPrediction(
            learnedRepresentationList, 
            cardinalityPredictionLayers
        );
        return cardinalityPrediction;
        
    }

    public meanSquaredError(prediction: tf.Tensor, actual: tf.Tensor){
        return tf.tidy(()=>{
            return tf.sum(tf.pow(tf.sub(prediction, actual),2));
        });
    }

    public meanAbsoluteError(prediction: tf.Tensor, actual: tf.Tensor){
        return tf.tidy(()=>{
            return tf.sum(tf.abs(tf.sub(prediction, actual)));
        });
    }
}

export interface IModelTrainerOfflineArgs{
    optimizer: string;
    learningRate: number;
}

export function stringLiteralArray<T extends string>(a: T[]) {
    return a;
}

export const optimizerOptions = stringLiteralArray(['adam', 'sgd']);
// type optimizerOptionsTypes = typeof optimizerOptions[number];


export interface IGraphCardinalityPredictionLayers{
    // Pooling layer to get singular graph representation
    pooling: tf.layers.Layer,
    // Hidden layer to introduce non-more complex function approx
    hiddenLayer: DenseOwnImplementation
    // Activation to introduce non-linearity
    activationHiddenLayer: tf.layers.Layer
    // Finally linear layer with no activation to get regression result
    linearLayer: DenseOwnImplementation
}