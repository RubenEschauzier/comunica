import { IQueryGraphViews, IResultSetRepresentation } from '@comunica/mediator-join-reinforcement-learning';
import * as tf from '@tensorflow/tfjs-node';
import { squaredDifference } from '@tensorflow/tfjs-node';
import { ModelTreeLSTM } from './treeLSTM';
import { GraphConvolutionModel } from '@comunica/mediator-join-reinforcement-learning/lib/GraphConvolution';
import { IQueryGraphEncodingModels } from './instanceModel';

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
                        forwardPass(leafFeaturesBatch[i].squeeze(), tf.tensor(graphViews[i].subSubView));
                        const objObjRepresentation = modelsGCN.modelObjObj.
                        forwardPass(leafFeaturesBatch[i].squeeze(), tf.tensor(graphViews[i].objObjView));
                        const objSubjRepresentation = modelsGCN.modelObjSubj.
                        forwardPass(leafFeaturesBatch[i].squeeze(), tf.tensor(graphViews[i].objSubView));

                        const learnedRepresentation = tf.concat([subjSubjRepresentation, objObjRepresentation, objSubjRepresentation], 1);
                        const learnedRepresentationList = tf.split(learnedRepresentation, learnedRepresentation.shape[0]);
                        const learnedRepresentationResultSet: IResultSetRepresentation = {
                          hiddenStates: learnedRepresentationList,
                          memoryCell: learnedRepresentationList.map(x=>tf.zeros(x.shape))
                        };
                        qValuesRecursive.push(modelLSTM.forwardPassRecursive(learnedRepresentationResultSet, partialJoinTree[i]));
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
    public trainOfflineExperienceReplay(partialJoinTree: number[][][], resultSetRepresentations: IResultSetRepresentation[], 
        executionTimes: number[], model: ModelTreeLSTM, batchSize: number){
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

    public validateModel(){
        
    }

    public meanSquaredError(prediction: tf.Tensor, actual: tf.Tensor){
        return tf.tidy(()=>{
            return tf.sum(tf.pow(tf.sub(prediction, actual),2));
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
