import { IResultSetRepresentation } from '@comunica/mediator-join-reinforcement-learning';
import * as tf from '@tensorflow/tfjs-node';
import { losses, squaredDifference } from '@tensorflow/tfjs-node';
import { Console } from 'console';
import { ModelTreeLSTM } from './treeLSTM';

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
        qValues: tf.Tensor[], executionTimes: number[], model: ModelTreeLSTM, batchSize: number){
        Error.stackTraceLimit = Infinity;
        console.log("Start offline traing")
        if (qValues.length!=executionTimes.length){
            throw new Error("Got unequal number of qVales and executionTimes");
        }
        // const testTensor1 = tf.tensor([[1, 4]]);
        // const testTensor2 = tf.tensor([[2, 4]]);
        // const testTensor3 = tf.tensor([[4, 2]]);
        // const testTensor4 = tf.tensor([[1, 4]]);
        // const testTensor5 = tf.tensor([[3, 6]]);
        // const testTensor6 = tf.tensor([[4, 3]]);
        // const testTensor7 = tf.tensor([[4, 4]]);
        // const testTensor8 = tf.tensor([[2, 4]]);
        // const testTensor9 = tf.tensor([[4, 4]]);

        // tf.variableGrads(()=>{
        //     const qValuesRecursive: tf.Tensor[] = []

        //     for (const tree of partialJoinTree){
        //         qValuesRecursive.push(model.forwardPassRecursive(resultSetRepresentation, tree));
        //     }
        //     const loss = tf.sum(squaredDifference(tf.concat(qValuesRecursive).squeeze(), tf.tensor(executionTimes)));
        //     // const loss = this.meanSquaredError(tf.stack(qValuesRecursive).squeeze(), tf.stack(executionTimes));
        //     return loss.squeeze();
        // });
        console.log("executionTimes")
        console.log(executionTimes);
        return tf.tidy(()=>{
            const numBatches = Math.ceil(executionTimes.length/batchSize);
            let episodeLoss: number = 0;
            for (let b=0; b<numBatches;b++){
                const treesToExecute: number[][][] = partialJoinTree.slice(b*batchSize, Math.min((b+1)*batchSize, qValues.length));
                const loss: tf.Scalar|null = this.optimizer.minimize(()=>{
                    const qValuesRecursive: tf.Tensor[] = [];
                    for (const tree of treesToExecute){
                        qValuesRecursive.push(model.forwardPassRecursive(resultSetRepresentation, tree));
                    }

                    const qValuesBatch: tf.Tensor[] = qValues.slice(b*batchSize, Math.min((b+1)*batchSize, qValues.length));
                    if (qValuesBatch.length!=qValuesRecursive.length){
                        throw new Error("The recursively obtained qValues list is not equal in length to query execution qValue list");
                    }
                    for (let i=0; i<qValuesBatch.length;i++){
                        if (!tf.equal(qValuesBatch[i], qValuesRecursive[i])){
                            throw new Error("The recursive and query execution qValues are not equal");
                        }
                    }
                    const executionTimesBatch: tf.Tensor[] = executionTimes.slice(b*batchSize, Math.min((b+1)*batchSize, qValues.length)).map(x=>tf.tensor(x));

                    const loss = tf.sum(squaredDifference(tf.concat(qValuesRecursive).squeeze(), tf.concat(executionTimesBatch).squeeze()));
                    return loss.squeeze();
        
                    // const batchLoss = this.meanSquaredError(tf.stack(qValuesRecursive).squeeze(), tf.stack(executionTimesBatch));
                    // const scalarLoss: tf.Scalar = tf.squeeze(batchLoss);
                    // return scalarLoss;
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
type optimizerOptionsTypes = typeof optimizerOptions[number];
