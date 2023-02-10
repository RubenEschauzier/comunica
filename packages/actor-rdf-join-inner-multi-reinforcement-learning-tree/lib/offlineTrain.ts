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
        const testTensor1 = tf.tensor([[0],[4],[4],[4]]);
        const testTensor2 = tf.tensor([[4],[4],[5],[1]]);
        const testTensor3 = tf.tensor([[4],[2],[4],[4]]);
        const testTensor4 = tf.tensor([[1],[4],[2],[3]]);
        const testTensor5 = tf.tensor([[4],[6],[4],[1]]);
        const testTensor6 = tf.tensor([[4],[3],[7],[4]]);
        const testTensor7 = tf.tensor([[4],[4],[4],[4]]);
        const testTensor8 = tf.tensor([[4],[4],[4],[4]]);
        const testTensor9 = tf.tensor([[4],[4],[4],[4]]);

        tf.variableGrads(()=>{
            const qValuesRecursive: tf.Tensor[] = []
            // const outputTestBinary = model.binaryTreeLSTMLayer[0].call(tf.stack([testTensor1,testTensor2,testTensor3,testTensor4,testTensor5]))
            // GRADIENT GOES WRONG HERE, IT SEEMS DUE TO THE STACKED INPUTS, MAYBE DIFFERENT INPUT TYPE?
            const output = model.childSumTreeLSTMLayer[0].callTestWeird([testTensor1, tf.stack([testTensor2, testTensor3]), tf.stack([testTensor4, testTensor5])]);
            const outputNormal = model.childSumTreeLSTMLayer[0].call([testTensor1, tf.stack([testTensor2, testTensor3]), tf.stack([testTensor4, testTensor5])]);
            output[0].print();  output[1].print();
            outputNormal[0].print(); outputNormal[1].print();

            // const outputTestChildSum = model.childSumTreeLSTMLayer[0].callTestGradientFix([testTensor1, testTensor2, testTensor3, testTensor4, testTensor5]);
            const outputTest = model.qValueNetwork.forwardPassFromConfig(output[0]);
            // for (const tree of partialJoinTree){
            //     // qValuesRecursive.push(model.forwardPassRecursive(resultSetRepresentation, tree));
            //     qValuesRecursive.push(model.forwardPass(resultSetRepresentation, tree[0]).qValue);
            //     break;
            // }
            // console.log(tf.stack(qValuesRecursive).squeeze().shape);
            // console.log(tf.stack(executionTimes).squeeze().shape);
            // console.log(tf.tensor(executionTimes).dataSync());
            // console.log(tf.tensor(executionTimes).shape);
            console.log(outputTest.shape)
            const test = squaredDifference(outputTest, tf.tensor(executionTimes[0]));
            // const loss = this.meanSquaredError(tf.stack(qValuesRecursive).squeeze(), tf.stack(executionTimes));
            return test.squeeze();
        });
        console.log("Done computing gradients")
        // return tf.tidy(()=>{
        //     const numBatches = Math.ceil(executionTimes.length/batchSize);
        //     let episodeLoss: number = 0;
        //     for (let b=0; b<numBatches;b++){
        //         const treesToExecute: number[][][] = partialJoinTree.slice(b*batchSize, Math.min((b+1)*batchSize, qValues.length));
        //         const loss: tf.Scalar|null = this.optimizer.minimize(()=>{
        //             const qValuesRecursive: tf.Tensor[] = [];
        //             for (const tree of treesToExecute){
        //                 qValuesRecursive.push(model.forwardPassRecursive(resultSetRepresentation, tree));
        //             }

        //             const qValuesBatch: tf.Tensor[] = qValues.slice(b*batchSize, Math.min((b+1)*batchSize, qValues.length));
        //             if (qValuesBatch.length!=qValuesRecursive.length){
        //                 throw new Error("The recursively obtained qValues list is not equal in length to query execution qValue list");
        //             }
        //             for (let i=0; i<qValuesBatch.length;i++){
        //                 if (!tf.equal(qValuesBatch[i], qValuesRecursive[i])){
        //                     throw new Error("The recursive and query execution qValues are not equal");
        //                 }
        //             }

        //             const executionTimesBatch: tf.Tensor[] = executionTimes.slice(b*batchSize, Math.min((b+1)*batchSize, qValues.length)).map(x=>tf.tensor(x));
        //             const batchLoss = this.meanSquaredError(tf.stack(qValuesRecursive).squeeze(), tf.stack(executionTimesBatch));
        //             const scalarLoss: tf.Scalar = tf.squeeze(batchLoss);
        //             return scalarLoss;
        //         }, true);
        //         console.log(loss);
        //         if (loss==null){
        //             throw new Error("Received null loss");
        //         }
        //         const lossArray = loss.dataSync();
        //         if (lossArray.length > 1){
        //             throw new Error("Loss is array with multiple elements");
        //         }
        //         episodeLoss += lossArray[0];
        //     }
        //     return episodeLoss/executionTimes.length;
        // });
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
