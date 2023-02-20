import * as tf from '@tensorflow/tfjs-node';

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

    public trainOfflineBatched(qValues: tf.Tensor[], executionTimes: number[], batchSize: number){
        if (qValues.length!=executionTimes.length){
            throw new Error("Got unequal number of qVales and executionTimes");
        }
        console.log(qValues[0]);
        return tf.tidy(()=>{
            const numBatches = Math.ceil(executionTimes.length/batchSize);
            let episodeLoss: number = 0;
            for (let b=0; b<numBatches;b++){
                const loss: tf.Scalar|null = this.optimizer.minimize(()=>{
                    const qValuesBatch: tf.Tensor[] = qValues.slice(b*batchSize, Math.min((b+1)*batchSize, qValues.length));
                    const executionTimesBatch: tf.Tensor[] = executionTimes.slice(b*batchSize, Math.min((b+1)*batchSize, qValues.length)).map(x=>tf.tensor(x));
                    console.log()
                    const batchLoss = this.meanSquaredError(tf.stack(qValuesBatch), tf.stack(executionTimesBatch));
                    return tf.squeeze(batchLoss);
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
    optimizer: optimizerOptionsTypes;
    learningRate: number;
}

export function stringLiteralArray<T extends string>(a: T[]) {
    return a;
}

export const optimizerOptions = stringLiteralArray(['adam', 'sgd']);
type optimizerOptionsTypes = typeof optimizerOptions[number];
