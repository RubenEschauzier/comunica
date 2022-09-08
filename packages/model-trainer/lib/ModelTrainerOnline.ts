import { graphConvolutionModel } from "@comunica/actor-rdf-join-inner-multi-reinforcement-learning/lib/GraphNeuralNetwork";
import { NodeStateSpace, StateSpaceTree } from "@comunica/mediator-join-reinforcement-learning";
import * as tf from '@tensorflow/tfjs';

import { PassThrough } from "stream";
import { modelHolder } from "./modelHolder";

export class ModelTrainer{
    model: graphConvolutionModel;
    // optimizer: tf.AdamOptimizer;
    optimizer: tf.SGDOptimizer;

    path;

    public constructor(modelHolder: modelHolder){
        this.path = require('path');
        this.optimizer = tf.train.sgd(.05);
        this.model = modelHolder.getModel();
    }

    public loadModel(){
        this.model = new graphConvolutionModel();
        this.model.loadModel();
    }

    public saveModel(){
        this.model.saveModel();
    }
    
    private getModelDirectory(){          
        const modelDir: string = this.path.join(__dirname, '../../actor-rdf-join-inner-multi-reinforcement-learning/model');
        return modelDir;
    }


    public async trainModel(joinState: StateSpaceTree, executionTime: number){
        const loss = this.optimizer.minimize(() => {
            const predTensor: tf.Tensor = this.finalForwardPass(joinState);
            const y: tf.Tensor = tf.fill([predTensor.shape[0], 1], executionTime);
            const loss = tf.losses.meanSquaredError(y, predTensor);
            const lossSqueezed: tf.Scalar = tf.squeeze(loss);
            // loss.data().then(l => console.log('Loss', l[0]));
            return lossSqueezed;
        });
    }

    public async trainModeloffline(adjacencyMatrixes: number[][][], featureMatrix: number[][], executionTime: number[]): Promise<number>{
                
        const loss =  this.optimizer.minimize(() => {

            const y: tf.Tensor = tf.tensor(executionTime, [executionTime.length,1]);
            // const valuePredictions: tf.Tensor = tf.zeros([executionTime.length,1]);
            const valuePredictions: tf.Tensor[] = []
            // console.log(y);
            // console.log(await y.data());
            
            for (let i = 0;i<adjacencyMatrixes.length;i++){
                const adjTensor = tf.tensor2d(adjacencyMatrixes[i]);

                /* Pretend we don't know the prediction output of our join node for training purposes*/
                featureMatrix[i][featureMatrix[i].length-1] = 0;
                const forwardPassOutput: tf.Tensor[] = this.model.forwardPass(tf.tensor2d(featureMatrix[i], [adjacencyMatrixes[i].length,1]), adjTensor) as tf.Tensor[];

                const joinValuePrediction: tf.Tensor = forwardPassOutput[0].slice([forwardPassOutput[0].shape[0]-1, 0]);
                const y: tf.Tensor = tf.fill([joinValuePrediction.shape[0], 1], executionTime[i]);    

                valuePredictions.push(joinValuePrediction);
            }
            const predictionTensor: tf.Tensor = tf.concat(valuePredictions);
            const loss = tf.losses.meanSquaredError(y, predictionTensor);
            const scalarLoss: tf.Scalar = tf.squeeze(loss);
            return scalarLoss
            }, true)!;
        const episodeLoss: number = (await loss.data())[0]
        console.log(episodeLoss)


        return episodeLoss;
    }

    private finalForwardPass(joinState: StateSpaceTree): tf.Tensor<tf.Rank>{
        const finalJoin: number[] = joinState.nodesArray.filter(function(x) {
            if (!x.joined){
                return x;
            }
        }).map(x => x.id);
        const newParent: NodeStateSpace = new NodeStateSpace(joinState.numNodes, 0);
        joinState.addParent(finalJoin, newParent);

        const adjTensor = tf.tensor2d(joinState.adjacencyMatrix);

        let featureMatrix = [];
        for(var i=0;i<joinState.nodesArray.length;i++){
          featureMatrix.push(joinState.nodesArray[i].cardinality);
        }
    
        /* Execute forward pass */
        const forwardPassOutput = this.model.forwardPass(tf.tensor2d(featureMatrix, [joinState.numNodes,1]), adjTensor) as tf.Tensor[];
        const predArray = forwardPassOutput[0].slice(joinState.numLeaveNodes, joinState.numNodes - joinState.numLeaveNodes);  
        return predArray
    }
}

