import { graphConvolutionModel } from "@comunica/actor-rdf-join-inner-multi-reinforcement-learning/lib/GraphNeuralNetwork";
import { NodeStateSpace, StateSpaceTree } from "@comunica/mediator-join-reinforcement-learning";
import * as tf from '@tensorflow/tfjs';

import { PassThrough } from "stream";

export class ModelTrainer{
    joinState: StateSpaceTree
    adjMatrix: number[][];
    features: number[];
    executionTime: number;
    model: graphConvolutionModel;
    optimizer;
    path;

    public constructor(episode: StateSpaceTree, executionTime: number, optimizer?:any){
        this.path = require('path');
        // this.adjMatrix = adjMatrix;
        // this.features = features
        this.joinState = episode;
        this.executionTime = executionTime;

        this.optimizer = (optimizer) ? optimizer: tf.train.adam(.01);

        try{
            this.loadModel();
        } catch (error){
            console.log(error);
            console.log("Something went wrong during model loading, check if it exists");
        }
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

    public async trainModel(){
        const loss = this.optimizer.minimize(() => {
            const predTensor: tf.Tensor = this.finalForwardPass();
            const y: tf.Tensor = tf.fill([predTensor.shape[0], 1], this.executionTime);
            const loss = tf.losses.meanSquaredError(y, predTensor);
            loss.data().then(l => console.log('Loss', l[0]));
            return loss;
        });
    }

    private finalForwardPass(): tf.Tensor<tf.Rank>{
        const finalJoin: number[] = this.joinState.nodesArray.filter(function(x) {
            if (!x.joined){
                return x;
            }
        }).map(x => x.id);
        const newParent: NodeStateSpace = new NodeStateSpace(this.joinState.numNodes, 0);
        this.joinState.addParent(finalJoin, newParent);

        const adjTensor = tf.tensor2d(this.joinState.adjacencyMatrix);

        let featureMatrix = [];
        for(var i=0;i<this.joinState.nodesArray.length;i++){
          featureMatrix.push(this.joinState.nodesArray[i].cardinality);
        }
    
        /* Execute forward pass */
        const forwardPassOutput = this.model.forwardPass(tf.tensor2d(featureMatrix, [this.joinState.numNodes,1]), adjTensor) as tf.Tensor;
        const predArray = forwardPassOutput.slice(this.joinState.numLeaveNodes, this.joinState.numNodes - this.joinState.numLeaveNodes);  
        return predArray
    }
}

