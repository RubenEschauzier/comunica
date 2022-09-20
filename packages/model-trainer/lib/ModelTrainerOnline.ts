import { graphConvolutionModel } from "@comunica/actor-rdf-join-inner-multi-reinforcement-learning/lib/GraphNeuralNetwork";
import { NodeStateSpace, StateSpaceTree } from "@comunica/mediator-join-reinforcement-learning";
import * as tf from '@tensorflow/tfjs';

import { PassThrough } from "stream";
import { modelHolder } from "./modelHolder";

export class ModelTrainer{
    model: graphConvolutionModel;
    // optimizer: tf.AdamOptimizer;
    optimizer: tf.AdamOptimizer;

    path;

    public constructor(modelHolder: modelHolder){
        this.path = require('path');
        this.optimizer = tf.train.adam(.05);
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

    public async trainModeloffline(adjacencyMatrixes: number[][][], featureMatrix: number[][], executionTime: number[], numEntries: number): Promise<number>{
        // const yArray = [10, 50, 12, 33];
        // const featureMatrixPre = [[1,2,3,4,5,6],[7,8,9,10,11,12], [1,2,3,4,5,7], [7,8,9,10,11,12]];
        // const adjacencyMatrixesPre = [[[1,1,0,0,0,0],[1,1,1,0,0,0], [0,1,1,1,0,0], [0,0,1,1,1,0], [0,0,0,1,1,1], [0,0,0,0,1,1]],
        // [[1,1,0,0,0,0],[1,1,1,0,0,0], [0,1,1,1,0,0], [0,0,1,1,1,0], [0,0,0,1,1,1], [0,0,0,0,1,1]],
        // [[1,1,0,0,0,0],[1,1,1,0,0,0], [0,1,1,1,0,0], [0,0,1,1,1,0], [0,0,0,1,1,1], [0,0,0,0,1,1]],
        // [[1,1,0,0,0,0],[1,1,1,0,0,0], [0,1,1,1,0,0], [0,0,1,1,1,0], [0,0,0,1,1,1], [0,0,0,0,1,1]]];
        
        // const yPre = tf.tensor(yArray);
        return tf.tidy(() => {
            const yPre: tf.Tensor = tf.fill([adjacencyMatrixes.length, 1], 40);
            // Hardcoded for ease of use, denotes the value to clip "normalization" by
            const numGraphLayers = 4;
            const loss =  this.optimizer.minimize(() => {
    
                const y: tf.Tensor = tf.tensor(executionTime, [executionTime.length,1]);
                const nodeDegreeArray: tf.Tensor[] = [];
                // const valuePredictions: tf.Tensor = tf.zeros([executionTime.length,1]);
                const valuePredictions: tf.Tensor[] = []
                const policyPredictions: tf.Tensor[] = []
                
                for (let i = 0;i<adjacencyMatrixes.length;i++){
                    const adjTensor = tf.tensor2d(adjacencyMatrixes[i]);
                    let connectionTensor = tf.matMul(adjTensor, adjTensor)
                    for (let j = 1; j<numEntries; j++){
                        connectionTensor = tf.matMul(connectionTensor, adjTensor).clipByValue(0,1);
                    }
                    // console.log("Adjacency Tensor");
                    // adjTensor.print();
                    // console.log("Connection Tensor");
                    // connectionTensor.print();
                    const nodeDegree: tf.Tensor = connectionTensor.sum(1);
                    // console.log("Here is node degree ");
                    // nodeDegree.print();
    
                    const degreeLastNode = nodeDegree.slice([nodeDegree.shape[0]-1]);
                    // console.log("Here is the degree of last node");
                    // degreeLastNode.print()
                    nodeDegreeArray.push(degreeLastNode)  
                    /* Pretend we don't know the prediction output of our join node for training purposes*/
                    
                    // const forwardPassOutput: tf.Tensor[] = this.model.forwardPass(tf.tensor2d(featureMatrix[i], [adjacencyMatrixes[i].length,1]), adjTensor) as tf.Tensor[];
                    const forwardPassOutput: tf.Tensor[] = this.model.forwardPassTest(tf.tensor2d(featureMatrix[i], [adjacencyMatrixes[i].length,1]), adjTensor) as tf.Tensor[];
                    const joinValuePrediction: tf.Tensor = forwardPassOutput[0].slice([forwardPassOutput[0].shape[0]-1, 0]);
                    const joinPolicyPrediction: tf.Tensor = forwardPassOutput[1].slice([forwardPassOutput[0].shape[0]-1, 0]);
    
    
                    valuePredictions.push(joinValuePrediction);
                    policyPredictions.push(joinPolicyPrediction);
                    
                    // Here we "normalise" the y by the amount of preceding joins
                }
                const nodeDegreeTensor = tf.sub(tf.stack(nodeDegreeArray),1).clipByValue(0, (numEntries-1)*2);
                // nodeDegreeTensor.print();
                const predictionTensor: tf.Tensor = tf.concat(valuePredictions);
                const policyTensor: tf.Tensor = tf.concat(policyPredictions);
                // console.log("Prediction:");
                // predictionTensor.print();
                console.log("Y:");
                y.print();
                // console.log("Norm factor");
                // tf.div(tf.sub(numGraphLayers*3, nodeDegreeTensor),3).print()
                const normY = tf.div(y, tf.sub((numEntries-1)*2, nodeDegreeTensor));
                console.log("Prediction:")
                predictionTensor.print()
                // console.log("Policy prediction");
                // policyTensor.print()
                // console.log("Division constant:");
                // tf.div(tf.sub(numGraphLayers*3, nodeDegreeTensor),3).print()
                // console.log("Norm y");
                // normY.print();
                
                const loss = tf.losses.meanSquaredError(y, predictionTensor);
                const scalarLoss: tf.Scalar = tf.squeeze(loss);
    
                return scalarLoss
                }, true)!;
            const episodeLoss: number = (loss.dataSync())[0]
            return episodeLoss;
        });
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

