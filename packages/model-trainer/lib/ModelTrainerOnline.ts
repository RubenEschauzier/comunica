import { enumStringBody } from "@babel/types";
import { graphConvolutionModel } from "@comunica/actor-rdf-join-inner-multi-reinforcement-learning/lib/GraphNeuralNetwork";
import { NodeStateSpace, StateSpaceTree } from "@comunica/mediator-join-reinforcement-learning";
import * as tf from '@tensorflow/tfjs';

import { PassThrough } from "stream";
import { modelHolder } from "./modelHolder";

export class ModelTrainer{
    model: graphConvolutionModel;
    // optimizer: tf.SGDOptimizer;
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

    public async trainModeloffline(adjacencyMatrixes: number[][][], featureMatrix: number[][][], executionTime: number[], numEntries: number): Promise<number>{
        const yArray = [40, 40, 20, 20];
        const adjacencyMatrixesPre = [ [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [1, 1, 0, 1, 0], [0, 0, 1, 1, 1]],
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 1, 0, 1]], 
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 1, 1]],
        [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 0, 0, 1, 1]]]

        const featureMatrixesPre = [[[100, 1, 0, 1, 0,0,0, 0], [200, 1, 1, 1, 0,0,0, 0], [50, 1, 0, 0, 0,0,0, 0], [50, 0, 0, 0, 0,0,0, 5], [0, 0, 0, 0, 0,0,0, 6]],
        [[100, 1, 0, 1, 0,0,0, 0], [200, 1, 1, 1, 0,0,0, 0], [50, 1, 0, 0, 0,0,0, 0], [0, 0, 0, 0, 0,0,0, 5]],
        [[100, 1, 0, 1, 0,0,0, 0], [200, 1, 1, 1, 0,0,0, 0], [50, 1, 0, 0, 0,0,0, 0], [0, 0, 0, 0, 0,0,0, 4]],
        [[100, 1, 0, 1, 0,0,0, 0], [200, 1, 1, 1, 0,0,0, 0], [50, 1, 0, 0, 0,0,0, 0], [15, 0, 0, 0, 0,0,0, 4], [0, 0, 0, 0, 0,0,0, 6]]]
        // const featureMatrixPre = [[1,2,3,4,5,6],[7,8,9,10,11,12], [1,2,3,4,5,7], [7,8,9,10,11,12]];
        // const adjacencyMatrixesPre = [[[1,1,0,0,0,0],[1,1,1,0,0,0], [0,1,1,1,0,0], [0,0,1,1,1,0], [0,0,0,1,1,1], [0,0,0,0,1,1]],
        // [[1,1,0,0,0,0],[1,1,1,0,0,0], [0,1,1,1,0,0], [0,0,1,1,1,0], [0,0,0,1,1,1], [0,0,0,0,1,1]],
        // [[1,1,0,0,0,0],[1,1,1,0,0,0], [0,1,1,1,0,0], [0,0,1,1,1,0], [0,0,0,1,1,1], [0,0,0,0,1,1]],
        // [[1,1,0,0,0,0],[1,1,1,0,0,0], [0,1,1,1,0,0], [0,0,1,1,1,0], [0,0,0,1,1,1], [0,0,0,0,1,1]]];
        
        
        const yPre = tf.tensor(yArray);
        return tf.tidy(() => {
            // const yPre: tf.Tensor = tf.fill([adjacencyMatrixes.length, 1], 40);
            const loss =  this.optimizer.minimize(() => {
    
                const y: tf.Tensor = tf.tensor(executionTime, [executionTime.length,1]);
                const nodeDegreeArray: tf.Tensor[] = [];
                // const valuePredictions: tf.Tensor = tf.zeros([executionTime.length,1]);
                const valuePredictions: tf.Tensor[] = []
                const policyPredictions: tf.Tensor[] = []

                const totalLoss = 0;
                for (let i = 0;i<adjacencyMatrixesPre.length;i++){
                    // adjacencyMatrixes[i][adjacencyMatrixes[i].length-1][adjacencyMatrixes[i].length-1] = 0;
                    const adjTensorPre = tf.tensor2d(adjacencyMatrixesPre[i]);
                    const featureMatrixPre = tf.tensor2d(featureMatrixesPre[i]);

                    // const adjTensor = tf.tensor2d(adjacencyMatrixes[i]);
                    // let connectionTensor = tf.matMul(adjTensor, adjTensor)
                    // for (let j = 1; j<numEntries; j++){
                    //     connectionTensor = tf.matMul(connectionTensor, adjTensor).clipByValue(0,1);
                    // }
                    // const nodeDegree: tf.Tensor = connectionTensor.sum(1);    
                    // const degreeLastNode = nodeDegree.slice([nodeDegree.shape[0]-1]);
                    // nodeDegreeArray.push(degreeLastNode)  
                    /* Pretend we don't know the prediction output of our join node for training purposes*/
                    const forwardPassOutput: tf.Tensor[] = this.model.forwardPassTest(tf.tensor2d(featureMatrixesPre[i], [adjacencyMatrixesPre[i].length, featureMatrixesPre[i][0].length]), adjTensorPre) as tf.Tensor[];
                    const joinValuePrediction: tf.Tensor = forwardPassOutput[0].slice([forwardPassOutput[0].shape[0]-1, 0]);
                    const joinPolicyPrediction: tf.Tensor = forwardPassOutput[1].slice([forwardPassOutput[0].shape[0]-1, 0]);
                        
    
                    valuePredictions.push(joinValuePrediction);
                    policyPredictions.push(joinPolicyPrediction);
                    
                }
                // const nodeDegreeTensor = tf.sub(tf.stack(nodeDegreeArray),1).clipByValue(0, (numEntries-1)*2);

                const predictionTensor: tf.Tensor = tf.concat(valuePredictions);
                // const policyTensor: tf.Tensor = tf.concat(policyPredictions);
                // const normY = tf.div(y, tf.sub((numEntries-1)*2, nodeDegreeTensor));
                console.log("Prediction:")
                console.log(predictionTensor.dataSync());
                console.log("Y Actual");
                console.log(yPre.dataSync());

                const loss = tf.losses.meanSquaredError(yPre, predictionTensor.squeeze());
                const scalarLoss: tf.Scalar = tf.squeeze(loss);
    
                return scalarLoss
                }, true)!;

            const episodeLoss: number = (loss.dataSync())[0]
            return episodeLoss;
        });
    }
    public getStd(inputTensor: tf.Tensor, means: tf.Tensor, featureSize: number){
        const newShape: number[] = [inputTensor.shape[0] / featureSize, featureSize];
        // Reshape flat tensor to represent all features of nodes in sample
        // NOTE: May want to only include every node once, else big effect of leaf nodes on this procedure 
        const reshapedTensor: tf.Tensor = inputTensor.reshape(newShape);
        const n: number = reshapedTensor.shape[0];

        // Sample standard deviation of input features
        const std: tf.Tensor = tf.sqrt(tf.div(tf.sum(tf.square(tf.sub(reshapedTensor, means)), 0, false), n));
        // tf.moments(reshapedTensor,[0], true).variance.print()
        return std
    }

    public async trainModelOfflineBatched(adjacencyMatrixes: number[][][], featureMatrix: number[][][], executionTime: number[], batchSize: number): Promise<number>{
        const featureSize: number = 8;

        const yArray = [40, 40, 20, 20];
        const adjacencyMatrixesPre = [ [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [1, 1, 0, 1, 0], [0, 0, 1, 1, 0]],
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 1, 0, 0]], 
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 1, 0]],
        [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 0, 0, 1, 0]]]

        const featureMatrixesPre = [[[100, 1, 0, 1, 0,0,0, 0], [200, 1, 1, 1, 0,0,0, 0], [50, 1, 0, 0, 1,1,1, 0], [50, 2, 1, 2, 0,0,0, 5], [0, 3, 1, 2, 0,0,0, 6]],
        [[100, 1, 0, 1, 0,0,0, 0], [200, 1, 1, 1, 0,0,0, 0], [50, 1, 0, 0, 1,1,1, 0], [0, 2, 1, 2, 0,0,0, 5]],
        [[100, 1, 0, 1, 0,0,0, 0], [200, 1, 1, 1, 0,0,0, 0], [50, 1, 0, 0, 1,1,1, 0], [0, 2, 1, 1, 0,0,0, 4]],
        [[100, 1, 0, 1, 0,0,0, 0], [200, 1, 1, 1, 0,0,0, 0], [50, 1, 0, 0, 1,1,1, 0], [15, 2, 1, 1, 0,0,0, 4], [0, 3, 1, 2, 0,0,0, 6]]]        

        const flatFeatureMatrixesPreTensor: tf.Tensor = tf.tensor(featureMatrixesPre.flat(2));
        const newShape: number[] = [flatFeatureMatrixesPreTensor.shape[0] / featureSize, featureSize];
        const reshapedTensor: tf.Tensor = flatFeatureMatrixesPreTensor.reshape(newShape);

        const meanFeatures: tf.Tensor = flatFeatureMatrixesPreTensor.reshape([flatFeatureMatrixesPreTensor.shape[0] / featureSize, featureSize]).mean(0, false);
        const std: tf.Tensor = this.getStd(flatFeatureMatrixesPreTensor, meanFeatures, featureSize);
        const scaledFeaturesTensor: tf.Tensor = tf.div(tf.sub(reshapedTensor, meanFeatures), std);
        const scaledFeatures = await scaledFeaturesTensor.array() as number[][];

        let startIndex: number = 0;
        const scaledFeaturesMatrixesPreReq: number[][][] = []
        for (let m=0;m<featureMatrixesPre.length;m++){
            const numNodesFeature = featureMatrixesPre[m].length;
            scaledFeaturesMatrixesPreReq.push(scaledFeatures.slice(startIndex, startIndex+numNodesFeature))
            startIndex += numNodesFeature
        }
        const scaledFeaturesMatrixesPre: number[][][] = []

        for (let m=0;m<featureMatrixesPre.length;m++){
            const episode: number[][] = []
            for (let n=0;n<featureMatrixesPre[m].length;n++){
                const toPush = featureMatrixesPre[m][n];
                
                toPush[0] = scaledFeaturesMatrixesPreReq[m][n][0];
                toPush[7] = scaledFeaturesMatrixesPreReq[m][n][7];

                episode.push(toPush)
            }
            scaledFeaturesMatrixesPre.push(episode);
        }


        batchSize=2;
        let yPre = tf.tensor(yArray);
        yPre = tf.div(tf.sub(yPre,yPre.mean(0, true)), tf.moments(yPre).variance.sqrt())
        let totalLoss: number = 0;

        return tf.tidy(() => {
            for (let b=0; b<Math.floor(adjacencyMatrixesPre.length/batchSize) ; b++){
                const loss = this.optimizer.minimize(()=>{
                    const valuePredictions: tf.Tensor[] = []
                    const policyPredictions: tf.Tensor[] = []
    
                    for (let i = 0;i<batchSize;i++){
                        const adjTensorPre = tf.tensor2d(adjacencyMatrixesPre[b*batchSize+i]);

                        /* Pretend we don't know the prediction output of our join node for training purposes*/
                        const forwardPassOutput: tf.Tensor[] = this.model.forwardPassTest(tf.tensor2d(scaledFeaturesMatrixesPre[b*batchSize+i], 
                            [adjacencyMatrixesPre[b*batchSize+i].length, scaledFeaturesMatrixesPre[b*batchSize+i][0].length]), adjTensorPre) as tf.Tensor[];

                        const joinValuePrediction: tf.Tensor = forwardPassOutput[0].slice([forwardPassOutput[0].shape[0]-1, 0]);
                        const joinPolicyPrediction: tf.Tensor = forwardPassOutput[1].slice([forwardPassOutput[0].shape[0]-1, 0]);
                            
        
                        valuePredictions.push(joinValuePrediction);
                        policyPredictions.push(joinPolicyPrediction);
                        
                    }
    
                    const predictionTensor: tf.Tensor = tf.concat(valuePredictions);
                    console.log("TEST -- Batch Result:")
                    predictionTensor.print();
                    yPre.slice(b*batchSize, batchSize).print();
                    const loss = tf.losses.meanSquaredError(yPre.slice(b*batchSize, batchSize).squeeze(), predictionTensor.squeeze());
                    const scalarLoss: tf.Scalar = tf.squeeze(loss);
        
                    return scalarLoss
    
                }, true)!;
                totalLoss += loss.dataSync()[0];
            }
            return totalLoss;
        }); 
    }

    private finalForwardPass(joinState: StateSpaceTree): tf.Tensor<tf.Rank>{
        const finalJoin: number[] = joinState.nodesArray.filter(function(x) {
            if (!x.joined){
                return x;
            }
        }).map(x => x.id);

        const newParent: NodeStateSpace = new NodeStateSpace(joinState.numNodes, [0,0,0,0,0,0,0,1]);
        joinState.addParent(finalJoin, newParent);

        const adjTensor = tf.tensor2d(joinState.adjacencyMatrix);

        let featureMatrix = [];
        for(let i=0;i<joinState.nodesArray.length;i++){
            featureMatrix.push(joinState.nodesArray[i].features);
        }

        const cardinalities: number[] = featureMatrix.map((x) => x[0]);
        const maxCardinality: number = Math.max(...cardinalities);
        const minCardinality: number = Math.min(...cardinalities);
  
        for (let i=0; i<featureMatrix.length;i++){
          featureMatrix[i][0] = (featureMatrix[i][0]-minCardinality) / (maxCardinality - minCardinality)
        }
  
        // for(var i=0;i<joinState.nodesArray.length;i++){
        //   featureMatrix.push(joinState.nodesArray[i].cardinality);
        // }
    
        /* Execute forward pass */
        const forwardPassOutput = this.model.forwardPass(tf.tensor2d(featureMatrix, [joinState.numNodes,1]), adjTensor) as tf.Tensor[];
        const predArray = forwardPassOutput[0].slice(joinState.numLeaveNodes, joinState.numNodes - joinState.numLeaveNodes);  
        return predArray
    }
}

