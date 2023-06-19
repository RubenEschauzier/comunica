import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as path from 'path';
import { DenseOwnImplementation } from '@comunica/actor-rdf-join-inner-multi-reinforcement-learning-tree/lib/fullyConnectedLayers';

// This is tensorflow for cpu, if we were to train and it takes long we can change to gpu probably.
// https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-62acf5b143d0

export class graphConvolutionLayer extends tf.layers.Layer{
    activationLayer: any;
    mInput: tf.Tensor2D;
    inputSize: number;
    outputSize: number;
    mAdjacency: tf.Tensor;
    mIdentity: tf.Tensor;
    mAdjacencyHat: tf.Tensor;
    mWeights: tf.Variable;
    heInitTerm: tf.Tensor;
    testWeights: tf.layers.Layer;
    modelDirectory: string;

    public static className: string = 'graphConvolutionLayer'; 

    constructor(inputSize: number, outputSize: number, activationName: string, layerName?: string) {
        super({});
        this.modelDirectory = this.getModelDirectory();
        // Define the activation layer used in this layer
        this.activationLayer = tf.layers.activation(<any>{activation: activationName});
        // Define input feature size
        this.inputSize = inputSize;
        // Define output feature size, which is size of node representations
        this.outputSize = outputSize;
        // Trainable weights of convolution
        this.heInitTerm = tf.sqrt(tf.div(tf.tensor(2), tf.tensor(outputSize)));
        this.mWeights = tf.variable(tf.mul(tf.randomNormal([this.inputSize, this.outputSize],0,1),this.heInitTerm), true,
        layerName);
        this.heInitTerm.dispose();
            

    }
    public async loadWeights(loadPath: string): Promise<void>{
        const fileLocation = path.join(this.modelDirectory, loadPath);
        const weights = await this.readWeightFile(fileLocation);
        this.mWeights.assign(weights);
    }

    private readWeightFile(fileLocationWeights: string){
        const data = fs.readFileSync(fileLocationWeights, 'utf8');
        const weightArray: number[][] = JSON.parse(data);
        const weights: tf.Tensor = tf.tensor(weightArray);
        return weights;
    }

    public saveWeights(savePath: string){
        const weights =  this.mWeights.arraySync();
        const fileLocation = path.join(this.modelDirectory, savePath);
        fs.writeFileSync(fileLocation, JSON.stringify(weights));
    }

    call(input: tf.Tensor,  mAdjacency: tf.Tensor, kwargs?: any): tf.Tensor {
        return tf.tidy(() => {
            // Get inverted square of node degree diagonal matrix 
            const mD: tf.Tensor2D = tf.sum(mAdjacency, 1);
            const mDInv: tf.Tensor = tf.diag(tf.rsqrt(mD));

            // Normalised adjecency matrix, we perform this is initialisation to not compute it in the call
            const mAdjacencyHat: tf.Tensor2D = tf.matMul(tf.matMul(mDInv, mAdjacency), mDInv);

            // Tensor that denotes the signal travel in convolution
            const mSignalTravel: tf.Tensor = tf.matMul(mAdjacencyHat, input);
            // Output of convolution, by multiplying with weight matrix and applying non-linear activation function
            const mWeightedSignal: tf.Tensor = this.activationLayer.apply(tf.matMul(mSignalTravel, this.mWeights));
            return mWeightedSignal;
            }
        )
    }

    getClassName() {
        return 'Graph Convolution';
    }

    private getModelDirectory(){          
        const modelDir: string = path.join(__dirname, '../../actor-rdf-join-inner-multi-reinforcement-learning/model');
        return modelDir;
    }


}

export class graphConvolutionModel{
    public layers: [DenseOwnImplementation | tf.layers.Layer | graphConvolutionLayer, layerIdentifier][];
    public config: IModelConfig;
    public constructor(){
        this.layers = [];

        const fileLocation = this.getModelDirectory() + '/model-config/model-config-gcn.json';
        this.config = this.loadConfig(fileLocation);
        this.initModelRandom();
    }
    public initModelRandom(){
        console.log(this.config.layers.length);
        for (const layerConfig of this.config.layers){
            if (layerConfig.type == 'gcn'){
                this.layers.push([new graphConvolutionLayer(layerConfig.inputSize, layerConfig.outputSize, layerConfig.activation!), layerConfig.type]);
            }
            if (layerConfig.type == 'dense'){
                this.layers.push([new DenseOwnImplementation(layerConfig.inputSize, layerConfig.outputSize), layerConfig.type]);
            }
            if (layerConfig.type == 'activation'){
                this.layers.push([tf.layers.activation({activation: layerConfig.activation!, inputShape: [layerConfig.inputSize]}), layerConfig.type]);
            }
            if (layerConfig.type == 'dropout'){
                this.layers.push([tf.layers.dropout({rate: layerConfig.rate!}), layerConfig.type]);
            }
        }
    }

    public initModelWeights(){

    }

    public forwardPass(input: tf.Tensor, mAdjacency: tf.Tensor){
        let x = input;
        for (let i = 0; i<this.layers.length; i++){
            console.log("Before")
            console.log(this.layers[i][1])
            console.log(x.shape)
            if (this.layers[i][1]=='gcn'){
                x = this.layers[i][0].call(x, mAdjacency) as tf.Tensor;
            }
            else{
                const nonGraphLayer: DenseOwnImplementation | tf.layers.Layer = this.layers[i][0];
                x = nonGraphLayer.call(x, {}) as tf.Tensor;
            }
            console.log(x.shape)
        }
        console.log("And done! :)")
        return x;
    }
    public loadConfig(fileLocation: string){
        const data = fs.readFileSync(fileLocation, 'utf8');
        const parsedConfig: IModelConfig = JSON.parse(data);
        return parsedConfig;
    }

    /**
     * Temporary function that points to where the model directory is, VERY TEMPORARY!! ( I hope :))
     * @returns 
     */
    private getModelDirectory(){          
        const modelDir: string = path.join(__dirname, '../../actor-rdf-join-inner-multi-reinforcement-learning-tree/model');
        return modelDir;
    }

}

export interface IlayerConfig{
    /* 
    * Type of Layer 
    */
    type: layerIdentifier;
    /*
    * Input size of layer
    */
    inputSize: number;
    /*
    * Output size of layer
    */
    outputSize: number;
    /**
     * Rate (Optional for dropout)
     */
    rate?: number;
    /**
     * Activation type (If activation layer)
     */
    activation?: activationIdentifier
}

export interface IModelConfig{
    /*
    * Where weights are stored
    */
    weightLocation: string;
    /**
     * Layer configs
     */
    layers: IlayerConfig[];

}

// Needed to intantiate the different activation functions 

export function stringLiteralArray<T extends string>(a: T[]) {
    return a;
  }

export const activationOptions = stringLiteralArray([
    'elu', 'hardSigmoid', 'linear', 'relu', 'relu6', 'selu', 'sigmoid',
    'softmax', 'softplus', 'softsign', 'tanh', 'swish', 'mish'
  ]);
type activationIdentifier = typeof activationOptions[number]

export const layerOptions = stringLiteralArray(['gcn', 'dense', 'activation', 'batchNorm', 'dropout']);
type layerIdentifier = typeof layerOptions[number];
