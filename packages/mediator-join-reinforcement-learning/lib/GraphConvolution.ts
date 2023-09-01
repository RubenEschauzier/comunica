import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as path from 'path';
import { DenseOwnImplementation } from '@comunica/actor-rdf-join-inner-multi-reinforcement-learning-tree/lib/fullyConnectedLayers';

// This is tensorflow for cpu, if we were to train and it takes long we can change to gpu probably.
// https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-62acf5b143d0

export class GraphConvolutionLayer extends tf.layers.Layer{
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

    constructor(inputSize: number, outputSize: number, activationName: string, layerName?: string, modelDirectory?: string) {
        super({});
        this.modelDirectory = this.getModelDirectory();
        if (!(modelDirectory == undefined)){
            this.modelDirectory = modelDirectory
        }
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
    public loadWeights(loadPath: string): void{
        const fileLocation = path.join(this.modelDirectory, loadPath);
        const weights = this.readWeightFile(fileLocation);
        this.mWeights.assign(weights);
    }

    private readWeightFile(fileLocationWeights: string){
        const data = fs.readFileSync(fileLocationWeights, 'utf8');
        const weightArray: number[][] = JSON.parse(data);
        const weights: tf.Tensor = tf.tensor(weightArray);
        return weights;
    }

    public saveWeights(savePath: string){
        const fileLocation = path.join(this.modelDirectory, savePath);
        const weights =  this.mWeights.arraySync();
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
    public getWeights(trainableOnly?: boolean | undefined): tf.Tensor<tf.Rank>[] {
        return [this.mWeights];
    }

    getClassName() {
        return 'Graph Convolution';
    }

    private getModelDirectory(){          
        const modelDir: string = path.join(__dirname, '../../actor-rdf-join-inner-multi-reinforcement-learning/model');
        return modelDir;
    }

}
// According to the author, GCN on directed graphs should use $D^{-1}*A$ instead of $D^{-1/2}*A*D^{-1/2}
// https://github.com/tkipf/gcn/issues/91
// In this implementation of directed graph convolution, we take the adj matrix [[1,1],[0,1]] as the following graph A ----> B. 
// Furthermore, the degree matrix becomes [[2,0], [0,1]]. Then from this, 
// we get that the feature of A will be combination of A and B, and B only B. 
// This is counterintuitive, so maybe it should be changed. Note that this is not a problem on the undirected version.
export class GraphConvolutionLayerDirected extends tf.layers.Layer{
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

    public static className: string = 'graphConvolutionLayerDirected'; 

    constructor(inputSize: number, outputSize: number, activationName: string, layerName?: string, modelDirectory?: string) {
        super({});
        this.modelDirectory = this.getModelDirectory();
        if (!(modelDirectory == undefined)){
            this.modelDirectory = modelDirectory
        }
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
    public loadWeights(loadPath: string): void{
        const fileLocation = path.join(this.modelDirectory, loadPath);
        const weights = this.readWeightFile(fileLocation);
        this.mWeights.assign(weights);
    }

    private readWeightFile(fileLocationWeights: string){
        const data = fs.readFileSync(fileLocationWeights, 'utf8');
        const weightArray: number[][] = JSON.parse(data);
        const weights: tf.Tensor = tf.tensor(weightArray);
        return weights;
    }

    public saveWeights(savePath: string){
        const fileLocation = path.join(this.modelDirectory, savePath);
        const weights =  this.mWeights.arraySync();
        fs.writeFileSync(fileLocation, JSON.stringify(weights));
    }

    call(input: tf.Tensor,  mAdjacency: tf.Tensor, kwargs?: any): tf.Tensor {
        return tf.tidy(() => {
            // Get inverted square of node degree diagonal matrix 
            const mD: tf.Tensor2D = tf.sum(mAdjacency, 1);
            const mDInv: tf.Tensor = tf.diag(tf.reciprocal(mD));

            // Normalised adjecency matrix, we perform this is initialisation to not compute it in the call
            const mAdjacencyHat: tf.Tensor2D = tf.matMul(mDInv, mAdjacency);

            // Tensor that denotes the signal travel in convolution
            const mSignalTravel: tf.Tensor = tf.matMul(mAdjacencyHat, input);
            // Output of convolution, by multiplying with weight matrix and applying non-linear activation function
            const mWeightedSignal: tf.Tensor = this.activationLayer.apply(tf.matMul(mSignalTravel, this.mWeights));
            return mWeightedSignal;
            }
        )
    }
    
    public getWeights(trainableOnly?: boolean | undefined): tf.Tensor<tf.Rank>[] {
        return [this.mWeights];
    }

    getClassName() {
        return 'Graph Convolution Directed';
    }

    private getModelDirectory(){          
        const modelDir: string = path.join(__dirname, '../../actor-rdf-join-inner-multi-reinforcement-learning/model');
        return modelDir;
    }
}


export class GraphConvolutionModel{
    public layers: [DenseOwnImplementation | tf.layers.Layer | GraphConvolutionLayer, layerIdentifier][];
    public config: IModelConfig;
    public modelDirectory: string;

    public constructor(modelDirectory?: string){
        this.layers = [];
        this.modelDirectory = (modelDirectory != undefined ? modelDirectory : this.getModelDirectory());
        const fileLocation = modelDirectory + '/model-config/model-config-gcn.json';
        this.config = this.loadConfig(fileLocation);
    }

    public initModelRandom(){
        for (const layerConfig of this.config.layers){
            if (layerConfig.type == 'gcn'){
                this.layers.push([new GraphConvolutionLayer(layerConfig.inputSize, layerConfig.outputSize, layerConfig.activation!), layerConfig.type]);
            }
            if (layerConfig.type == 'gcndirected'){
                this.layers.push([new GraphConvolutionLayerDirected(layerConfig.inputSize, layerConfig.outputSize, layerConfig.activation!), layerConfig.type]);
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
        for (const layerConfig of this.config.layers){
            if (layerConfig.type == 'gcn'){
                const newGcnLayer = new GraphConvolutionLayer(layerConfig.inputSize, layerConfig.outputSize, layerConfig.activation!, undefined, this.modelDirectory);
                try{
                    newGcnLayer.loadWeights(`${this.config.weightLocation}/gcn-weights-test/${layerConfig.weightLocation}`);
                }
                catch(err: any){
                    console.warn(`Failed to load weights from weightfile ${layerConfig.weightLocation}: error: ${err.message}`)
                }
                this.layers.push([newGcnLayer, layerConfig.type]);
            }
            if (layerConfig.type == 'gcndirected'){
                const newGcnLayer = new GraphConvolutionLayerDirected(layerConfig.inputSize, layerConfig.outputSize, layerConfig.activation!, undefined, this.modelDirectory);
                try{
                    newGcnLayer.loadWeights(`${this.config.weightLocation}/gcn-weights-test/${layerConfig.weightLocation}`);
                }
                catch(err: any){
                    console.warn(`Failed to load weights from weightfile ${layerConfig.weightLocation}: error: ${err.message}`)
                }
                this.layers.push([newGcnLayer, layerConfig.type]);
            }
            if (layerConfig.type == 'dense'){
                const newDenseLayer = new DenseOwnImplementation(layerConfig.inputSize, layerConfig.outputSize, undefined, undefined, this.modelDirectory);
                try{
                    newDenseLayer.loadWeights(`${this.config.weightLocation}/dense-weights-test/${layerConfig.weightLocation}`);
                    newDenseLayer.loadBias(`${this.config.weightLocation}/dense-weights-test/${layerConfig.biasLocation}`);
                }
                catch(err: any){
                    console.warn(`Failed to load weights from weightfile ${layerConfig.weightLocation} and biasFile ${layerConfig.biasLocation}: error ${err.message}`);
                }
                this.layers.push([newDenseLayer, layerConfig.type]);
            }
            if (layerConfig.type == 'activation'){
                this.layers.push([tf.layers.activation({activation: layerConfig.activation!, inputShape: [layerConfig.inputSize]}), layerConfig.type]);
            }
            if (layerConfig.type == 'dropout'){
                this.layers.push([tf.layers.dropout({rate: layerConfig.rate!}), layerConfig.type]);
            }
        }
    }

    public saveModel(){
        for (let i = 0; i<this.layers.length; i++){
            if (this.layers[i][1] == 'gcn'){
                const gcnLayer = <GraphConvolutionLayer> this.layers[i][0];
                gcnLayer.saveWeights(`${this.config.weightLocation}/gcn-saved-weights-test/${this.config.layers[i].weightLocation}`);
            }
            if (this.layers[i][1] == 'dense'){
                const denseLayer = <DenseOwnImplementation> this.layers[i][0];
                denseLayer.saveWeights(`${this.config.weightLocation}/dense-saved-weights-test/${this.config.layers[i].weightLocation}`);
                denseLayer.saveBias(`${this.config.weightLocation}/dense-saved-weights-test/${this.config.layers[i].biasLocation}`)
            }
        }
    }

    public forwardPass(input: tf.Tensor, mAdjacency: tf.Tensor){
        let x = input;
        for (let i = 0; i<this.layers.length; i++){
            if (this.layers[i][1]=='gcn'){
                x = this.layers[i][0].call(x, mAdjacency) as tf.Tensor;
            }
            else{
                const nonGraphLayer: DenseOwnImplementation | tf.layers.Layer = this.layers[i][0];
                x = nonGraphLayer.call(x, {}) as tf.Tensor;
            }
        }
        return x;
    }
    public loadConfig(fileLocation: string){
        const data = fs.readFileSync(fileLocation, 'utf8');
        const parsedConfig: IModelConfig = JSON.parse(data);
        return parsedConfig;
    }

    public getConfig(){
        return this.config;
    }

    public getLayers(){
        return this.layers;
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
    /*
    * Where to find weights to initialise layer with
    */
    weightLocation?: string;
    /*
    * Where to find bias to initialise layer with
    */
    biasLocation?: string;
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

export const layerOptions = stringLiteralArray(['gcn', 'gcndirected', 'dense', 'activation', 'batchNorm', 'dropout']);
type layerIdentifier = typeof layerOptions[number];
