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

    public static className: string = 'graphConvolutionLayer'; 

    constructor(inputSize: number, outputSize: number, activationName: string, layerName?: string, modelDirectory?: string) {
        super({});
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
        const weights = this.readWeightFile(loadPath);
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
        fs.writeFileSync(savePath, JSON.stringify(weights));
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

    public disposeLayer(){
        // Dispose all weights before endpoint shutdown
        this.mWeights.dispose()
    }

    public getClassName() {
        return 'Graph Convolution';
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

    public static className: string = 'graphConvolutionLayerDirected'; 

    constructor(inputSize: number, outputSize: number, activationName: string, layerName?: string, modelDirectory?: string) {
        super({});
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
    public loadWeights(fileLocation: string): void{
        const weights = this.readWeightFile(fileLocation);
        this.mWeights.assign(weights);
    }

    private readWeightFile(fileLocationWeights: string){
        const data = fs.readFileSync(fileLocationWeights, 'utf8');
        const weightArray: number[][] = JSON.parse(data);
        const weights: tf.Tensor = tf.tensor(weightArray);
        return weights;
    }

    public saveWeights(fileLocation: string){
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

    public getClassName() {
        return 'Graph Convolution Directed';
    }

    public disposeLayer(){
        // Dispose all weights before endpoint shutdown
        this.mWeights.dispose()
    }
}


export class GraphConvolutionModel{
    public layers: [DenseOwnImplementation | tf.layers.Layer | GraphConvolutionLayer, layerIdentifier][];
    public config: IModelConfig;
    public initialised: boolean;
    public static configName: string = 'model-config-gcn.json';

    public constructor(modelDirectory: string){
        this.layers = [];
        const configLocation = path.join(modelDirectory, GraphConvolutionModel.configName);
        this.config = this.loadConfig(configLocation);
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

    public initModelWeights(modelDirectory: string){
        let gcnIndex = 0;
        let gcnDirectedIndex = 0;
        let denseIndex = 0
        for (const layerConfig of this.config.layers){
            if (layerConfig.type == 'gcn'){
                const newGcnLayer = new GraphConvolutionLayer(layerConfig.inputSize, layerConfig.outputSize, layerConfig.activation!, undefined, modelDirectory);
                try{
                    newGcnLayer.loadWeights(path.join(modelDirectory, this.config.weightLocationGcn, `gcn-weight-${gcnIndex}.txt`));
                }
                catch(err: any){
                    console.warn(`Failed to load weights from weightfile, error: ${err.message}`)
                }
                this.layers.push([newGcnLayer, layerConfig.type]);
                gcnIndex += 1;
            }
            if (layerConfig.type == 'gcndirected'){
                const newGcnLayer = new GraphConvolutionLayerDirected(layerConfig.inputSize, layerConfig.outputSize, layerConfig.activation!, undefined, modelDirectory);
                try{
                    newGcnLayer.loadWeights(path.join(modelDirectory, this.config.weightLocationGcnDirected, `gcn-weight-${gcnDirectedIndex}.txt`));
                }
                catch(err: any){
                    console.warn(`Failed to load weights from weightfile, error: ${err.message}`)
                }
                this.layers.push([newGcnLayer, layerConfig.type]);
                gcnDirectedIndex += 1;
            }
            if (layerConfig.type == 'dense'){
                const newDenseLayer = new DenseOwnImplementation(layerConfig.inputSize, layerConfig.outputSize, undefined, undefined, modelDirectory);
                try{
                    newDenseLayer.loadWeights(
                        path.join(modelDirectory, this.config.weightLocationDense, `dense`, `dense-weight-${denseIndex}.txt`
                    ));
                    newDenseLayer.loadBias(
                        path.join(modelDirectory, this.config.weightLocationDense, `bias`, `dense-bias-${denseIndex}.txt`
                    ));
                }
                catch(err: any){
                    console.warn(`Failed to load weights from weightfile, error ${err.message}`);
                }
                this.layers.push([newDenseLayer, layerConfig.type]);
                denseIndex += 1;
            }

            if (layerConfig.type == 'activation'){
                this.layers.push([tf.layers.activation({activation: layerConfig.activation!, inputShape: [layerConfig.inputSize]}), layerConfig.type]);
            }
            if (layerConfig.type == 'dropout'){
                this.layers.push([tf.layers.dropout({rate: layerConfig.rate!}), layerConfig.type]);
            }
        }
    }
    /**
     * Creates directory structure that the layer weights will be saved in, standard structure looks like this:
     * model-dir/
     *  |- model-config-gcn.json
     *  |- weights-gcn/
     *  |- weights-gcn-directed/
     *  |- weights-dense/
     *    |- bias/
     *    |- dense/
     */
    public createGcnDirectoryStructure(modelDirectory: string){
        console.log("GCN Create Dir Structure")
        if (!fs.existsSync(path.join(modelDirectory, this.config.weightLocationGcn))){
            fs.mkdirSync(path.join(modelDirectory, this.config.weightLocationGcn));
        }
        if (!fs.existsSync(path.join(modelDirectory, this.config.weightLocationGcnDirected))){
            fs.mkdirSync(path.join(modelDirectory, this.config.weightLocationGcnDirected));
        }
        if (!fs.existsSync(path.join(modelDirectory, this.config.weightLocationDense))){
            fs.mkdirSync(path.join(modelDirectory, this.config.weightLocationDense));
        }
        if (!fs.existsSync(path.join(modelDirectory, this.config.weightLocationDense, "bias"))){
            fs.mkdirSync(path.join(modelDirectory, this.config.weightLocationDense, "bias"));
        }
        if (!fs.existsSync(path.join(modelDirectory, this.config.weightLocationDense, "dense"))){
            fs.mkdirSync(path.join(modelDirectory, this.config.weightLocationDense, "dense"));
        }
        console.log("GCN Create Dir Structure DONE")
    }

    /**
     * Saves the model state and config to specified location. Follows a predetermined file structure from weight location directory.
     * @param modelDirectory model directory absolute path, this allows the user to specify where the save must happen.
     * This can be used to save checkpoints to a different location than where the model config is initially defined.
     */
    public saveModel(modelDirectory: string){
        this.createGcnDirectoryStructure(modelDirectory);
        this.saveConfig(path.join(modelDirectory, GraphConvolutionModel.configName));
        let numGcn = 0;
        let numGcnDirected = 0;
        let numDense = 0;
        for (let i = 0; i<this.layers.length; i++){
            if (this.layers[i][1] == 'gcn'){
                const gcnLayer = <GraphConvolutionLayer> this.layers[i][0];
                gcnLayer.saveWeights(path.join(modelDirectory, this.config.weightLocationGcn, `gcn-weight-${numGcn}.txt`));
                numGcn += 1;
            }

            if (this.layers[i][1] == 'gcndirected'){
                const gcnLayer = <GraphConvolutionLayerDirected> this.layers[i][0];
                gcnLayer.saveWeights(path.join(modelDirectory, this.config.weightLocationGcnDirected, `gcn-weight-${numGcnDirected}.txt`));
                numGcnDirected += 1;
            }

            if (this.layers[i][1] == 'dense'){
                const denseLayer = <DenseOwnImplementation> this.layers[i][0];
                denseLayer.saveWeights(path.join(modelDirectory, this.config.weightLocationDense, "dense",  `dense-weight-${numDense}.txt`));
                denseLayer.saveBias(path.join(modelDirectory, this.config.weightLocationDense, "bias",  `dense-bias-${numDense}.txt`));
                numDense += 1;
            }
        }
    }

    public forwardPass(input: tf.Tensor, mAdjacency: tf.Tensor){
        let x = input;
        for (let i = 0; i< this.layers.length; i++){
            if (this.layers[i][1]=='gcn'||this.layers[i][1]=='gcndirected'){
                x = this.layers[i][0].call(x, mAdjacency) as tf.Tensor;
            }
            else{
                const nonGraphLayer: DenseOwnImplementation | tf.layers.Layer = this.layers[i][0];
                x = nonGraphLayer.call(tf.transpose(x), {}) as tf.Tensor;
            }
        }
        return x;
    }
    public loadConfig(fileLocation: string){
        const data = fs.readFileSync(fileLocation, 'utf8');
        const parsedConfig: IModelConfig = JSON.parse(data);
        return parsedConfig;
    }

    public saveConfig(fileLocation: string){
        fs.writeFileSync(fileLocation, JSON.stringify(this.config));
    }

    public getConfig(){
        return this.config;
    }

    public getLayers(){
        return this.layers;
    }

    public flushModel(): void{
        for (const layer of this.layers){
            if (layer[1] == 'gcn'){
                (<GraphConvolutionLayer> layer[0]).disposeLayer();
            }
            if (layer[1] == 'gcndirected'){
                (<GraphConvolutionLayerDirected> layer[0]).disposeLayer();
            }

            if (layer[1] == 'dense' || 'gcn' || 'gcndirected'){
                (<DenseOwnImplementation> layer[0]).disposeLayer()
            }
        }
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
     * Where weights are stored for gcn (relative to model directory)
     */
    weightLocationGcn: string;
    /*
     * Where weights are stored for directed gcn (relative to model directory)
     */
    weightLocationGcnDirected: string;

    /*
     * Where weights are stored for dense layers (relative to model directory)
     */
    weightLocationDense: string;
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
