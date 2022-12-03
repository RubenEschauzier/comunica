import { TypeofTypeAnnotation } from '@babel/types';
import { hasContextSingleSourceOfType } from '@comunica/bus-rdf-resolve-quad-pattern';
import * as tf from '@tensorflow/tfjs-node';
import { kMaxLength } from 'buffer';
import * as fs from 'fs';
import * as path from 'path';

// This is tensorflow for cpu, if we were to train and it takes long we can change to gpu probably.
// https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-62acf5b143d0

export class graphConvolutionLayer extends tf.layers.Layer{

    // Disgusting any, should look up what a tf.layers.activation is
    activationLayer: any;
    mInput: tf.Tensor2D;
    inputSize: number;
    outputSize: number;
    mAdjacency: tf.Tensor2D;
    mIdentity: tf.Tensor2D;
    mAdjacencyHat: tf.Tensor2D;
    // mVariable: tf.Tensor2D;
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
        // this.mWeights = tf.variable(tf.randomUniform([this.inputSize, this.outputSize], 0, .1), true,
        //                             layerName);
        this.heInitTerm = tf.sqrt(tf.div(tf.tensor(2), tf.tensor(outputSize)));
        this.mWeights = tf.variable(tf.mul(tf.randomNormal([this.inputSize, this.outputSize],0,1),this.heInitTerm), true,
        layerName);
        this.heInitTerm.dispose();
        // this.mWeights = tf.variable(tf.tensor([ 2.29773776, -237.61560258,  -18.59292076,  -91.67139649,
        //     0.        ,    0.        ,    0.        ,   19.7895892 ]), true, layerName).reshape([this.inputSize, this.outputSize]);
            

    }
    public async loadWeights(loadPath: string): Promise<void>{
        const fileLocation = path.join(this.modelDirectory, loadPath);
        const weights = await this.readWeightFile(fileLocation);
        this.mWeights.assign(weights);
    }

    private readWeightFile(fileLocationWeights: string){
        const readWeights = new Promise<tf.Tensor>( (resolve, reject) => {
            fs.readFile(fileLocationWeights, 'utf8', async function (err, data) {
                if (err) throw err;
                const weightArray: number[][] = JSON.parse(data);
                // const weights: tf.Tensor[] = weightArray.map((x: number[]) => tf.tensor(x));
                const weights: tf.Tensor = tf.tensor(weightArray);
                resolve(weights);
              });
        })
        return readWeights;

    }

    public async saveWeights(savePath: string): Promise<void> {
        const weights = await this.mWeights.array();
        const fileLocation = path.join(this.modelDirectory, savePath);
        fs.writeFile(fileLocation, JSON.stringify(weights), function(err: any) {
            if(err) {
                return console.log(err);
            }
        }); 
    }

    
    // I should call build function for flexibility, not sure if I need it yet, but might become needed
    call(input: tf.Tensor2D|tf.Tensor,  mAdjacency: tf.Tensor2D, kwargs?: any): tf.Tensor {
        return tf.tidy(() => {
            /*  Get inverted square of node degree diagonal matrix */
            const mD: tf.Tensor2D = tf.sum(mAdjacency, 1);
            const mDInv: tf.Tensor = tf.diag(tf.rsqrt(mD));

            const mDInvFull: tf.Tensor = tf.diag(tf.reciprocal(mD));

            // Normalised adjecency matrix, we perform this is initialisation to not compute it in the call
            // const mAdjacencyHat: tf.Tensor2D = tf.matMul(tf.matMul(mDInv, mAdjacency), mDInv);
            // Normalised adjacency matrix according to kipf for directed graphs;
            const mAdjacencyHat: tf.Tensor2D = tf.matMul(mDInvFull, mAdjacency);


            // Tensor that denotes the signal travel in convolution
            const mSignalTravel: tf.Tensor = tf.matMul(mAdjacencyHat, input);
            // Output of convolution, by multiplying with weight matrix and applying non-linear activation function
            // Check if activation function is ok
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

export class denseOwnImplementation extends tf.layers.Layer{
    heInitTerm: tf.Tensor;
    mWeights: tf.Variable;
    mBias: tf.Variable;
    inputDim: number;
    outputDim: number;
    modelDirectory: string;


    public constructor(inputDim: number, outputDim: number){
        super({});
        this.inputDim = inputDim;
        this.outputDim = outputDim;
        this.heInitTerm = tf.sqrt(tf.div(tf.tensor(2), tf.tensor(this.outputDim)));
        this.mWeights = tf.variable(tf.mul(tf.randomNormal([this.inputDim, this.outputDim],0,1),this.heInitTerm), true);
        this.mBias = tf.variable(tf.mul(tf.randomNormal([1, outputDim],0,1), this.heInitTerm), true);
        this.modelDirectory = this.getModelDirectory();
    }

    call(inputs: tf.Tensor2D|tf.Tensor){
        return tf.tidy(() => {
            return tf.add(tf.matMul(inputs, this.mWeights), this.mBias.broadcastTo([inputs.shape[0], this.outputDim]));
        });
    }
    getClassName() {
        return 'Dense Own';
    }

    public async saveWeights(savePath: string): Promise<void> {
        const weights = await this.mWeights.array();
        const fileLocation = path.join(this.modelDirectory, savePath);
        fs.writeFile(fileLocation, JSON.stringify(weights), function(err: any) {
            if(err) {
                return console.log(err);
            }
        }); 
    }
    public async saveBias(savePath:string): Promise<void> {
        const weights = await this.mBias.array();
        const fileLocation = path.join(this.modelDirectory, savePath);
        fs.writeFile(fileLocation, JSON.stringify(weights), function(err: any) {
            if(err) {
                return console.log(err);
            }
        }); 
    }

    public async loadWeights(loadPath: string): Promise<void>{
        const fileLocation = path.join(this.modelDirectory, loadPath);
        const weights = await this.readWeightFile(fileLocation);
        this.mWeights.assign(weights);
    }
    public async loadBias(loadPath: string): Promise<void>{
        const fileLocation = path.join(this.modelDirectory, loadPath);
        const weights = await this.readWeightFile(fileLocation);
        this.mBias.assign(weights);
    }

    private readWeightFile(fileLocationWeights: string){
        const readWeights = new Promise<tf.Tensor>( (resolve, reject) => {
            fs.readFile(fileLocationWeights, 'utf8', async function (err, data) {
                if (err) throw err;
                const weightArray: number[][] = JSON.parse(data);
                // const weights: tf.Tensor[] = weightArray.map((x: number[]) => tf.tensor(x));
                const weights: tf.Tensor = tf.tensor(weightArray);
                resolve(weights);
              });
        })
        return readWeights;

    }

    private getModelDirectory(){          
        const modelDir: string = path.join(__dirname, '../../actor-rdf-join-inner-multi-reinforcement-learning/model');
        return modelDir;
    }

}
const test: tf.layers.Layer = tf.layers.batchNormalization({trainable: true});


// export class graphConvolutionLayerTest extends tf.layers.Layer{

//     // Disgusting any, should look up what a tf.layers.activation is
//     activationLayer: any;
//     inputSize: number;
//     outputSize: number;
//     mWeights: tf.LayerVariable;
//     activationName: string;
//     name: string;



//     constructor(inputSize: number, outputSize: number, activationName: string, name: string, config: Object) {
//         super(config);
//         // Define the activation layer used in this layer
//         this.activationName;
//         this.activationLayer = tf.layers.activation(<any>{activation: activationName});
//         // Define input feature size
//         this.inputSize = inputSize;
//         // Define output feature size, which is size of node representations
//         this.outputSize = outputSize;
//         this.name = name;

    
//     }
//     build(inputShape: number[]){
//         console.log("Building it!");
//         console.log(inputShape);
//         this.mWeights = this.addWeight(this.name, [inputShape[1], inputShape[2]], 'float32', tf.initializers.randomNormal({}))
//     }

    
//     // I should call build function for flexibility, not sure if I need it yet, but might become needed
//     call(input: tf.Tensor,  mAdjacency: tf.Tensor2D, kwargs?: any) {
//         // Trainable convolution weight matrix
//         return tf.tidy(() => {
//             /*  Get inverted square of node degree diagonal matrix */
//             const mD: tf.Tensor2D = tf.sum(mAdjacency, 1);
//             const mDInv: tf.Tensor = tf.diag(tf.rsqrt(mD));

//             // Normalised adjecency matrix, we perform this is initialisation to not compute it in the call
//             const mAdjacencyHat: tf.Tensor2D = tf.matMul(tf.matMul(mDInv, mAdjacency), mDInv);
//             // Tensor that denotes the signal travel in convolution
//             const mSignalTravel: tf.Tensor = tf.matMul(mAdjacencyHat, input);
//             // Output of convolution, by multiplying with weight matrix and applying non-linear activation function
//             // Check if activation function is ok
//             const mWeightedSignal: tf.Tensor = this.activationLayer.apply(tf.matMul(mSignalTravel, this.mWeights.read()));
//             return mWeightedSignal;
//             }
//         )
//     }

//     getConfig(){
//         const config = super.getConfig();
//         Object.assign(config, {activationName: this.activationName, inputSize: this.inputSize, outputSize: this.outputSize});
//         return config;
//     }

//     getClassName() {
//         return 'Graph_Convolution';
//     }

// }

// export class graphConvolutionModel{
//     /* For now a simple constructor, should prob use a json config file and with pretrained layers*/
//     /* For saving new_model = tf.keras.models.load_model('model.h5', custom_objects={'CustomLayer': CustomLayer}) */
//     layer1: graphConvolutionLayer;
//     layer2: graphConvolutionLayer;
//     layer3: tf.layers.Layer

//     constructor(){
//         this.layer1 = new graphConvolutionLayer(1, 6, "relu");
//         this.layer2 = new graphConvolutionLayer(6, 6, "relu");
//         this.layer3 = tf.layers.dense({inputShape: [6], units: 1, activation: 'linear'});
//     }
//     forwardPass(input: tf.Tensor2D,  mAdjacency: tf.Tensor2D, kwargs?: any){
//         const hiddenState: tf.Tensor = this.layer1.call(input, mAdjacency);
//         const nodeRepresentations: tf.Tensor = this.layer2.call(tf.reshape(hiddenState, [mAdjacency.shape[0],6]), mAdjacency);
//         const output = this.layer3.apply(nodeRepresentations); 
//         return output;
//     }
// }

export class graphConvolutionModel{
    /* For now a simple constructor, should prob use a json config file and with pretrained layers*/
    /* For saving new_model = tf.keras.models.load_model('model.h5', custom_objects={'CustomLayer': CustomLayer}) */
    graphConvolutionalLayer1: graphConvolutionLayer;
    graphConvolutionalLayer2: graphConvolutionLayer;

    denseLayerValue: tf.layers.Layer;
    denseLayerPolicy: tf.layers.Layer;

    reluLayer: tf.layers.Layer;

    denseLayerValueFileName: string;
    denseLayerPolicyFileName: string;

    model: tf.Sequential;
    
    allLayersModel: [graphConvolutionLayer|tf.layers.Layer|denseOwnImplementation, layerIdentifier][][];
    layersHidden: [graphConvolutionLayer|tf.layers.Layer|denseOwnImplementation, layerIdentifier][];
    layersValue: [graphConvolutionLayer|tf.layers.Layer|denseOwnImplementation, layerIdentifier][];
    layersPolicy: [graphConvolutionLayer|tf.layers.Layer|denseOwnImplementation, layerIdentifier][];

    fs: any;     path: any;
    modelDirectory: string;

    public constructor(loss?: any, optimizer?: any){
        this.fs = require('fs');
        this.path = require('path');

        this.modelDirectory = this.getModelDirectory();
        this.denseLayerValueFileName = 'denseLayerValue';
        this.denseLayerPolicyFileName = 'denseLayerValue';
    
        // this.graphConvolutionalLayer1 = new graphConvolutionLayer(1, 6, "softplus");
        // this.graphConvolutionalLayer2 = new graphConvolutionLayer(6, 6, "softplus");
        this.denseLayerValue = tf.layers.dense({inputShape: [6], units: 1, activation: 'linear', 'trainable': true});
        this.denseLayerPolicy = tf.layers.dense({inputShape: [6], units: 1, activation: 'sigmoid', 'trainable': true});

        this.reluLayer = tf.layers.activation({activation: 'softplus', inputShape: [1], 'name': 'finalReluLayer'});

        this.layersHidden = [];
        this.layersValue = [];
        this.layersPolicy = [];

        this.allLayersModel = [this.layersHidden, this.layersValue, this.layersPolicy];
    }

    public forwardPass(inputFeatures: tf.Tensor2D, mAdjacency: tf.Tensor2D){
        return tf.tidy(() => {
            // inputFeatures.print();
            const hiddenState: tf.Tensor = this.graphConvolutionalLayer1.call(inputFeatures, mAdjacency);
            // hiddenState.print();
            const nodeRepresentations: tf.Tensor = this.graphConvolutionalLayer2.call(tf.reshape(hiddenState, [mAdjacency.shape[0], 6]), mAdjacency);
            // nodeRepresentations.print();
            const outputValue = this.denseLayerValue.apply(nodeRepresentations); 
            const outputPolicy = this.denseLayerPolicy.apply(nodeRepresentations);
            // const test = outputValue as tf.Tensor;
            // test.print();
            const outputValueRelu = this.reluLayer.apply(outputValue);
            return [outputValueRelu, outputPolicy] as tf.Tensor[];
        }
        )
    }

    public saveModel(epochString: string){
        tf.serialization.registerClass(graphConvolutionLayer);
        const firstLayer: graphConvolutionLayer = this.layersHidden[0][0] as graphConvolutionLayer
        const modelDir: string = firstLayer.modelDirectory
        fs.mkdirSync(modelDir + '/' + epochString, { recursive: true })

        let numGraph = 0
        let numDense = 0;
        for (const layerType of this.layersHidden){
            if (layerType[1]=='graph'){
                const layerGraph: graphConvolutionLayer = layerType[0] as graphConvolutionLayer;
                layerGraph.saveWeights('/' + epochString+'/'+'gcnLayer'+ numGraph.toString());
                numGraph+=1;
            }
            if(layerType[1]=='dense'){
                const layerGraph: denseOwnImplementation = layerType[0] as denseOwnImplementation;
                layerGraph.saveWeights('/' + epochString+'/'+'denseLayer' + numDense.toString());
                layerGraph.saveBias('/' + epochString+'/'+'denseLayerBias' + numDense.toString());

                numDense+=1;
            }
        }
        let numDenseValue = 0;
        for (const layerType of this.layersValue){
            if(layerType[1] == 'dense'){
                const layerGraph: denseOwnImplementation = layerType[0] as denseOwnImplementation;
                layerGraph.saveWeights('/' + epochString+'/'+'denseValueLayer' + numDenseValue.toString());
                layerGraph.saveBias('/' + epochString + '/' + 'denseLayerValueBias' + numDenseValue.toString());

                numDenseValue+=1;
            }
        }
        let numDensePolicy = 0;
        for (const layerType of this.layersPolicy){
            if(layerType[1] == 'dense'){
                const layerGraph: denseOwnImplementation = layerType[0] as denseOwnImplementation;
                layerGraph.saveWeights('/' + epochString+'/'+'densePolicyLayer' + numDensePolicy.toString());
                layerGraph.saveBias('/' + epochString + '/' + 'denseLayerPolicyBias' + numDensePolicy.toString());

                numDensePolicy+=1;
            }
        }
    }

    public async loadModel(epochString: string){
        let numGraph = 0;
        let numDense = 0;
        for(const layerType of this.layersHidden){
            if (layerType[1]=='graph'){
                const layerGraph: graphConvolutionLayer = layerType[0] as graphConvolutionLayer;
                await layerGraph.loadWeights('/' + epochString+'/'+'gcnLayer'+ numGraph.toString());
                numGraph+=1;
            }
            if(layerType[1]=='dense'){
                const layerDense: denseOwnImplementation = layerType[0] as denseOwnImplementation;
                await layerDense.loadWeights('/' + epochString+'/'+'denseLayer' + numDense.toString());
                await layerDense.loadBias('/' + epochString+ '/' + 'denseLayerBias' + numDense.toString());
                numDense+=1;
            }        
        }
        let numDenseValue = 0;
        for(const layerType of this.layersValue){
            if(layerType[1] == 'dense'){
                const layerDense: denseOwnImplementation = layerType[0] as denseOwnImplementation;
                await layerDense.loadWeights('/' + epochString+'/'+'denseValueLayer' + numDenseValue.toString());
                await layerDense.loadBias('/' + epochString + '/' + 'denseLayerValueBias' + numDenseValue.toString());

                numDenseValue+=1;
            }
        }
        let numDensePolicy = 0;

        for(const layerType of this.layersPolicy){
            if(layerType[1] == 'dense'){
                const layerDense: denseOwnImplementation = layerType[0] as denseOwnImplementation;
                await layerDense.loadWeights('/' + epochString+'/'+'densePolicyLayer' + numDensePolicy.toString());
                await layerDense.loadBias('/' + epochString + '/' + 'denseLayerPolicyBias' + numDensePolicy.toString());
                numDensePolicy+=1;
            }
        }
        // const layer1 = new graphConvolutionLayer(1, 6, 'relu');
        // layer1.loadWeights('gcnLayer1');

        // const layer2 = new graphConvolutionLayer(6, 6, 'relu');
        // layer2.loadWeights('gcnLayer2');

        // const denseLayerValue = await this.loadDenseLayer(this.denseLayerValueFileName);
        // const denseLayerPolicy = await this.loadDenseLayer(this.denseLayerPolicyFileName);

        // this.graphConvolutionalLayer1 = layer1; this.graphConvolutionalLayer2 = layer2; this.denseLayerValue = denseLayerValue; this.denseLayerPolicy = denseLayerPolicy;
    }

    public async saveDenseLayer(layer: tf.layers.Layer, fileName: string){
        const denseConfig: tf.serialization.ConfigDict = layer.getConfig();
        const fileLocationConfig: string = this.path.join(this.modelDirectory, fileName+'Config');
        const fileLocationWeights: string = this.path.join(this.modelDirectory, fileName+'Weights');

        fs.writeFile(fileLocationConfig, JSON.stringify(denseConfig), function(err: any) {
            if(err) {
                return console.log(err);
            }
        }); 
        const weightsLayer: tf.Tensor[] = layer.getWeights();

        let weightsLayerArray: any = await Promise.all(weightsLayer.map(async x => await x.array()));

        fs.writeFile(fileLocationWeights, JSON.stringify(weightsLayerArray), function(err: any) {
            if(err) {
                return console.log(err);
            }
        }); 


    }

    public async loadDenseLayer(fileName: string){
        const fileLocationWeights = this.path.join(this.modelDirectory, fileName + 'Weights');
        const initialisedDenseLayer = new Promise<tf.layers.Layer>( (resolve, reject) => {

            fs.readFile(fileLocationWeights, 'utf8', async function (err, data) {
                if (err) throw err;
                const weightArray: number[][] = JSON.parse(data);
                const weights: tf.Tensor[] = weightArray.map((x: number[]) => tf.tensor(x));
                const finalDenseLayer: tf.layers.Layer = tf.layers.dense({inputShape: [6], units: 1, activation: 'linear', weights: weights});
                resolve(finalDenseLayer);
              });
        })

        return initialisedDenseLayer        
    }

    private getModelDirectory(){          
        const modelDir: string = this.path.join(__dirname, '../../actor-rdf-join-inner-multi-reinforcement-learning/model');
        return modelDir;
    }

    private loadModelConfig(){
        const modelDir = this.getModelDirectory();
        const modelConfig = new Promise<IModelConfig>( (resolve, reject) => {

            fs.readFile(modelDir + '/modelConfig.json', 'utf8', async function (err, data) {
                if (err) throw err;
                const parsedData = JSON.parse(data)
                const hiddenStateLayers: IlayerConfig[] = parsedData['hiddenStateLayers'];
                const valueHeadLayers: IlayerConfig[] = parsedData['valueHeadLayers'];
                const policyHeadLayers: IlayerConfig[] = parsedData['policyHeadLayers']
                resolve({hiddenStateLayers: hiddenStateLayers, valueHeadLayers: valueHeadLayers, policyHeadLayers: policyHeadLayers});
              });
        })
        return modelConfig
    }

    public async instantiateModel(){
        const modelConfig = await this.loadModelConfig();
        // This is quick and dirty method, should prob do some combined iteration but not sure how that works in javascript yet
        const keysInConfig = (Object.keys(modelConfig) as Array<keyof IModelConfig>)
        for(const [index, key] of keysInConfig.entries()){
            for (const layerConfig of modelConfig[key]){
                if (layerConfig.type == 'graph'){
                    this.allLayersModel[index].push([new graphConvolutionLayer(layerConfig.inputSize, layerConfig.outputSize, layerConfig.activation), layerConfig.type]);
                }
                if (layerConfig.type == 'dense'){
                    const testDense = new denseOwnImplementation(layerConfig.inputSize, layerConfig.outputSize);
                    this.allLayersModel[index].push([testDense, layerConfig.type]);
                }
                if (layerConfig.type == 'activation'){
                    this.allLayersModel[index].push([tf.layers.activation({activation: layerConfig.activation, inputShape: [layerConfig.inputSize]}), layerConfig.type]);
                }
                if (layerConfig.type == 'batchNorm'){
                    this.allLayersModel[index].push([tf.layers.batchNormalization({axis:1}), layerConfig.type])
                }
                if(layerConfig.type == 'dropout'){
                    this.allLayersModel[index].push([tf.layers.dropout({rate: .5}),layerConfig.type])
                }

            }
        }
    }

    public forwardPassFromConfig(inputFeatures: tf.Tensor2D, mAdjacency: tf.Tensor2D, validation?: boolean){
        return tf.tidy(() => {
            let x: tf.Tensor = this.layersHidden[0][0].call(inputFeatures, mAdjacency) as tf.Tensor;
            for (let i=1; i<this.layersHidden.length;i++){
                const layerI = this.layersHidden[i];
                if (layerI[1] == 'graph'){
                    x = layerI[0].call(x, mAdjacency) as tf.Tensor2D;
                }
                else if (layerI[1] == 'dropout'){
                    if (!validation){
                        x=tf.dropout(x, .25)
                    }
                }
                else{
                    x = layerI[0].apply(x) as tf.Tensor2D;
                }
            }

            let outputValue: tf.Tensor = this.layersValue[0][0].apply(x) as tf.Tensor;
            for (let j=1; j<this.layersValue.length;j++){
                const layerI = this.layersValue[j];
                outputValue = layerI[0].apply(outputValue) as tf.Tensor;
            }
            let outputPolicy: tf.Tensor = this.layersPolicy[0][0].apply(x) as tf.Tensor;
            for (let k=1; k<this.layersPolicy.length;k++){
                const layerK = this.layersPolicy[k];
                outputPolicy = layerK[0].apply(outputPolicy) as tf.Tensor;
            }
            
            return [outputValue, outputPolicy] as tf.Tensor[];
        })
    }

}

export class graphConvolutionModelFunctional{
    graphConvolutionalLayer1: graphConvolutionLayer;
    graphConvolutionalLayer2: graphConvolutionLayer;

    denseLayerValue: tf.layers.Layer;
    denseLayerPolicy: tf.layers.Layer;

    reluLayer: tf.layers.Layer;

    denseLayerValueFileName: string;
    denseLayerPolicyFileName: string;

    model: tf.Sequential;

    fs: any;
    path: any;
    modelDirectory: string;

    public constructor(inputFeatureDim: number, outputFeatureDim: number){
        const model = tf.sequential()
        model.add(tf.layers.dense({units: inputFeatureDim, inputShape: [inputFeatureDim]}))
        model.add(tf.layers.dense({units: 16}))
        model.add(tf.layers.dense({units: 16}))
        model.add(tf.layers.dense({units: outputFeatureDim}))
        model.compile({optimizer: 'sgd', loss: 'meanSquaredError'})
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
    * Activation if applicable
    */
    activation: activationIdentifier;
}

export interface IModelConfig{
    /*
    * The layers used in the embedding part of the model
    */
    hiddenStateLayers: IlayerConfig[];
    /*
    * The layers used in generating the expected value of a join
    */
    valueHeadLayers: IlayerConfig[];
    /*
    * The layers used in generating the probability of a join
    (UNUSED FOR NOW)
    */
    policyHeadLayers: IlayerConfig[];
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

export const layerOptions = stringLiteralArray(['graph', 'dense', 'activation', 'batchNorm', 'dropout']);
type layerIdentifier = typeof layerOptions[number];
// // Test functions:
// const file = fs.readFileSync('Reinforcement-Learning-Join-Mediator/data/soc-karate.mtx','utf8');
// const fileSplit = file.split("\n");
// const numNodes: number = 34;

// const connectivityArray = new Array(numNodes);
// for (let i = 0; i < numNodes; i++) {
//     connectivityArray[i] = new Array(numNodes).fill(0);
// }

// for (let i = 24; i<fileSplit.length-1;i++){
//     // FileSplit[i] is of form int int, so we split again on space
//     const indexArray: string[] = fileSplit[i].split(' ');
//     connectivityArray[parseInt(indexArray[0], 10)-1][parseInt(indexArray[1], 10)-1] = 1;
//     connectivityArray[parseInt(indexArray[1], 10)-1][parseInt(indexArray[0], 10)-1] = 1;
// }
// const adjTensor = tf.tensor2d(connectivityArray);
// const featureMatrix = new Array(1)
// featureMatrix[0]= Array.from({length: numNodes}, () => Math.floor(Math.random() * 10));

// let layer: graphConvolutionLayer = new graphConvolutionLayer(1, 12, "relu", "learnedWeightsL1");
// let layer1: graphConvolutionLayer = new graphConvolutionLayer(12, 12, "relu", "learnedWeightsL2");
// const output: tf.Tensor = layer.call(tf.reshape(tf.tensor2d(featureMatrix), [34,1]), adjTensor);
// const output1: tf.Tensor = layer1.call(tf.reshape(output, [34,12]), adjTensor);
// output.print()
// output1.print()