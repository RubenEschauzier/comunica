import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';

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
    mVariable: tf.Tensor2D;
    mWeights: tf.Variable;


    constructor(inputSize: number, outputSize: number, activationName: string, layerName?: string) {
        super({});
        // Define the activation layer used in this layer
        this.activationLayer = tf.layers.activation(<any>{activation: activationName});
        // Define input feature size
        this.inputSize = inputSize;
        // Define output feature size, which is size of node representations
        this.outputSize = outputSize;
        // Graph adjacency matrix used to regulate signal travel
        // this.mAdjacency = adjacencyMatrix;

        // Add self-loops to adjacency matrix to ensure own node features are used to create node representation
        // this.mIdentity = tf.eye(this.mAdjacency.shape[0], this.mAdjacency.shape[1])
        // this.mAdjacencyHat = tf.add(this.mAdjacency,this.mIdentity);

        // Diagonal matrix that normalises the adjacency matrix in convolution
        // this.mD = tf.sum(this.mAdjacencyHat, 0);
        // this.mDInv = tf.diag(tf.rsqrt(this.mD));

        // // Normalised adjecency matrix, we perform this is initialisation to not compute it in the call
        // this.mAdjacencyHat = tf.matMul(tf.matMul(this.mDInv, this.mAdjacencyHat), this.mDInv);
        // // Trainable convolution weight matrix
        if (layerName){
            this.mWeights = tf.variable(tf.randomNormal([this.inputSize, this.outputSize]), true,
            layerName);
        }
        else{
            this.mWeights = tf.variable(tf.randomNormal([this.inputSize, this.outputSize]), true);
        }


    }

    
    // I should call build function for flexibility, not sure if I need it yet, but might become needed
    call(input: tf.Tensor2D,  mAdjacency: tf.Tensor2D, kwargs?: any) {
        /*  Get inverted square of node degree diagonal matrix */
        const mD: tf.Tensor2D = tf.sum(mAdjacency, 1);

        const mDInv: tf.Tensor = tf.diag(tf.rsqrt(mD));

        // Normalised adjecency matrix, we perform this is initialisation to not compute it in the call
        const mAdjacencyHat: tf.Tensor2D = tf.matMul(tf.matMul(mDInv, mAdjacency), mDInv);
        // Trainable convolution weight matrix
        return tf.tidy(() => {
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

}

export class graphConvolutionLayerTest extends tf.layers.Layer{

    // Disgusting any, should look up what a tf.layers.activation is
    activationLayer: any;
    inputSize: number;
    outputSize: number;
    mWeights: tf.LayerVariable;
    activationName: string;
    name: string;



    constructor(inputSize: number, outputSize: number, activationName: string, name: string, config: Object) {
        super(config);
        // Define the activation layer used in this layer
        this.activationName;
        this.activationLayer = tf.layers.activation(<any>{activation: activationName});
        // Define input feature size
        this.inputSize = inputSize;
        // Define output feature size, which is size of node representations
        this.outputSize = outputSize;
        this.name = name;

    
    }
    build(inputShape: number[]){
        console.log("Building it!");
        console.log(inputShape);
        this.mWeights = this.addWeight(this.name, [inputShape[1], inputShape[2]], 'float32', tf.initializers.randomNormal({}))
    }

    
    // I should call build function for flexibility, not sure if I need it yet, but might become needed
    call(input: tf.Tensor,  mAdjacency: tf.Tensor2D, kwargs?: any) {
        // Trainable convolution weight matrix
        return tf.tidy(() => {
            /*  Get inverted square of node degree diagonal matrix */
            const mD: tf.Tensor2D = tf.sum(mAdjacency, 1);
            const mDInv: tf.Tensor = tf.diag(tf.rsqrt(mD));

            // Normalised adjecency matrix, we perform this is initialisation to not compute it in the call
            const mAdjacencyHat: tf.Tensor2D = tf.matMul(tf.matMul(mDInv, mAdjacency), mDInv);
            // Tensor that denotes the signal travel in convolution
            const mSignalTravel: tf.Tensor = tf.matMul(mAdjacencyHat, input);
            // Output of convolution, by multiplying with weight matrix and applying non-linear activation function
            // Check if activation function is ok
            const mWeightedSignal: tf.Tensor = this.activationLayer.apply(tf.matMul(mSignalTravel, this.mWeights.read()));
            return mWeightedSignal;
            }
        )
    }

    getConfig(){
        const config = super.getConfig();
        Object.assign(config, {activationName: this.activationName, inputSize: this.inputSize, outputSize: this.outputSize});
        return config;
    }

    getClassName() {
        return 'Graph_Convolution';
    }

}

export class graphConvolutionModel{
    /* For now a simple constructor, should prob use a json config file and with pretrained layers*/
    /* For saving new_model = tf.keras.models.load_model('model.h5', custom_objects={'CustomLayer': CustomLayer}) */
    layer1: graphConvolutionLayer;
    layer2: graphConvolutionLayer;
    layer3: tf.layers.Layer

    constructor(){
        this.layer1 = new graphConvolutionLayer(1, 6, "relu");
        this.layer2 = new graphConvolutionLayer(6, 6, "relu");
        this.layer3 = tf.layers.dense({inputShape: [6], units: 1, activation: 'linear'});
    }
    forwardPass(input: tf.Tensor2D,  mAdjacency: tf.Tensor2D, kwargs?: any){
        const hiddenState: tf.Tensor = this.layer1.call(input, mAdjacency);
        const nodeRepresentations: tf.Tensor = this.layer2.call(tf.reshape(hiddenState, [mAdjacency.shape[0],6]), mAdjacency);
        const output = this.layer3.apply(nodeRepresentations); 
        return output;
    }
}

export class graphConvolutionModel2{
    /* For now a simple constructor, should prob use a json config file and with pretrained layers*/
    /* For saving new_model = tf.keras.models.load_model('model.h5', custom_objects={'CustomLayer': CustomLayer}) */
    layer1: graphConvolutionLayer;
    layer2: graphConvolutionLayer;
    layer3: tf.layers.Layer

    constructor(){
        this.layer1 = new graphConvolutionLayer(1, 6, "relu");
        this.layer2 = new graphConvolutionLayer(6, 6, "relu");
        this.layer3 = tf.layers.dense({inputShape: [6], units: 1, activation: 'sigmoid'});



    }
    model(inputFeatures: tf.Tensor2D, mAdjacency: tf.Tensor2D){
        const hiddenState: tf.Tensor = this.layer1.call(inputFeatures, mAdjacency);
        const nodeRepresentations: tf.Tensor = this.layer2.call(tf.reshape(hiddenState, [mAdjacency.shape[0], 6]), mAdjacency);
        const output = this.layer3.apply(nodeRepresentations); 

        return output
    }
}

let test: graphConvolutionModel2 = new graphConvolutionModel2();

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
