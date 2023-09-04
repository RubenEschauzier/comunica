import * as tf from '@tensorflow/tfjs-node';
import * as path from 'path';
import * as fs from 'fs';
import { ILayerConfig } from './treeLSTM';

export class DenseOwnImplementation extends tf.layers.Layer{
    intermediateHeInit: tf.Tensor;
    heInitTerm: tf.Tensor;
    mWeights: tf.Variable;
    mBias: tf.Variable;
    inputDim: number;
    outputDim: number;
    modelDirectory: string;


    public constructor(inputDim: number, outputDim: number, weightName?: string, biasName?: string, modelDirectory?: string){
        super({});
        this.inputDim = inputDim;
        this.outputDim = outputDim;
        // Make intermediate value because we have to be able dispose to prevent memory leaks
        this.intermediateHeInit = tf.div(tf.tensor(2), tf.tensor(this.outputDim))
        this.heInitTerm = tf.sqrt(this.intermediateHeInit);
        this.mWeights = tf.variable(tf.mul(tf.randomNormal([this.outputDim, this.inputDim],0,1),this.heInitTerm), true);
        this.mBias = tf.variable(tf.mul(tf.randomNormal([outputDim,1],0,1), this.heInitTerm), true);
        this.modelDirectory = (modelDirectory != undefined ? modelDirectory : this.getModelDirectory());
    }

    call(inputs: tf.Tensor, kwargs: any){
        return tf.tidy(() => {
            return tf.add(tf.matMul(this.mWeights, inputs), this.mBias.broadcastTo([this.outputDim, 1]));
        });
    }

    getClassName() {
        return 'Dense Own';
    }

    public saveWeights(loadPath: string){
        const fileLocation = path.join(this.modelDirectory, loadPath);
        const weights = this.mWeights.arraySync();
        fs.writeFileSync(fileLocation, JSON.stringify(weights));
    }

    public saveBias(loadPath: string) {
        const fileLocation = path.join(this.modelDirectory, loadPath);
        const weights = this.mBias.arraySync();
        fs.writeFileSync(fileLocation, JSON.stringify(weights));
    }

    public loadWeights(loadPath: string): void{
        try{
            const fileLocation = path.join(this.modelDirectory, loadPath);
            const weights = this.readWeightFile(fileLocation);
            this.mWeights.assign(weights);
        }
        catch(err){
            throw err;
        }
    }

    public loadBias(loadPath: string): void{
        try{
            const fileLocation = path.join(this.modelDirectory, loadPath);
            const weights = this.readWeightFile(fileLocation);
            this.mBias.assign(weights);    
        }
        catch(err){
            throw err
        }
    }

    public getWeights(trainableOnly?: boolean | undefined): tf.Tensor<tf.Rank>[] {
        return [this.mWeights, this.mBias];
    }
    public disposeLayer(){
        // Dispose all weights in layer
        this.intermediateHeInit.dispose(); this.heInitTerm.dispose(); 
        this.mWeights.dispose(); this.mBias.dispose();
    }

    private readWeightFile(fileLocationWeights: string){
        const data: number[][] = JSON.parse(fs.readFileSync(fileLocationWeights, 'utf8'));
        const weights: tf.Tensor = tf.tensor(data);
        return weights
    }

    private getModelDirectory(){          
        const modelDir: string = path.join(__dirname, '../../actor-rdf-join-inner-multi-reinforcement-learning/model');
        return modelDir;
    }

    
}

export class QValueNetwork{
    // Hardcoded for easy  of testing, will change to config later!
    // networkLayersTest: ["dropout", number]|["activation"|"dense", DenseOwnImplementation|tf.layers.Layer][]
    networkLayers: (DenseOwnImplementation|tf.layers.Layer)[];
    public constructor(inputSize: number, outputSize: number){
        // Again hardcoded
        this.networkLayers = [];
    }

    public flushQNetwork(){
        for (let i = 0; i<this.networkLayers.length;i++){
            const layer: DenseOwnImplementation|tf.layers.Layer = this.networkLayers[i];
            if (layer instanceof DenseOwnImplementation){
                layer.disposeLayer();
            }
        }
        this.networkLayers = [];
    }

    public forwardPassFromConfig(input: tf.Tensor, kwargs: any){
        return tf.tidy(()=>{
            let x: tf.Tensor = this.networkLayers[0].apply(input) as tf.Tensor;
            for (let i=1;i<this.networkLayers.length;i++){
                x = this.networkLayers[i].call(x, {training: kwargs.training==undefined? false : kwargs.training}) as tf.Tensor;
            }
            return x;    
        })
    }

    public init(qValueNetworkConfig: ILayerConfig, weightsDir: string ){
        let denseIdx = 0;
        for (let k=0;k<qValueNetworkConfig.layers.length;k++){
            if (qValueNetworkConfig.layers[k].type=='dense'){
                const newLayer: DenseOwnImplementation = new DenseOwnImplementation(qValueNetworkConfig.layers[k].inputSize!, 
                qValueNetworkConfig.layers[k].outputSize!, 'dW'+k, 'dB'+k);
                if (qValueNetworkConfig.weightsConfig.loadWeights){
                    newLayer.loadWeights(weightsDir+qValueNetworkConfig.weightsConfig.weightLocation+denseIdx+'.txt');
                    newLayer.loadBias(weightsDir+qValueNetworkConfig.weightsConfig.weightLocationBias+denseIdx+'.txt');
                    denseIdx+=1;
                }                   
                this.networkLayers.push(newLayer)
            }
            if (qValueNetworkConfig.layers[k].type=='activation'){
                this.networkLayers.push(tf.layers.activation({activation: qValueNetworkConfig.layers[k].activation!}));
            }
            if(qValueNetworkConfig.layers[k].type=='dropout'){
                this.networkLayers.push(tf.layers.dropout({rate: qValueNetworkConfig.layers[k].rate!}));
            }
        }
    }
    public initRandom(qValueNetworkConfig: ILayerConfig, weightsDir: string){
        for (let k=0;k<qValueNetworkConfig.layers.length;k++){
            if (qValueNetworkConfig.layers[k].type=='dense'){
                const newLayer: DenseOwnImplementation = new DenseOwnImplementation(qValueNetworkConfig.layers[k].inputSize!, 
                qValueNetworkConfig.layers[k].outputSize!, 'dW'+k, 'dB'+k);
                this.networkLayers.push(newLayer)
            }
            if (qValueNetworkConfig.layers[k].type=='activation'){
                this.networkLayers.push(tf.layers.activation({activation: qValueNetworkConfig.layers[k].activation!}));
            }
            if(qValueNetworkConfig.layers[k].type=='dropout'){
                this.networkLayers.push(tf.layers.dropout({rate: qValueNetworkConfig.layers[k].rate!}));
            }
        }
    }

    public saveNetwork(qValueNetworkConfig: ILayerConfig, weightsDir: string, saveLocation?: string ){
        let denseIdx = 0;
        try{
            for (let i = 0; i<qValueNetworkConfig.layers.length; i++){
                if (qValueNetworkConfig.layers[i].type=='dense'){
                    const layerToSave: DenseOwnImplementation = this.networkLayers[i] as DenseOwnImplementation;
                    if (saveLocation){
                        layerToSave.saveWeights(weightsDir+saveLocation+"/denseWeights/weight"+denseIdx+".txt");
                        layerToSave.saveBias(weightsDir+saveLocation+"/denseBias/weight"+denseIdx+".txt");
                    }
                    else{
                        layerToSave.saveWeights(weightsDir+qValueNetworkConfig.weightsConfig.weightLocation+denseIdx+'.txt');
                        layerToSave.saveBias(weightsDir+qValueNetworkConfig.weightsConfig.weightLocationBias+denseIdx+'.txt');    
                    }
                    denseIdx+=1;
                }
            }
        }
        catch(err){
            console.log(err);
        }
    }
}