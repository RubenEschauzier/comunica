import * as tf from '@tensorflow/tfjs-node';
import * as path from 'path';
import * as fs from 'fs';
import { input } from '@tensorflow/tfjs-node';
import { denseOptionsId, ILayerConfig } from './treeLSTM';

export class DenseOwnImplementation extends tf.layers.Layer{
    heInitTerm: tf.Tensor;
    mWeights: tf.Variable;
    mBias: tf.Variable;
    inputDim: number;
    outputDim: number;
    modelDirectory: string;


    public constructor(inputDim: number, outputDim: number, weightName?: string, biasName?: string){
        super({});
        this.inputDim = inputDim;
        this.outputDim = outputDim;
        this.heInitTerm = tf.sqrt(tf.div(tf.tensor(2), tf.tensor(this.outputDim)));
        this.mWeights = tf.variable(tf.mul(tf.randomNormal([this.outputDim, this.inputDim],0,1),this.heInitTerm), true);
        this.mBias = tf.variable(tf.mul(tf.randomNormal([outputDim,1],0,1), this.heInitTerm), true);
        this.modelDirectory = this.getModelDirectory();
    }

    call(inputs: tf.Tensor, kwargs: any){
        return tf.tidy(() => {
            return tf.add(tf.matMul(this.mWeights, inputs), this.mBias.broadcastTo([this.outputDim, 1]));
        });
    }
    getClassName() {
        return 'Dense Own';
    }

    public async saveWeights(fileLocation: string): Promise<void> {
        const weights = await this.mWeights.array();
        fs.writeFile(fileLocation, JSON.stringify(weights), function(err: any) {
            if(err) {
                return console.log(err);
            }
        }); 
    }
    public async saveBias(fileLocation: string): Promise<void> {
        const weights = await this.mBias.array();
        fs.writeFile(fileLocation, JSON.stringify(weights), function(err: any) {
            if(err) {
                return console.log(err);
            }
        }); 
    }

    public async loadWeights(fileLocation: string): Promise<void>{
        const weights = await this.readWeightFile(fileLocation);
        this.mWeights.assign(weights);
    }

    public async loadBias(fileLocation: string): Promise<void>{
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

export class QValueNetwork{
    // Hardcoded for easy  of testing, will change to config later!
    networkLayersTest: ["dropout", number]|["activation"|"dense", DenseOwnImplementation|tf.layers.Layer][]
    networkLayers: (DenseOwnImplementation|tf.layers.Layer)[];
    public constructor(inputSize: number, outputSize: number){
        // Again hardcoded
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

    public async init(qValueNetworkConfig: ILayerConfig, weightsDir: string ){
        let denseIdx = 0;
        for (let k=0;k<qValueNetworkConfig.layers.length;k++){
            if (qValueNetworkConfig.layers[k].type=='dense'){
                const newLayer: DenseOwnImplementation = new DenseOwnImplementation(qValueNetworkConfig.layers[k].inputSize!, 
                qValueNetworkConfig.layers[k].outputSize!, 'dW'+k, 'dB'+k);
                if (qValueNetworkConfig.weightsConfig.loadWeights){
                    await newLayer.loadWeights(weightsDir+qValueNetworkConfig.weightsConfig.weightLocation+denseIdx+'.txt');
                    await newLayer.loadBias(weightsDir+qValueNetworkConfig.weightsConfig.weightLocationBias+denseIdx+'.txt');
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
        let denseIdx = 0;
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
        for (let i = 0; i<qValueNetworkConfig.layers.length; i++){
            if (qValueNetworkConfig.layers[i].type=='dense'){
                const layerToSave: DenseOwnImplementation = this.networkLayers[i] as DenseOwnImplementation;
                if (saveLocation){
                    layerToSave.saveWeights(weightsDir+saveLocation+"denseWeights/weight"+denseIdx+".txt");
                    layerToSave.saveBias(weightsDir+saveLocation+"denseBias/weight"+denseIdx+".txt");
                }
                layerToSave.saveWeights(weightsDir+qValueNetworkConfig.weightsConfig.weightLocation+denseIdx+'.txt');
                layerToSave.saveBias(weightsDir+qValueNetworkConfig.weightsConfig.weightLocationBias+denseIdx+'.txt');
                denseIdx+=1;
            }
        }
    }
}