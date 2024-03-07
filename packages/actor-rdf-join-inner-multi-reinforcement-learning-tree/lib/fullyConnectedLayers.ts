import * as fs from 'fs';
import * as path from 'path';
import * as tf from '@tensorflow/tfjs-node';
import type { ILayerConfig } from './treeLSTM';

export class DenseOwnImplementation extends tf.layers.Layer {
  intermediateHeInit: tf.Tensor;
  heInitTerm: tf.Tensor;
  mWeights: tf.Variable;
  mBias: tf.Variable;
  inputDim: number;
  outputDim: number;

  public constructor(inputDim: number, outputDim: number, weightName?: string, biasName?: string, modelDirectory?: string) {
    super({});
    this.inputDim = inputDim;
    this.outputDim = outputDim;
    // Make intermediate value because we have to be able dispose to prevent memory leaks
    this.intermediateHeInit = tf.div(tf.tensor(2), tf.tensor(this.outputDim));
    this.heInitTerm = tf.sqrt(this.intermediateHeInit);
    // THIS ALSO SWITCHED DIMENSIONS
    this.mWeights = tf.variable(tf.mul(tf.randomNormal([ this.inputDim, this.outputDim ], 0, 1), this.heInitTerm), true);
    this.mBias = tf.variable(tf.mul(tf.randomNormal([ 1, outputDim ], 0, 1), this.heInitTerm), true);
  }
  // THIS SWITCH DIMENSIONS TO ACCOMEDATE LACK OF TRANSPOSE IN GCN
  call(inputs: tf.Tensor, kwargs: any) {
    return tf.tidy(() => tf.add(tf.matMul(inputs, this.mWeights), this.mBias.broadcastTo([ 1, this.outputDim ])));
  }

  getClassName() {
    return 'Dense Own';
  }

  public saveWeights(fileLocation: string) {
    const weights = this.mWeights.arraySync();
    fs.writeFileSync(fileLocation, JSON.stringify(weights));
  }

  public saveBias(loadPath: string) {
    const weights = this.mBias.arraySync();
    fs.writeFileSync(loadPath, JSON.stringify(weights));
  }

  public loadWeights(fileLocation: string): void {
    try {
      const weights = this.readWeightFile(fileLocation);
      this.mWeights.assign(weights);
    } catch (error) {
      throw error;
    }
  }

  public loadBias(fileLocation: string): void {
    try {
      const weights = this.readWeightFile(fileLocation);
      this.mBias.assign(weights);
    } catch (error) {
      throw error;
    }
  }

  public getWeights(trainableOnly?: boolean | undefined): tf.Tensor[] {
    return [ this.mWeights, this.mBias ];
  }

  public disposeLayer() {
    // Dispose all weights in layer
    this.intermediateHeInit.dispose(); this.heInitTerm.dispose();
    this.mWeights.dispose(); this.mBias.dispose();
  }

  private readWeightFile(fileLocationWeights: string) {
    const data: number[][] = JSON.parse(fs.readFileSync(fileLocationWeights, 'utf8'));
    const weights: tf.Tensor = tf.tensor(data);
    return weights;
  }
}

export class QValueNetwork {
  // Hardcoded for easy  of testing, will change to config later!
  // networkLayersTest: ["dropout", number]|["activation"|"dense", DenseOwnImplementation|tf.layers.Layer][]
  networkLayers: (DenseOwnImplementation | tf.layers.Layer)[];
  public constructor(inputSize: number, outputSize: number) {
    // Again hardcoded
    this.networkLayers = [];
  }

  public flushQNetwork() {
    for (let i = 0; i < this.networkLayers.length; i++) {
      const layer: DenseOwnImplementation | tf.layers.Layer = this.networkLayers[i];
      if (layer instanceof DenseOwnImplementation) {
        layer.disposeLayer();
      }
    }
    this.networkLayers = [];
  }

  public forwardPassFromConfig(input: tf.Tensor, kwargs: any) {
    return tf.tidy(() => {
      let x: tf.Tensor = this.networkLayers[0].apply(input) as tf.Tensor;
      for (let i = 1; i < this.networkLayers.length; i++) {
        x = this.networkLayers[i].call(x, { training: kwargs.training == undefined ? false : kwargs.training }) as tf.Tensor;
      }
      return x;
    });
  }

  public init(qValueNetworkConfig: ILayerConfig, modelDirectory: string) {
    let denseIdx = 0;
    for (let k = 0; k < qValueNetworkConfig.layers.length; k++) {
      if (qValueNetworkConfig.layers[k].type == 'dense') {
        const newLayer: DenseOwnImplementation = new DenseOwnImplementation(qValueNetworkConfig.layers[k].inputSize!,
          qValueNetworkConfig.layers[k].outputSize!,
          `dW${k}`,
          `dB${k}`);
        if (qValueNetworkConfig.weightsConfig.loadWeights) {
          newLayer.loadWeights(
            path.join(
              modelDirectory,
              'weights-dense',
              'dense',
              `${qValueNetworkConfig.weightsConfig.weightLocation + denseIdx}.txt`,
            ),
          );
          newLayer.loadBias(
            path.join(
              modelDirectory,
              'weights-dense',
              'bias',
              `${qValueNetworkConfig.weightsConfig.weightLocation + denseIdx}.txt`,
            ),
          );
          denseIdx += 1;
        }
        this.networkLayers.push(newLayer);
      }
      if (qValueNetworkConfig.layers[k].type == 'activation') {
        this.networkLayers.push(tf.layers.activation({ activation: qValueNetworkConfig.layers[k].activation! }));
      }
      if (qValueNetworkConfig.layers[k].type == 'dropout') {
        this.networkLayers.push(tf.layers.dropout({ rate: qValueNetworkConfig.layers[k].rate! }));
      }
    }
  }

  public initRandom(qValueNetworkConfig: ILayerConfig) {
    for (let k = 0; k < qValueNetworkConfig.layers.length; k++) {
      if (qValueNetworkConfig.layers[k].type == 'dense') {
        const newLayer: DenseOwnImplementation = new DenseOwnImplementation(qValueNetworkConfig.layers[k].inputSize!,
          qValueNetworkConfig.layers[k].outputSize!,
          `dW${k}`,
          `dB${k}`);
        this.networkLayers.push(newLayer);
      }
      if (qValueNetworkConfig.layers[k].type == 'activation') {
        this.networkLayers.push(tf.layers.activation({ activation: qValueNetworkConfig.layers[k].activation! }));
      }
      if (qValueNetworkConfig.layers[k].type == 'dropout') {
        this.networkLayers.push(tf.layers.dropout({ rate: qValueNetworkConfig.layers[k].rate! }));
      }
    }
  }

  public saveNetwork(qValueNetworkConfig: ILayerConfig, modelDirectory: string) {
    let denseIdx = 0;
    try {
      for (let i = 0; i < qValueNetworkConfig.layers.length; i++) {
        if (qValueNetworkConfig.layers[i].type == 'dense') {
          const layerToSave: DenseOwnImplementation = <DenseOwnImplementation> this.networkLayers[i];
          layerToSave.saveWeights(path.join(
            modelDirectory,
            'weights-dense',
            'dense',
            `${qValueNetworkConfig.weightsConfig.weightLocation + denseIdx}.txt`,
          ));
          layerToSave.saveBias(path.join(
            modelDirectory,
            'weights-dense',
            'bias',
            `${qValueNetworkConfig.weightsConfig.weightLocation + denseIdx}.txt`,
          ));
          denseIdx += 1;
        }
      }
    } catch (error) {
      console.log(error);
    }
  }
}
