import { Actor, Bus, IAction, IActorOutput, IActorTest, Mediator } from '@comunica/core';
import { GraphConvolutionLayer, GraphConvolutionModel } from '../lib/GraphConvolution';
import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import { DenseOwnImplementation } from '@comunica/actor-rdf-join-inner-multi-reinforcement-learning-tree/lib/fullyConnectedLayers';

describe('GraphConvolution', () => {
  let graphConvolutionLayer: GraphConvolutionLayer;
  let graphConvolutionModel: GraphConvolutionModel
  beforeEach(() => {
  });

  describe('A GraphConvolution Layer', () => {

    beforeEach(() => {
        graphConvolutionLayer = new GraphConvolutionLayer(3, 4, 'relu', undefined, __dirname)
        graphConvolutionLayer.loadWeights("weights-test/weightsLayerTest.txt"); 
    });

    it('should correctly load weights', () => {
        return expect(graphConvolutionLayer.getWeights()[0].arraySync()).toEqual([[1,2,3,4], [2,2,2,2], [1,2,1,2]]);
    });

    it('should correctly save weights', () => {
        graphConvolutionLayer.saveWeights('weights-test/savedWeightsTest.txt');
        const savedWeights = JSON.parse(fs.readFileSync(__dirname+'/weights-test/savedWeightsTest.txt', 'utf-8'));
        fs.unlinkSync(__dirname+'/weights-test/savedWeightsTest.txt');
        return expect(savedWeights).toEqual([[1,2,3,4], [2,2,2,2], [1,2,1,2]]);
    });

    it('should correctly perform forward pass', () => {
        const input = tf.tensor([[1, 2, -2], [-1, 1, 1]]);
        const adjacency = tf.tensor([[1, 1], [ 0, 1 ]]);
        const expectedResult = [[2.9135, 2.4137, 2.4992, 1.9994], [2.0000, 2.0000, 0.0000, 0.0000]];

        const outputLayer: number[][] = graphConvolutionLayer.call(input, adjacency).arraySync() as number[][];

        for (let i = 0; i<expectedResult.length; i++){
            for (let j = 0; j<expectedResult[0].length; j++){
                expect(outputLayer[i][j]).toBeCloseTo(expectedResult[i][j], 2)
            }
        }
        return expect(outputLayer[1][3]).toBeCloseTo(expectedResult[1][3], 2);
    });
  });

  describe('A GraphConvolution Model', () => {

    beforeEach(() => {
        graphConvolutionModel = new GraphConvolutionModel(__dirname);
        
    });

    it('should correctly load config', () => {
        expect(graphConvolutionModel.getConfig()).toEqual({
            weightLocation: "model-from-config-weights-test",
            layers:[
                {type: "gcn", inputSize: 3, outputSize: 4, activation: "relu", weightLocation: "gcn-weights-test-1.txt"},
                {type: "dense", inputSize: 4, outputSize: 1, weightLocation: "dense-weights-test-1.txt", biasLocation: "dense-bias-test-1.txt"},
                {type: "activation", activation: "relu", inputSize: 1, outputSize: 1}
            ]
        });
    });

    it('should correctly instantiate layers', () => {
        graphConvolutionModel.initModelRandom();
        const modelLayers = graphConvolutionModel.getLayers();
        // Correct gcn layer instance
        expect(modelLayers[0][1]).toEqual('gcn');
        expect(modelLayers[0][0] instanceof GraphConvolutionLayer).toEqual(true);
        // Correct dense layer instance
        expect(modelLayers[1][1]).toEqual('dense');
        expect(modelLayers[1][0] instanceof DenseOwnImplementation).toEqual(true);
        // Correct activation
        expect(modelLayers[2][1]).toEqual('activation');
        expect(modelLayers[2][0] instanceof tf.layers.Layer).toEqual(true);
    });

    it('should correctly load weights', () => {
        graphConvolutionModel.initModelWeights();
        const modelLayers = graphConvolutionModel.getLayers();
        // Correct gcn layer instance
        expect(modelLayers[0][1]).toEqual('gcn');
        expect(modelLayers[0][0] instanceof GraphConvolutionLayer).toEqual(true);
        expect(modelLayers[0][0].getWeights()[0].arraySync()).toEqual([[1,2,3,4], [2,2,2,2], [1,2,1,2]]);
        // Correct dense layer instance
        expect(modelLayers[1][1]).toEqual('dense');
        expect(modelLayers[1][0] instanceof DenseOwnImplementation).toEqual(true);
        expect(modelLayers[1][0].getWeights()[0].arraySync()).toEqual([[1,-2,1,1]]);
        expect(modelLayers[1][0].getWeights()[1].arraySync()).toEqual([[1]]);
        // Correct activation
        expect(modelLayers[2][1]).toEqual('activation');
        expect(modelLayers[2][0] instanceof tf.layers.Layer).toEqual(true);
    });

    it('should correctly save weights', () => {
        graphConvolutionModel.initModelWeights();
        graphConvolutionModel.saveModel();

        const gcnWeightLocation = `${__dirname}/model-from-config-weights-test/gcn-saved-weights-test/gcn-weights-test-1.txt`;
        expect(JSON.parse(fs.readFileSync(gcnWeightLocation, 'utf-8'))).toEqual([[1,2,3,4], [2,2,2,2], [1,2,1,2]]);

        const denseWeightLocation = `${__dirname}/model-from-config-weights-test/dense-saved-weights-test/dense-weights-test-1.txt`;
        expect(JSON.parse(fs.readFileSync(denseWeightLocation, 'utf-8'))).toEqual([[1,-2, 1,1]]);

        const denseBiasLocation = `${__dirname}/model-from-config-weights-test/dense-saved-weights-test/dense-bias-test-1.txt`;
        expect(JSON.parse(fs.readFileSync(denseBiasLocation, 'utf-8'))).toEqual([[1]]);

        fs.unlinkSync(gcnWeightLocation);
        fs.unlinkSync(denseWeightLocation);
        fs.unlinkSync(denseBiasLocation);
    });

    it('should correctly perform forward pass', () => {
        const input = tf.tensor([[1, 2, -2], [-1, 1, 1]]);
        const adjacency = tf.tensor([[1, 1], [ 0, 1 ]]);
        const expectedResult = [[3.5847, 0]];

        graphConvolutionModel.initModelWeights();
        expect((<number[][]>graphConvolutionModel.forwardPass(input, adjacency).arraySync())[0][0]).toBeCloseTo(expectedResult[0][0]);
        expect((<number[][]>graphConvolutionModel.forwardPass(input, adjacency).arraySync())[0][1]).toBeCloseTo(expectedResult[0][1]);

    });
  });
});