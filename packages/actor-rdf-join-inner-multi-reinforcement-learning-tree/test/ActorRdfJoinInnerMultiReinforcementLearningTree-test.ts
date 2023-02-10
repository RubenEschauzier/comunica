import {binaryTreeLSTM} from '../lib/treeLSTM';
import * as fs from 'fs';
import * as path from 'path';
import {ActorRdfJoinInnerMultiReinforcementLearningTree} from '@comunica/actor-rdf-join-inner-multi-reinforcement-learning-tree';
import { Bus } from '@comunica/core';
import * as tf from '@tensorflow/tfjs-node';


describe('ActorRdfJoinInnerMultiReinforcementLearningTree', () => {
  let bus: any;

  beforeEach(() => {
    bus = new Bus({ name: 'bus' });
  });

  describe('An ActorRdfJoinInnerMultiReinforcementLearningTree instance', () => {
    let actor: ActorRdfJoinInnerMultiReinforcementLearningTree;

    describe('The ActorRdfJoinMultiSmallest module', () => {
      it('should be a function', () => {
        expect(ActorRdfJoinInnerMultiReinforcementLearningTree).toBeInstanceOf(Function);
      });
    });
});
});

describe('treeLSTMLayer', ()=>{
  const testConfig: IWeightConfig = readConfigFile();
  const parsedConfig = parseConfigFile(testConfig);
  const treeLSTMLayer = new binaryTreeLSTM(testConfig.outputSize);
  describe('Test Weights loading', () =>{
    it('Size array loaded weights and layer weights should be equal', () => {
      return expect(parsedConfig.weightsToAdd.length).toBe(treeLSTMLayer.allWeights.length); 
    });
  });
  treeLSTMLayer.setWeights(parsedConfig.weightsToAdd);
  describe('Test first and last input weight for equality', () => {
    it('First input weight should be equal to config', ()=>{
      return expect(Array.from(treeLSTMLayer.IWInput.dataSync())).toEqual(Array.from(tf.tensor(testConfig.inputWeights[0]).dataSync()));
    });
    it('Last input weight should be equal to config', ()=>{
      return expect(Array.from(treeLSTMLayer.UWInput.dataSync())).toEqual(Array.from(tf.tensor(testConfig.inputWeights[testConfig.inputWeights.length-1]).dataSync()));
    });
  });
  describe('Test first and last input gate weight for equality to config', () => {
    it('First input gate weight should be equal to config', ()=>{
      return expect(Array.from(treeLSTMLayer.lhIU.dataSync())).toEqual(Array.from(tf.tensor(testConfig.inputGateWeights[0]).dataSync()))
    });
    it('Last input gate weight should be equal to config', ()=>{
      return expect(Array.from(treeLSTMLayer.rhIU.dataSync())).toEqual(Array.from(tf.tensor(testConfig.inputGateWeights[testConfig.inputGateWeights.length-1]).dataSync()))
    });
  describe('Test first and last forget gate weight for equality to config', () => {
    it('First forget gate weight should be equal to config', ()=>{
      return expect(Array.from(treeLSTMLayer.lhFUL.dataSync())).toEqual(Array.from(tf.tensor(testConfig.forgetGateWeights[0]).dataSync()))
    });
    it('Last forget gate weight should be equal to config', ()=>{
      return expect(Array.from(treeLSTMLayer.rhFUR.dataSync())).toEqual(Array.from(tf.tensor(testConfig.forgetGateWeights[testConfig.forgetGateWeights.length-1]).dataSync()));
    });
  });
  describe('Test first and last output gate weight for equality to config', () => {
    it('First output gate weight should be equal to config', ()=>{
      return expect(Array.from(treeLSTMLayer.lhOU.dataSync())).toEqual(Array.from(tf.tensor(testConfig.outputGateWeights[0]).dataSync()));
    });
    it('Last output gate weight should be equal to config', ()=>{
      return expect(Array.from(treeLSTMLayer.rhOU.dataSync())).toEqual(Array.from(tf.tensor(testConfig.outputGateWeights[testConfig.outputGateWeights.length-1]).dataSync()));
    });
  });
  describe('Test first and last U gate weight for equality to config', () => {
    it('First U gate weight should be equal to config', ()=>{
      return expect(Array.from(treeLSTMLayer.lhUU.dataSync())).toEqual(Array.from(tf.tensor(testConfig.uGateWeights[0]).dataSync()));
    });
    it('Last U gate weight should be equal to config', ()=>{
      return expect(Array.from(treeLSTMLayer.rhUU.dataSync())).toEqual(Array.from(tf.tensor(testConfig.uGateWeights[testConfig.uGateWeights.length-1]).dataSync()));
    });
  });
  describe('Test fist and last bias vector for equality to config', () => {
    it('First bias vector should be equal to config', ()=>{
      return expect(Array.from(treeLSTMLayer.bI.dataSync())).toEqual(Array.from(tf.tensor(testConfig.biasVectors[0]).reshape([testConfig.outputSize, 1]).dataSync()))
    });
    it('Last bias vector gate weight should be equal to config', ()=>{
      return expect(Array.from(treeLSTMLayer.bU.dataSync())).toEqual(Array.from(tf.tensor(testConfig.biasVectors[testConfig.biasVectors.length-1]).reshape([testConfig.outputSize, 1]).dataSync()))
    });
  });
  // Size 5 list with tensors [inputCurrentNode, lhs, rhs, lmc, rmc]
  const splitInputs: tf.Tensor[] = tf.unstack(parsedConfig.inputs);

  // Current node input (0 in our use case)
  const inputsNode: tf.Tensor = splitInputs[0];

  // The hidden state and memory cell inputs from left and right node
  const lhs: tf.Tensor = splitInputs[1]; const rhs: tf.Tensor = splitInputs[2];
  const lmc: tf.Tensor = splitInputs[3]; const rmc: tf.Tensor = splitInputs[4]
              
  const inputGate = treeLSTMLayer.getGate(inputsNode, lhs, 
    rhs, treeLSTMLayer.IWInput, treeLSTMLayer.lhIU, treeLSTMLayer.rhIU, treeLSTMLayer.bI);
  const forgetGateL: tf.Tensor = treeLSTMLayer.getGate(inputsNode, lhs, rhs, treeLSTMLayer.FWInput, treeLSTMLayer.lhFUL, 
    treeLSTMLayer.rhFUL, treeLSTMLayer.bF);
  const forgetGateR: tf.Tensor = treeLSTMLayer.getGate(inputsNode, lhs, rhs, treeLSTMLayer.FWInput, treeLSTMLayer.lhFUR, 
    treeLSTMLayer.rhFUR, treeLSTMLayer.bF);
  const outputGate: tf.Tensor = treeLSTMLayer.getGate(inputsNode, lhs, rhs, treeLSTMLayer.OWInput, 
    treeLSTMLayer.lhOU, treeLSTMLayer.rhOU, treeLSTMLayer.bO);
  const uGate = treeLSTMLayer.getGate(inputsNode, lhs, 
    rhs, treeLSTMLayer.UWInput, treeLSTMLayer.lhUU, treeLSTMLayer.rhUU, treeLSTMLayer.bU);

  const memoryCell = treeLSTMLayer.getMemoryCell(inputGate, tf.tanh(uGate), forgetGateL, forgetGateR, lmc, rmc);
  const hiddenState: tf.Tensor = treeLSTMLayer.getHiddenState(outputGate, memoryCell);


  describe('Test Input Gate', () => {
    it('Input gate should match expected output', ()=>{
    
      return expect(Array.from(inputGate.dataSync())).toEqual(Array.from(parsedConfig.expectedValues.inputGate.dataSync()));
    });
  });

  describe('Test Forget Gate L', () => {
    it('Forget gate L should match expected output', ()=>{
      return expect(Array.from(forgetGateL.dataSync())).toEqual(Array.from(parsedConfig.expectedValues.forgetGateL.dataSync()));
    });
  });

  describe('Test Forget Gate R', () => {
    it('Forget gate R should match expected output', ()=>{
      return expect(Array.from(forgetGateR.dataSync())).toEqual(Array.from(parsedConfig.expectedValues.forgetGateR.dataSync()));
    });
  });

  describe('Test Output Gate', () => {
    it('Output gate should match expected output', ()=>{
      return expect(Array.from(outputGate.dataSync())).toEqual(Array.from(parsedConfig.expectedValues.outputGate.dataSync()));
    });
  });
  describe('Test U Gate', () => {
    it('U gate should match expected output', ()=>{
      return expect(Array.from(uGate.dataSync())).toEqual(Array.from(parsedConfig.expectedValues.uGate.dataSync()));
    });
  });
  // THIS IS OUTPUTSIZE SPECIFIC, BIT IFFY 
  describe('Test Memory Cell', () =>{
    it('Memory cell output should match expected output', ()=>{
      return expect(Array.from(memoryCell.dataSync())[0]).toBeCloseTo(Array.from(parsedConfig.expectedValues.memoryCell.dataSync())[0]);
    });
    it('Memory cell output should match expected output', ()=>{
      return expect(Array.from(memoryCell.dataSync())[1]).toBeCloseTo(Array.from(parsedConfig.expectedValues.memoryCell.dataSync())[1]);
    });
  });
  describe('Test Hidden State', () =>{
    it('Hidden state output should match expected output', ()=>{
      return expect(Array.from(hiddenState.dataSync())[0]).toBeCloseTo(Array.from(parsedConfig.expectedValues.hiddenState.dataSync())[0]);
    });
    it('Hidden state output should match expected output', ()=>{
      return expect(Array.from(hiddenState.dataSync())[1]).toBeCloseTo(Array.from(parsedConfig.expectedValues.hiddenState.dataSync())[1]);
    });
  });
  const outputTotalCall = treeLSTMLayer.call(parsedConfig.inputs);
  describe('Test LSTM-tree call memory cell output', () =>{
    it('Output should match expected output', ()=>{
      return expect(Array.from(outputTotalCall[1].dataSync())[0]).toBeCloseTo(Array.from(memoryCell.dataSync())[0]);
    });
    it('Output state output should match expected output', ()=>{
      return expect(Array.from(outputTotalCall[1].dataSync())[1]).toBeCloseTo(Array.from(memoryCell.dataSync())[1]);
    });
  });

  describe('Test LSTM-tree call hidden state output', () =>{
    it('Output should match expected output', ()=>{
      return expect(Array.from(outputTotalCall[0].dataSync())[0]).toBeCloseTo(Array.from(hiddenState.dataSync())[0]);
    });
    it('Output state output should match expected output', ()=>{
      return expect(Array.from(outputTotalCall[0].dataSync())[1]).toBeCloseTo(Array.from(hiddenState.dataSync())[1]);
    });
  });
  // describe('Test Input Gate', () => {
  //   it('Input gate should match expected output', ()=>{
  //     const testOutput = treeLSTMLayer.getGate(inputsNode, lhs, 
  //       rhs, treeLSTMLayer.IWInput, treeLSTMLayer.lhIU, treeLSTMLayer.rhIU, treeLSTMLayer.bI);
    
  //     return expect(Array.from(testOutput.dataSync())).toEqual(Array.from(parsedConfig.expectedValues.inputGate.dataSync()));
  //   });
  // });

  });

});

function readConfigFile(){
  let data = fs.readFileSync(path.resolve(__dirname, 'weightsUnitTest.json'), "utf-8");
  let testConfig = JSON.parse(data);
  return testConfig
}

function parseConfigFile(config: IWeightConfig): IParsedConfig{
  const weightsToAdd: tf.Tensor[] = [];
  const actualValues: IExpectedValues = emptyExpectedResults();

  for (const weightsInput of config.inputWeights){
    weightsToAdd.push(tf.tensor(weightsInput));
  }
  for (const weigthsInputGate of config.inputGateWeights){
    weightsToAdd.push(tf.tensor(weigthsInputGate));
  }
  for (const weightsForget of config.forgetGateWeights){
    weightsToAdd.push(tf.tensor(weightsForget));
  }
  for (const weightsOutput of config.outputGateWeights){
    weightsToAdd.push(tf.tensor(weightsOutput));
  }
  for (const weightsU of config.uGateWeights){
    weightsToAdd.push(tf.tensor(weightsU));
  }
  for (const bias of config.biasVectors){
    weightsToAdd.push(tf.tensor(bias).reshape([config.outputSize, 1]));
  }
  const inputsList: tf.Tensor[] = [];
  for (const inputValues of config.inputValues){
    inputsList.push(tf.tensor(inputValues).reshape([config.outputSize,1]));
  }
 const inputsTensor: tf.Tensor =tf.stack(inputsList);

  for(const expectedValueI of config.expectedValueInputGate){
    actualValues.inputGate = tf.tensor(expectedValueI);
  }
  for(const expectedValueFL of config.expectedValueForgetGateL){
    actualValues.forgetGateL = tf.tensor(expectedValueFL);
  }
  for(const expectedValueFR of config.expectedValueForgetGateR){
    actualValues.forgetGateR = tf.tensor(expectedValueFR);
  }

  for(const expectedValueO of config.expectedValueOutputGate){
    actualValues.outputGate = tf.tensor(expectedValueO);
  }
  for(const expectedValueU of config.expectedValueUGate){
    actualValues.uGate = tf.tensor(expectedValueU);
  }
  for(const expectedValueM of config.expectedValueMemoryCell){
    actualValues.memoryCell = tf.tensor(expectedValueM);
  }
  for(const expectedValueH of config.expectedValueHiddenState){
    actualValues.hiddenState = tf.tensor(expectedValueH);
  }
  return {weightsToAdd: weightsToAdd, inputs: inputsTensor, expectedValues:actualValues};
}



function emptyExpectedResults(): IExpectedValues {
  return {
    inputGate: tf.tensor([]),
    forgetGateL: tf.tensor([]),
    forgetGateR: tf.tensor([]),
    outputGate: tf.tensor([]),
    uGate: tf.tensor([]),
    memoryCell: tf.tensor([]),
    hiddenState: tf.tensor([]),
  }
}

interface IWeightConfig{
  outputSize: number;
  inputWeights: number[][][];
  inputGateWeights: number[][][];
  forgetGateWeights: number[][][];
  outputGateWeights: number[][][];
  uGateWeights: number[][][];
  biasVectors: number[][];
  inputValues: number[][],
  expectedValueInputGate: number[][],
  expectedValueForgetGateL: number[][],
  expectedValueForgetGateR: number[][],
  expectedValueOutputGate: number[][],
  expectedValueUGate: number[][],
  expectedValueMemoryCell: number[][],
  expectedValueHiddenState: number[][],
}

interface IParsedConfig{
  weightsToAdd: tf.Tensor[];
  inputs: tf.Tensor;
  expectedValues: IExpectedValues;
}

interface IExpectedValues{
  inputGate: tf.Tensor;
  forgetGateL: tf.Tensor;
  forgetGateR: tf.Tensor;
  outputGate: tf.Tensor;
  uGate: tf.Tensor;
  memoryCell: tf.Tensor;
  hiddenState: tf.Tensor;
}
