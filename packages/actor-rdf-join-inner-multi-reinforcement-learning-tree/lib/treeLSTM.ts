import { IResultSetRepresentation } from '@comunica/mediator-join-reinforcement-learning';
import { DenseOwnImplementation, QValueNetwork } from './fullyConnectedLayers';
import * as tf from '@tensorflow/tfjs-node';
import * as path from 'path';
import * as fs from 'fs';
import { ISingleResultSetRepresentation } from './ActorRdfJoinInnerMultiReinforcementLearningTree';

export class BinaryTreeLSTM extends tf.layers.Layer{
    IWInput: tf.Variable;
    FWInput: tf.Variable;
    OWInput: tf.Variable;
    UWInput: tf.Variable;

    lhIU: tf.Variable;
    lhFUL: tf.Variable;
    lhFUR: tf.Variable;
    lhOU: tf.Variable;
    lhUU: tf.Variable;

    rhIU: tf.Variable;
    rhFUL: tf.Variable;
    rhFUR: tf.Variable;

    rhOU: tf.Variable;
    rhUU: tf.Variable;

    bI: tf.Variable;
    bF: tf.Variable;
    bO: tf.Variable;
    bU: tf.Variable;
    
    numUnits: number;
    activationLayer: any;
    heInitTerm: tf.Tensor;
    allWeights: tf.Variable[];

    public static className: string = 'binaryTreeLSTM'; 
    /**
     * Constructor for the binary Tree LSTM, note that in our case we don't use input, since a intermediate join representation is entirely built from preceding joins.
     * To save computations we could remove this?
     * @param numUnits 
     */
    constructor(numUnits: number, activation: activationOptionsId, layerName?: string){
        super({});
        this.numUnits=numUnits;
        this.activationLayer = tf.layers.activation({activation: activation});

        // Initialisation term
        this.heInitTerm = tf.sqrt(tf.div(tf.tensor(2), tf.tensor(this.numUnits)));

        // Input Layers can probably be deleted
        this.IWInput = tf.variable(tf.mul(tf.randomNormal([this.numUnits, this.numUnits],0,1),this.heInitTerm), true,
        "IWb");
        this.FWInput = tf.variable(tf.mul(tf.randomNormal([this.numUnits, this.numUnits],0,1),this.heInitTerm), true,
        "FWb");
        this.OWInput = tf.variable(tf.mul(tf.randomNormal([this.numUnits, this.numUnits],0,1),this.heInitTerm), true,
        "OWb");
        this.UWInput = tf.variable(tf.mul(tf.randomNormal([this.numUnits, this.numUnits],0,1),this.heInitTerm), true,
        "UWb");

        // Weights for left and right side of binary tree lstm
        this.lhIU = tf.variable(tf.mul(tf.randomNormal([this.numUnits, this.numUnits],0,1),this.heInitTerm), true,
        "lhIU");
        this.lhOU = tf.variable(tf.mul(tf.randomNormal([this.numUnits, this.numUnits],0,1),this.heInitTerm), true,
        "lhOU");
        this.lhUU = tf.variable(tf.mul(tf.randomNormal([this.numUnits, this.numUnits],0,1),this.heInitTerm), true,
        "lhUU");
        this.rhIU = tf.variable(tf.mul(tf.randomNormal([this.numUnits, this.numUnits],0,1),this.heInitTerm), true,
        "rhIU");
        this.rhOU = tf.variable(tf.mul(tf.randomNormal([this.numUnits, this.numUnits],0,1),this.heInitTerm), true,
        "rhOU");
        this.rhUU = tf.variable(tf.mul(tf.randomNormal([this.numUnits, this.numUnits],0,1),this.heInitTerm), true,
        "rhUU");

        // Forget Gate Weights (4 total)
        this.lhFUL = tf.variable(tf.mul(tf.randomNormal([this.numUnits, this.numUnits],0,1),this.heInitTerm), true,
        "lhFUL");
        this.rhFUL = tf.variable(tf.mul(tf.randomNormal([this.numUnits, this.numUnits],0,1),this.heInitTerm), true,
        "rhFUL");
        this.lhFUR = tf.variable(tf.mul(tf.randomNormal([this.numUnits, this.numUnits],0,1),this.heInitTerm), true,
        "lhFUR");
        this.rhFUR = tf.variable(tf.mul(tf.randomNormal([this.numUnits, this.numUnits],0,1),this.heInitTerm), true,
        "rhFUR");


        // Bias vectors 
        this.bI = tf.variable(tf.mul(tf.randomNormal([this.numUnits, 1],0,1),this.heInitTerm), true,
        "bIb");
        this.bF = tf.variable(tf.mul(tf.randomNormal([this.numUnits, 1],0,1),this.heInitTerm), true,
        "bfb");
        this.bO = tf.variable(tf.mul(tf.randomNormal([this.numUnits, 1],0,1),this.heInitTerm), true,
        "bOb");
        this.bU = tf.variable(tf.mul(tf.randomNormal([this.numUnits, 1],0,1),this.heInitTerm), true,
        "bUb");

        this.allWeights = [ this.IWInput, this.FWInput, this.OWInput, this.UWInput, this.lhIU, this.rhIU, this.lhFUL, this.lhFUR, this.rhFUL, this.rhFUR, this.lhOU, this.rhOU,
            this.lhUU, this.rhUU, this.bI, this.bF, this.bO, this.bU]

        // Dispose after usage
        this.heInitTerm.dispose();
    }

    setWeights(weights: tf.Tensor<tf.Rank>[]): void {
        // Enter weights like the all weights list
        let i = 0;
        for (const wVar of this.allWeights){
            wVar.assign(weights[i]);
            i+=1;
        }
    }

    call(inputs: tf.Tensor): tf.Tensor[]{
        // TODO: COMPARE WITH EXISTING IMPLEMENTATIONS
        // Flexible configuration of model and saving
        return tf.tidy(()=> {
            // Size 5 list with tensors [inputCurrentNode, lhs, rhs, lmc, rmc]
            const splitInputs: tf.Tensor[] = tf.unstack(inputs);
            
            // Current node input (0 in our use case)
            const inputsNode: tf.Tensor = splitInputs[0];

            // The hidden state and memory cell inputs from left and right node
            const lhs: tf.Tensor = splitInputs[1]; const rhs: tf.Tensor = splitInputs[2];
            const lmc: tf.Tensor = splitInputs[3]; const rmc: tf.Tensor = splitInputs[4]
            
            const inputGate: tf.Tensor = this.activationLayer.apply(this.getGate(inputsNode, lhs, rhs, this.IWInput, this.lhIU, this.rhIU, this.bI));
            const forgetGateL: tf.Tensor = this.activationLayer.apply(this.getGate(inputsNode, lhs, rhs, this.FWInput, this.lhFUL, this.rhFUL, this.bF));
            const forgetGateR: tf.Tensor = this.activationLayer.apply(this.getGate(inputsNode, lhs, rhs, this.FWInput, this.lhFUR, this.rhFUR, this.bF));
            const outputGate: tf.Tensor = this.activationLayer.apply(this.getGate(inputsNode, lhs, rhs, this.OWInput, this.lhOU, this.rhOU, this.bO));
            const uGate: tf.Tensor = tf.tanh(this.getGate(inputsNode, lhs, rhs, this.UWInput, this.lhUU, this.rhUU, this.bU));

            const memoryCell: tf.Tensor = this.getMemoryCell(inputGate, uGate, forgetGateL, forgetGateR, lmc, rmc);
            const hiddenState: tf.Tensor = this.getHiddenState(outputGate, memoryCell);

            return [hiddenState, memoryCell];
        });
    }

    getGate(input:tf.Tensor, lhs:tf.Tensor, rhs: tf.Tensor, inputWeight: tf.Variable, lhUWeight: tf.Variable, rhUWeight: tf.Variable, biasVec: tf.Variable){
        // Calculate i_{j} = W_{input} * x_{j} + \sum_{k=1}^{2} U_{k}^{i} h_{jk} + b^{i}
        // W_{input} * x_{j} = inRepI, \sum_{k=1}^{2} U_{k}^{i} h_{jk}=hsRepI, i_{j} = inputGate

        return tf.tidy(()=>{
            const inRepG: tf.Tensor = tf.matMul(inputWeight, input);
            const hsRepG: tf.Tensor = tf.add(tf.matMul(lhUWeight, lhs), tf.matMul(rhUWeight, rhs));
            const gateRep: tf.Tensor = tf.add(tf.add(inRepG, hsRepG), biasVec);
            return gateRep    
        });
    }
    getMemoryCell(inputGate:tf.Tensor, uGate: tf.Tensor, forgetGateL: tf.Tensor, forgetGateR: tf.Tensor, lmc: tf.Tensor, rmc: tf.Tensor){
        return tf.tidy(()=>{
            return tf.add(tf.mul(inputGate, uGate), tf.add(tf.mul(forgetGateL, lmc), tf.mul(forgetGateR,rmc)));
        });
    }
    getHiddenState(outputGate:tf.Tensor, memoryCell:tf.Tensor){
        return tf.tidy(()=>{
            return tf.mul(outputGate, tf.tanh(memoryCell));
        });
    }
    public loadWeights(fileLocation: string){
        const weightValues: number[][][] = JSON.parse(fs.readFileSync(fileLocation, 'utf8'));
        for (let i = 0;i<weightValues.length;i++){
            this.allWeights[i].assign(tf.tensor(weightValues[i]));
        }
    }

    public saveWeights(fileLocation: string){
        const weightValues: number[][][] = this.allWeights.map(x=>{
            return x.arraySync();
        }) as number[][][];
        fs.writeFileSync(fileLocation, JSON.stringify(weightValues));
    }

}

export class ChildSumTreeLSTM extends tf.layers.Layer{
    numUnits: number;
    heInitTerm: tf.Tensor;
    allWeights: tf.Variable[];

    IWInput: tf.Variable;
    FWInput: tf.Variable;
    OWInput: tf.Variable;
    UWInput: tf.Variable;

    IU: tf.Variable;
    FU: tf.Variable;
    OU: tf.Variable;
    UU: tf.Variable;

    bI: tf.Variable;
    bF: tf.Variable;
    bO: tf.Variable;
    bU: tf.Variable;

    activationLayer: any;

    public static className: string = 'childSumLSTM'; 

    constructor(numUnits: number, activation: activationOptionsId, layerName?: string){
        super({});
        this.numUnits=numUnits;
        this.activationLayer = tf.layers.activation({activation: activation});

        this.heInitTerm = tf.sqrt(tf.div(tf.tensor(2), tf.tensor(this.numUnits)));
        this.IWInput = tf.variable(tf.mul(tf.randomNormal([this.numUnits, this.numUnits],0,1),this.heInitTerm), true,
        "IW");
        this.FWInput = tf.variable(tf.mul(tf.randomNormal([this.numUnits, this.numUnits],0,1),this.heInitTerm), true,
        "FW");
        this.OWInput = tf.variable(tf.mul(tf.randomNormal([this.numUnits, this.numUnits],0,1),this.heInitTerm), true,
        "OW");
        this.UWInput = tf.variable(tf.mul(tf.randomNormal([this.numUnits, this.numUnits],0,1),this.heInitTerm), true,
        "UW");

        this.IU = tf.variable(tf.mul(tf.randomNormal([this.numUnits, this.numUnits],0,1),this.heInitTerm), true,
        "IU");
        this.FU = tf.variable(tf.mul(tf.randomNormal([this.numUnits, this.numUnits],0,1),this.heInitTerm), true,
        "FU");
        this.OU = tf.variable(tf.mul(tf.randomNormal([this.numUnits, this.numUnits],0,1),this.heInitTerm), true,
        "OU");
        this.UU = tf.variable(tf.mul(tf.randomNormal([this.numUnits, this.numUnits],0,1),this.heInitTerm), true,
        "UU");

        this.bI = tf.variable(tf.mul(tf.randomNormal([this.numUnits, 1],0,1),this.heInitTerm), true,
        "bI");
        this.bF = tf.variable(tf.mul(tf.randomNormal([this.numUnits, 1],0,1),this.heInitTerm), true,
        "bF");
        this.bO = tf.variable(tf.mul(tf.randomNormal([this.numUnits, 1],0,1),this.heInitTerm), true,
        "bO");
        this.bU = tf.variable(tf.mul(tf.randomNormal([this.numUnits, 1],0,1),this.heInitTerm), true,
        "bU");

        this.allWeights = [ this.IWInput, this.FWInput, this.OWInput, this.UWInput, this.IU, this.FU, this.OU, this.UU, this.bI, this.bF, this.bO, this.bU];
    }
    /**
     * 
     * @param inputs Inputs a list of stacked tensors as follows: [inputFeature, child hidden states, child memory cells]
     * 
     */
    
    call(inputs: tf.Tensor[]){
        return tf.tidy(()=>{
            
            const inputFeature: tf.Tensor = inputs[0]; const hiddenStatesChildren: tf.Tensor = inputs[1]; const memoryCellsChildren: tf.Tensor = inputs[2];
            // Shapes: [ 4, 1 ] [ 2, 4, 1 ] [ 2, 4, 1 ]

            const hiddenStateTilde: tf.Tensor = tf.sum(hiddenStatesChildren, 0);
    
            const inputGate: tf.Tensor = this.activationLayer.apply(this.getGate(inputFeature, hiddenStateTilde, this.IWInput, this.IU, this.bU));
            const forgetGates: tf.Tensor = this.activationLayer.apply(this.getGate(inputFeature, hiddenStatesChildren, this.FWInput, this.FU, this.bF));
            const outputGate: tf.Tensor = this.activationLayer.apply(this.getGate(inputFeature, hiddenStateTilde, this.OWInput, this.OU, this.bO));
            const uGate: tf.Tensor = tf.tanh(this.getGate(inputFeature, hiddenStateTilde, this.UWInput, this.UU, this.bU));
    
            const memoryCellCurrent: tf.Tensor = this.getMemoryCell(inputGate, uGate, forgetGates, memoryCellsChildren);
            const hiddenStateCurrent: tf.Tensor = this.getHiddenState(outputGate, memoryCellCurrent);
            
            return [hiddenStateCurrent, memoryCellCurrent];
        });
    }
    callTestWeird(inputs: tf.Tensor[]){
        return tf.tidy(()=>{
            
            const inputFeature: tf.Tensor = inputs[0]; const hiddenStatesChildren: tf.Tensor = inputs[1]; const memoryCellsChildren: tf.Tensor = inputs[2];
            // console.log(memoryCellsChildren.shape)
            // // Shapes: [ 4, 1 ] [ 2, 4, 1 ] [ 2, 4, 1 ]

            const hiddenStateTilde: tf.Tensor = tf.sum(hiddenStatesChildren, 0);
    
            const inputGate: tf.Tensor = this.activationLayer.apply(this.getGate(inputFeature, hiddenStateTilde, this.IWInput, this.IU, this.bU));
            // const forgetGates: tf.Tensor = this.activationLayer.apply(this.getGate(inputFeature, hiddenStatesChildren, this.FWInput, this.FU, this.bF));
            // const test = tf.reshape(this.FU, hiddenStatesChildren.shape);
            const inputFeatureForget: tf.Tensor = inputFeature.reshape([1, inputFeature.shape[0]]);
            const forgetGatesCollapsed: tf.Tensor = this.activationLayer.apply(this.getGateForget(inputFeatureForget, hiddenStatesChildren.squeeze(), this.FWInput, this.FU, this.bF));
            // const forgetGatesCollapsed: tf.Tensor = this.activationLayer.apply(tf.sum(tf.matMul(hiddenStatesChildren.squeeze(), this.FU),0));
            const outputGate: tf.Tensor = this.activationLayer.apply(this.getGate(inputFeature, hiddenStateTilde, this.OWInput, this.OU, this.bO));
            const uGate: tf.Tensor = tf.tanh(this.getGate(inputFeature, hiddenStateTilde, this.UWInput, this.UU, this.bU));
    
            const memoryCellCurrent: tf.Tensor = this.getMemoryCellTest(inputGate, uGate, forgetGatesCollapsed, memoryCellsChildren);
            const hiddenStateCurrent: tf.Tensor = this.getHiddenState(outputGate, memoryCellCurrent);
            console.log("Mem cell shape")
            console.log(memoryCellCurrent.shape);
            return [hiddenStateCurrent, memoryCellCurrent];
        });
    }
    /**
     * 
     * @param inputs Inputs a list of tensors as follows: [inputFeature, child hidden states, child memory cells]
     * 
     */
    
    // callTestGradientFix(inputs: tf.Tensor[]){
    //     if(inputs.length<3){
    //         throw new Error("Childsum tree LSTM requires atleast one input, hiddenState and memoryCell tensor as input")
    //     }
    //     if ((inputs.length-1)%2!=0){
    //         throw new Error("Unequal number of hiddenState and memoryCell inputs");
    //     }
    //     const middleIndex = ((inputs.length-1)/2);
    //     return tf.tidy(()=>{
    //         const inputFeature: tf.Tensor = inputs[0]; const hiddenStatesChildrenArray: tf.Tensor[]=inputs.slice(1, 1+middleIndex); 
    //         const memoryCellsChildrenArray: tf.Tensor[] = inputs.slice(middleIndex+1);
    //         const hiddenStatesChildren: tf.Tensor = tf.stack([hiddenStatesChildrenArray[0]]);
    //         const memoryCellsChildren: tf.Tensor = tf.stack([memoryCellsChildrenArray[0]]);
    
    //         const hiddenStateTilde: tf.Tensor = tf.sum(hiddenStatesChildren, 0);
    
    //         // Make more flexible with config!
    //         const inputGate: tf.Tensor = this.activationLayer.apply(this.getGate(inputFeature, hiddenStateTilde, this.IWInput, this.IU, this.bU));
    //         const forgetGates: tf.Tensor = this.activationLayer.apply(this.getGateForget(inputFeature, hiddenStatesChildrenArray, this.FWInput, this.FU, this.bF));
    //         const outputGate: tf.Tensor = this.activationLayer.apply(this.getGate(inputFeature, hiddenStateTilde, this.OWInput, this.OU, this.bO));
    //         const uGate: tf.Tensor = tf.tanh(this.getGate(inputFeature, hiddenStateTilde, this.UWInput, this.UU, this.bU));
    
    //         const memoryCellCurrent: tf.Tensor = this.getMemoryCell(inputGate, uGate, forgetGates, memoryCellsChildren);
    //         const hiddenStateCurrent: tf.Tensor = this.getHiddenState(outputGate, memoryCellCurrent);
            
    //         return [hiddenStateCurrent, memoryCellCurrent];
    //     });
    // }
    

    private getGate(inputNode: tf.Tensor, inputHiddenState: tf.Tensor, inputWeight: tf.Variable, hiddenStateWeight: tf.Variable, biasVec: tf.Variable){
        return tf.add(tf.add(tf.matMul(inputWeight, inputNode), tf.matMul(hiddenStateWeight, inputHiddenState)), biasVec);
    }


    private getGateForget(inputNode: tf.Tensor, inputHiddenStates: tf.Tensor, inputWeight: tf.Variable, hiddenStateWeight: tf.Variable, biasVec: tf.Variable){
        const intermediate: tf.Tensor = tf.add(tf.matMul(inputNode, inputWeight), tf.matMul(inputHiddenStates, hiddenStateWeight));
        return tf.add(intermediate.reshape([intermediate.shape[1]!,intermediate.shape[0]]), biasVec);

    }

    private getMemoryCell(inputGate: tf.Tensor, uGate: tf.Tensor, forgetGates: tf.Tensor, memoryCellsChildren: tf.Tensor){
        // console.log(forgetGates.shape);
        // console.log(memoryCellsChildren.shape);
        const memoryCellChildrenReshaped = memoryCellsChildren.reshape([memoryCellsChildren.shape[1]!,memoryCellsChildren.shape[0]]);
        console.log("Int result shapes")
        console.log(forgetGates.shape)
        console.log(memoryCellsChildren.shape);
        console.log(tf.sum(tf.mul(forgetGates, memoryCellsChildren), 0).shape);
        console.log(tf.mul(inputGate, uGate).shape);
        return tf.add(tf.mul(inputGate, uGate), tf.sum(tf.mul(forgetGates, memoryCellsChildren), 0));
    }

    private getMemoryCellTest(inputGate: tf.Tensor, uGate: tf.Tensor, forgetGates: tf.Tensor, memoryCellsChildren: tf.Tensor){
        // console.log(forgetGates.shape);
        // console.log(memoryCellsChildren.shape);
        const memoryCellChildrenReshaped = memoryCellsChildren.reshape([memoryCellsChildren.shape[1]!,memoryCellsChildren.shape[0]]);
        console.log(memoryCellChildrenReshaped.shape)
        console.log(tf.sum(tf.mul(forgetGates, memoryCellChildrenReshaped), 1, true).shape);
        console.log(tf.mul(inputGate, uGate).shape);
        return tf.add(tf.mul(inputGate, uGate), tf.sum(tf.mul(forgetGates, memoryCellChildrenReshaped), 1, true));
    }


    private getHiddenState(outputGate: tf.Tensor, currentMemoryCell: tf.Tensor){
        return tf.mul(outputGate, tf.tanh(currentMemoryCell));
    }

    public loadWeights(fileLocation: string){
        const weightValues: number[][][] = JSON.parse(fs.readFileSync(fileLocation, 'utf8'));
        for (let i = 0;i<weightValues.length;i++){
            this.allWeights[i].assign(tf.tensor(weightValues[i]));
        }
    }

    public saveWeights(fileLocation: string){
        const weightValues: number[][][] = this.allWeights.map(x=>{
            return x.arraySync();
        }) as number[][][];
        fs.writeFileSync(fileLocation, JSON.stringify(weightValues));
    }

}

export class ModelTreeLSTM{
    binaryTreeLSTMLayer: BinaryTreeLSTM[];
    childSumTreeLSTMLayer: ChildSumTreeLSTM[];
    qValueNetwork: QValueNetwork;

    configLocation: string;
    weightsDir: string;

    initialised: boolean;
    loadedConfig: IModelConfig;

    public constructor(inputSize: number){
        this.binaryTreeLSTMLayer = [];
        this.childSumTreeLSTMLayer = [];
        this.qValueNetwork = new QValueNetwork(inputSize,1);
        this.configLocation = path.join(__dirname, '../model/model-config/model-config.json');
        this.weightsDir = path.join(__dirname, '../model/weights');
        this.initialised = false;
    }

    public async initModel(){
        const modelConfig = await this.loadConfig();
        this.loadedConfig = modelConfig;

        const binaryLSTMConfig = modelConfig.binaryTreeLSTM;
        for(let i=0;i<binaryLSTMConfig.layers.length;i++){
            const newLayer: BinaryTreeLSTM = new BinaryTreeLSTM(binaryLSTMConfig.layers[i].numUnits!,binaryLSTMConfig.layers[i].activation!);
            if (binaryLSTMConfig.weightsConfig.loadWeights){
                newLayer.loadWeights(this.weightsDir+binaryLSTMConfig.weightsConfig.weightLocation);
            }
            this.binaryTreeLSTMLayer.push(newLayer);
        }
        const childSumConfig = modelConfig.childSumTreeLSTM;
        for (let j=0;j<childSumConfig.layers.length;j++){
            const newLayer: ChildSumTreeLSTM = new ChildSumTreeLSTM(childSumConfig.layers[j].numUnits!, childSumConfig.layers[j].activation!);
            if (childSumConfig.weightsConfig.loadWeights){
                newLayer.loadWeights(this.weightsDir+childSumConfig.weightsConfig.weightLocation);
            }
            this.childSumTreeLSTMLayer.push(newLayer);
        }
        await this.qValueNetwork.init(modelConfig.qValueNetwork, this.weightsDir);
        this.initialised=true;
    }

    public async saveModel(){
        // THIS CAN NOT SAVE MULTIPLE LAYERS
        for (const layer of this.binaryTreeLSTMLayer){
            layer.saveWeights(this.weightsDir+this.loadedConfig.binaryTreeLSTM.weightsConfig.weightLocation)
        }
        for (const layer of this.childSumTreeLSTMLayer){
            layer.saveWeights(this.weightsDir+this.loadedConfig.childSumTreeLSTM.weightsConfig.weightLocation)
        }
        // This can
        this.qValueNetwork.saveNetwork(this.loadedConfig.qValueNetwork, this.weightsDir);

    }

    public async loadConfig(){
        return await new Promise<IModelConfig>((resolve, reject)=>{
            fs.readFile(this.configLocation, 'utf8', async (err, data) =>{
                if (err) throw err;
                const parsedConfig: IModelConfig = JSON.parse(data);
                const errs: Error[]=this.validateConfig(parsedConfig);
                if (errs.length>0){
                    console.warn(errs);
                }
                resolve(parsedConfig);
            });
        });
    }

    public forwardPass(resultSetFeatures: IResultSetRepresentation, idx: number[]): IModelOutput{
        const outputTensors = tf.tidy(() =>{
            // Turn off for real use?
            if (resultSetFeatures.hiddenStates.length!=resultSetFeatures.memoryCell.length){
                throw new Error("Model given hiddenState and memoryCell arrays with different sizes");
            }
            if (Math.max(...idx) > resultSetFeatures.hiddenStates.length - 1|| Math.min(...idx) < 0){
                throw new Error("Model given join indexes out of range of array");
            }
            // Input to binary Tree LSTM
            const inputTensor: tf.Tensor = tf.zeros([this.binaryTreeLSTMLayer[0].numUnits, 1]);

            const binaryLSTMInput: tf.Tensor = tf.stack([inputTensor, resultSetFeatures.hiddenStates[idx[0]], resultSetFeatures.hiddenStates[idx[1]],
            resultSetFeatures.memoryCell[idx[0]], resultSetFeatures.memoryCell[idx[1]]]);

            const joinRepresentation: tf.Tensor[] = this.binaryTreeLSTMLayer[0].call(binaryLSTMInput);
            const joinHiddenState: tf.Tensor = joinRepresentation[0]; const joinMemoryCell = joinRepresentation[1];
            /* This part we take all features and pass this into childsum tree lstm */

            // Sort high to low such that we can remove without consideration of what removing elements from array does to indexing
            idx = idx.sort((a,b)=> b-a);
            for (const i of idx){
                // Dispose tensors before removing pointer to avoid "floating" tensors
                resultSetFeatures.hiddenStates[i].dispose(); resultSetFeatures.memoryCell[i].dispose();
                // Remove features associated with index
                resultSetFeatures.hiddenStates.splice(i,1); resultSetFeatures.memoryCell.splice(i,1);
            }
            // Add join features
            resultSetFeatures.hiddenStates.push(joinHiddenState); resultSetFeatures.memoryCell.push(joinMemoryCell);

            const childSumInput: tf.Tensor[] = [inputTensor, tf.stack(resultSetFeatures.hiddenStates), tf.stack(resultSetFeatures.memoryCell)];
            const childSumOutput: tf.Tensor[] = this.childSumTreeLSTMLayer[0].call(childSumInput);

            // TODO: Query representation

            // Feed representation into dense layer to get Q-value
            const qValue = this.qValueNetwork.forwardPassFromConfig(childSumOutput[0]);
            return [qValue, joinHiddenState, joinMemoryCell];
        });

        const output: IModelOutput = {
            qValue: outputTensors[0],
            hiddenState: outputTensors[1],
            memoryCell: outputTensors[2]
        };
        return output
    }

    /**
     * Function that allows the model to optimize the neural network. The other forward pass method can't since it reuses output from previous forward pass executions.
     * This should reduce forward pass execution time, but prevents the optimizer from calculating gradient
     * @param inputFeatures The leaf node feature representations
     * @param treeState The feature indexes of made joins
     */

    // TODO: TEST THIS!!!! Also fixed a bug possibly, in binaryLSTMInput = ... I used to do hiddenStates[0] and hiddenStates[1] instead of index,
    // that was most likely incorrect
    public forwardPassRecursive(inputFeatures: IResultSetRepresentation, treeState: number[][]){        
        const inputFeaturesCloned: IResultSetRepresentation =   {hiddenStates: inputFeatures.hiddenStates.map(x=>x.clone()),
        memoryCell: inputFeatures.memoryCell.map(x=>x.clone())};
        // Recursively execute joins of join state 
        const inputTensor: tf.Tensor = tf.zeros([this.binaryTreeLSTMLayer[0].numUnits,1]);

        for (const idx of treeState){
            const binaryLSTMInput: tf.Tensor = tf.stack([inputTensor, inputFeaturesCloned.hiddenStates[idx[0]], inputFeaturesCloned.hiddenStates[idx[1]],
                inputFeaturesCloned.memoryCell[idx[0]], inputFeaturesCloned.memoryCell[idx[1]]]);

            const joinRepresentation: tf.Tensor[] = this.binaryTreeLSTMLayer[0].call(binaryLSTMInput);
            const joinHiddenState: tf.Tensor = joinRepresentation[0]; const joinMemoryCell: tf.Tensor = joinRepresentation[1];
            /* This part we take all features and pass this into childsum tree lstm */
            // Sort high to low such that we can remove without consideration of what removing elements from array does to indexing
            const removeIdx = idx.sort((a,b)=> b-a);
            for (const i of removeIdx){
                // Dispose tensors before removing pointer to avoid "floating" tensors
                inputFeaturesCloned.hiddenStates[i].dispose(); inputFeaturesCloned.memoryCell[i].dispose();
                // Remove features associated with index
                inputFeaturesCloned.hiddenStates.splice(i,1); inputFeaturesCloned.memoryCell.splice(i,1);
            }
            // Add join features
            inputFeaturesCloned.hiddenStates.push(joinHiddenState); inputFeaturesCloned.memoryCell.push(joinMemoryCell);

        }

        const childSumInput: tf.Tensor[] = [inputTensor, tf.stack(inputFeaturesCloned.hiddenStates), tf.stack(inputFeaturesCloned.memoryCell)];
        const childSumOutput: tf.Tensor[] = this.childSumTreeLSTMLayer[0].call(childSumInput);
        // TODO: Query representation

        // Feed representation into dense layer to get Q-value
        const qValue = this.qValueNetwork.forwardPassFromConfig(childSumOutput[0]);
        return qValue
    
    }

    public validateConfig(modelConfig: IModelConfig){
        const errorsValidateBinary: Error[] = this.validateLayerConfig(modelConfig.binaryTreeLSTM);
        const errorsValidateChildSum: Error[] = this.validateLayerConfig(modelConfig.childSumTreeLSTM);
        const errorsValidateDense: Error[] = this.validateLayerConfig(modelConfig.qValueNetwork);

        const allErrors = [...errorsValidateBinary, ...errorsValidateChildSum, ...errorsValidateDense];
        return allErrors;
        // throw errorValidate;
    }
    public validateLayerConfig(layerConfig: ILayerConfig){
        const errs: Error[] = []
        if (!layerOptionsModel.includes(layerConfig.type)){
            errs.push(new Error(`Invalid layer config type, got: ${layerConfig.type}, expected: ${layerOptionsModel}`))
        }
        if (layerConfig.type=='dense'){
            const layerConfigErr=this.validateLayerConfigDense(layerConfig.layers)
            if (layerConfigErr){
                errs.push(...layerConfigErr);
            }
        }
        if (layerConfig.type=='lstm'){
            const layerConfigErr=this.validateLayerConfigLSTM(layerConfig.layers)
            if (layerConfigErr){
                errs.push(...layerConfigErr);
            }
        }
        if(!this.validateWeightConfig(layerConfig.weightsConfig)){
            errs.push(new Error("Passed invalid weight config"));
        };
        return errs
    }

    public validateWeightConfig(weightConfig: IWeightsConfig){
        if (weightConfig.loadWeights && !weightConfig.weightLocation){
            return false;
        }
        return weightConfig.loadWeights!=undefined && typeof(weightConfig.loadWeights)=='boolean' && typeof (weightConfig.weightLocation)=='string';
    }
    public validateLayerConfigLSTM(layerConfigArray: ILayerDetailsConfig[]){
        const errs: Error[] = [];

        for (const layerConfigLSTM of layerConfigArray){
            if (layerConfigLSTM.type != 'childSumTreeLSTM' && layerConfigLSTM.type != 'binaryTreeLSTM'){
                errs.push(new Error(`Got invalid type parameter for LSTM, got ${layerConfigLSTM.type}, expected childSumTreeLSTM, binaryTreeLSTM`));
            }
            if (!layerConfigLSTM.numUnits || typeof(layerConfigLSTM.numUnits) != 'number'){
                errs.push(new Error(`Got invalid numUnits parameter for LSTM, got ${layerConfigLSTM.numUnits} expected 'number'`));
            }
            if (!layerConfigLSTM.activation || !activationOptions.includes(layerConfigLSTM.activation)){
                errs.push(new Error(`Got invalid activation parameter, got ${layerConfigLSTM.activation}, expected ${activationOptions}`));
            }
        }
        return errs
    }
    public validateLayerConfigDense(layerConfigArray: ILayerDetailsConfig[]){
        const errs: Error[] = [];
        for (const layerConfigDense of layerConfigArray){
            if (layerConfigDense.type != 'dense' && layerConfigDense.type != 'activation'){
                errs.push(new Error(`Got invalid type parameter for Dense, got ${layerConfigDense.type}, expected 'dense', 'activation'`));
            }
            if(!layerConfigDense.inputSize||typeof(layerConfigDense.inputSize) != 'number'||
            !layerConfigDense.outputSize||typeof(layerConfigDense.outputSize) != 'number'){
                errs.push(new Error(`Got invalid input or output size parameters for Dense, got ${layerConfigDense.inputSize}, ${layerConfigDense.outputSize}, expected
                numbers`));
            }
            if (layerConfigDense.type=='activation'&& (!layerConfigDense.activation || !activationOptions.includes(layerConfigDense.activation))){
                errs.push(new Error(`Got invalid activation parameter, got ${layerConfigDense.activation}, expected ${activationOptions}`));
            }
        }
        return errs;
    }
}

export interface IModelConfig{
    "binaryTreeLSTM": ILayerConfig
    "childSumTreeLSTM": ILayerConfig
    "qValueNetwork": ILayerConfig
}

export interface ILayerConfig{
    "weightsConfig": IWeightsConfig,
    "type": layerOptionsModelId,
    "layers": ILayerDetailsConfig[]
}

export interface IWeightsConfig{
    "loadWeights": boolean;
    "weightLocation": string;
    "weightLocationBias"?: string;
}

export interface ILayerDetailsConfig{
    "type": treeLstmLayerId
    "activation"?: activationOptionsId
    "numUnits"?: number
    "inputSize"?: number
    "outputSize"?: number
}

export interface IModelOutput{
    qValue: tf.Tensor;
    hiddenState: tf.Tensor;
    memoryCell: tf.Tensor;
}

export function stringLiteralArray<T extends string>(a: T[]) {
    return a;
}

export const layerOptionsTreeLstm = stringLiteralArray(['childSumTreeLSTM', 'binaryTreeLSTM', 'dense', 'activation']);
export const layerOptionsDense = stringLiteralArray(['dense', 'activation']);
export const activationOptions = stringLiteralArray(['elu', 'hardSigmoid', 'linear', 'relu', 'relu6', 
'selu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'swish', 'mish']);
export const layerOptionsModel = stringLiteralArray(['lstm', 'dense']);
type treeLstmLayerId = typeof layerOptionsTreeLstm[number];
type activationOptionsId = typeof activationOptions[number];
type denseOptionsId = typeof layerOptionsDense[number];
type layerOptionsModelId = typeof layerOptionsModel[number];

// const testTreeLSTM = new BinaryTreeLSTM(3);

// const testIT: tf.Tensor = tf.tensor([0,0,0]).reshape([3,1]);
// const testHLT: tf.Tensor = tf.tensor([1,2,3]).reshape([3,1]);
// const testHRT: tf.Tensor = tf.tensor([1,1,1]).reshape([3,1]);
// const testMLT: tf.Tensor = tf.tensor([1,2,3]).reshape([3,1]);
// const testMRT: tf.Tensor = tf.tensor([1,2,3]).reshape([3,1]);
// const output: tf.Tensor[] = testTreeLSTM.call(tf.stack([testIT, testHLT, testHRT, testMLT, testMRT]));
// tf.stack(output).print();

// const testChildSum = new ChildSumTreeLSTM(3);
// const testInput: tf.Tensor = tf.tensor([0,0,0]).reshape([3,1]);

// const testH1 = tf.tensor2d([[1],[2],[1]]);
// const testH2 = tf.tensor2d([[1],[1],[1]]);
// const testH3 = tf.tensor2d([[0],[0],[0]]);
// const testH = tf.stack([testH1, testH2, testH3]);

// const testC1 = tf.tensor2d([[3],[1],[3]]);
// const testC2 = tf.tensor2d([[2],[2],[2]]);
// const testC3 = tf.tensor2d([[0],[0],[1]]);
// const testC = tf.stack([testC1, testC2, testC3]);
// const testResult = testChildSum.call([testInput, testH, testC]);
// testResult[0].print();
// testResult[1].print();