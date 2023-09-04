import { IAggregateValues, IRunningMoments } from "@comunica/mediator-join-reinforcement-learning";
import { ModelTreeLSTM } from "./treeLSTM";
import * as fs from "fs";
import { GraphConvolutionModel } from "@comunica/mediator-join-reinforcement-learning/lib/GraphConvolution";
import * as path from "path";

export class InstanceModelLSTM{
    private modelTreeLSTM: ModelTreeLSTM;
    initMoments: boolean;
    public constructor(){
        this.modelTreeLSTM = new ModelTreeLSTM(0);
        this.initMoments = false;
    };

    public async initModel(weightsDir?: string){
        if (!this.modelTreeLSTM.initialised){
            await this.modelTreeLSTM.initModel(weightsDir);
            this.modelTreeLSTM.initialised=true;
        }
    };

    public async initModelRandom(){
        if (!this.modelTreeLSTM.initialised){
            await this.modelTreeLSTM.initModelRandom();
            this.modelTreeLSTM.initialised=true;
        }
    }

    /**
     * Method to flush any loaded model weights from the model
     * This is used when during model instantiation something goes wrong and we want to start `clean`
     */
    public flushModel(){
        this.modelTreeLSTM.flushModel();
        this.modelTreeLSTM.initialised=false;
    }

    public getModel(): ModelTreeLSTM{
        return this.modelTreeLSTM;
    };

    public saveModel(weightsDir?: string){
        this.modelTreeLSTM.saveModel(weightsDir);
    };

    public saveRunningMoments(runningMoments: IRunningMoments, runningMomentsPath: string){
        let serialiseObject: {[key: string|number]: any} = {};
        serialiseObject['indexes'] = runningMoments.indexes;
        for (const [key, value] of runningMoments.runningStats){
          serialiseObject[key] = value;
        }
        fs.writeFileSync(runningMomentsPath, JSON.stringify(serialiseObject));
    };
    
    public loadRunningMoments(runningMomentsPath: string){
        const data = JSON.parse(fs.readFileSync(runningMomentsPath, 'utf8'));
        const indexes = data.indexes
        const runningIndexStats: Map<number, IAggregateValues> = new Map();

        for (const index of indexes){
            const strIndex = index.toString()
            runningIndexStats.set(index, data[strIndex]);
        }

        const loadedRunningMoments: IRunningMoments = {indexes: indexes, runningStats: runningIndexStats}
        this.initMoments = true;
        return loadedRunningMoments;
    };
}


export class InstanceModelGCN{
    private modelSubjSubj: GraphConvolutionModel;
    private modelObjObj: GraphConvolutionModel;
    private modelObjSubj: GraphConvolutionModel;

    initMoments: boolean;
    public constructor(){
        this.modelSubjSubj = new GraphConvolutionModel(path.join(__dirname, '../model/gcn-model-subj-subj'));
        this.modelObjObj = new GraphConvolutionModel(path.join(__dirname, '../model/gcn-model-obj-obj'));
        this.modelObjSubj = new GraphConvolutionModel(path.join(__dirname, '../model/gcn-model-obj-subj'));

        this.initMoments = false;
    };

    public initModel(){
        // Try to load weights if available, else initialise random and give warning
        this.modelSubjSubj.initModelWeights();
        this.modelObjObj.initModelWeights();
        this.modelObjSubj.initModelWeights();
    };

    /**
     * Method to flush any loaded model weights from the model
     * This is used when during model instantiation something goes wrong and we want to start `clean`
     */

    public getModels(): IQueryGraphEncodingModels{
        return {
            modelSubjSubj: this.modelSubjSubj,
            modelObjObj: this.modelObjObj,
            modelObjSubj: this.modelObjSubj
        }
    };

    public saveModel(){
        this.modelSubjSubj.saveModel();
        this.modelObjObj.saveModel();
        this.modelObjSubj.saveModel();
    };
}

export interface IQueryGraphEncodingModels{
    modelSubjSubj: GraphConvolutionModel;
    modelObjObj: GraphConvolutionModel;
    modelObjSubj: GraphConvolutionModel;
}