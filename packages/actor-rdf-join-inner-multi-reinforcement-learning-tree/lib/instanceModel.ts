import { IAggregateValues, IRunningMoments } from "@comunica/mediator-join-reinforcement-learning";
import { ModelTreeLSTM } from "./treeLSTM";
import * as fs from "fs";
import * as path from "path";

export class InstanceModel{
    private modelTreeLSTM: ModelTreeLSTM;
    init: boolean;
    initMoments: boolean;
    public constructor(){
        this.modelTreeLSTM = new ModelTreeLSTM(0);
        this.init = false;
        this.initMoments = false;
    };

    public async initModel(weightsDir?: string){
        if (!this.modelTreeLSTM.initialised){
            await this.modelTreeLSTM.initModel(weightsDir);
            this.init=true;
            this.modelTreeLSTM.initialised=true;
        }
    };
    public async initModelRandom(){
        if (!this.modelTreeLSTM.initialised){
            await this.modelTreeLSTM.initModelRandom();
            this.modelTreeLSTM.initialised=true;
            this.init = true    
        }
    }

    /**
     * Method to flush any loaded model weights from the model
     * This is used when during model instantiation something goes wrong and we want to start `clean`
     */
    public flushModel(){
        this.modelTreeLSTM.flushModel();
    }

    public getModel(): ModelTreeLSTM{
        return this.modelTreeLSTM
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

    // public loadRunningMoments(){

    // }

    // public saveRunningMoments(){

    // }
}