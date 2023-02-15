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

    public async initModel(){
        if (!this.modelTreeLSTM.initialised){
            await this.modelTreeLSTM.initModel();
        }
        this.init=true;
    };

    public getModel(): ModelTreeLSTM{
        return this.modelTreeLSTM
    };

    public saveModel(){
        this.modelTreeLSTM.saveModel();
    };

    public saveRunningMoments(runningMoments: IRunningMoments, runningMomentsPath: string){
        path.join(__dirname, runningMomentsPath);
        let serialiseObject: {[key: string|number]: any} = {};
        serialiseObject['indexes'] = runningMoments.indexes;
        for (const [key, value] of runningMoments.runningStats){
          serialiseObject[key] = value;
        }
        fs.writeFile(path.join(__dirname, runningMomentsPath), JSON.stringify(serialiseObject), function(err: any) {
            if(err) {
                return console.log(err);
            }
        }); 
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