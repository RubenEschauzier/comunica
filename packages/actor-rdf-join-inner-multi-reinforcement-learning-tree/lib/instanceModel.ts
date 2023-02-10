import { ModelTreeLSTM } from "./treeLSTM";

export class InstanceModel{
    private modelTreeLSTM: ModelTreeLSTM;
    init: boolean;
    public constructor(){
        this.modelTreeLSTM = new ModelTreeLSTM(0);
        this.init = false
    }
    public async initModel(){
        if (!this.modelTreeLSTM.initialised){
            await this.modelTreeLSTM.initModel();
        }
        this.init=true;
    }

    public getModel(): ModelTreeLSTM{
        return this.modelTreeLSTM
    }
}