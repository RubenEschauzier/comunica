import * as fs from 'fs';
import * as path from 'path';
import type { IAggregateValues, IRunningMoments } from '@comunica/mediator-join-reinforcement-learning';
import { GraphConvolutionModel } from '@comunica/mediator-join-reinforcement-learning/lib/GraphConvolution';
import { ModelTreeLSTM } from './treeLSTM';

export class InstanceModelLSTM {
  private modelTreeLSTM: ModelTreeLSTM;
  initMoments: boolean;
  public constructor(modelDirectory: string) {
    this.modelTreeLSTM = new ModelTreeLSTM(0, modelDirectory);
    this.initMoments = false;
  }

  public async initModel(modelDir: string) {
    if (!this.modelTreeLSTM.initialised) {
      await this.modelTreeLSTM.initModel(modelDir);
      this.modelTreeLSTM.initialised = true;
    }
  }

  public async initModelRandom(modelDir: string) {
    if (!this.modelTreeLSTM.initialised) {
      await this.modelTreeLSTM.initModelRandom(modelDir);
      this.modelTreeLSTM.initialised = true;
    }
  }

  /**
     * Method to flush any loaded model weights from the model
     * This is used when during model instantiation something goes wrong and we want to start `clean`
     */
  public flushModel() {
    this.modelTreeLSTM.flushModel();
    this.modelTreeLSTM.initialised = false;
  }

  public getModel(): ModelTreeLSTM {
    return this.modelTreeLSTM;
  }

  public saveModel(weightsDir: string) {
    this.modelTreeLSTM.saveModel(weightsDir);
  }

  public saveRunningMoments(runningMoments: IRunningMoments, runningMomentsPath: string) {
    const serialiseObject: Record<string | number, any> = {};
    serialiseObject.indexes = runningMoments.indexes;
    for (const [ key, value ] of runningMoments.runningStats) {
      serialiseObject[key] = value;
    }
    fs.writeFileSync(runningMomentsPath, JSON.stringify(serialiseObject));
  }

  public loadRunningMoments(runningMomentsPath: string) {
    const data = JSON.parse(fs.readFileSync(runningMomentsPath, 'utf8'));
    const indexes = data.indexes;
    const runningIndexStats: Map<number, IAggregateValues> = new Map();

    for (const index of indexes) {
      const strIndex = index.toString();
      runningIndexStats.set(index, data[strIndex]);
    }

    const loadedRunningMoments: IRunningMoments = { indexes, runningStats: runningIndexStats };
    this.initMoments = true;
    return loadedRunningMoments;
  }
}

export class InstanceModelGCN {
  private readonly modelSubjSubj: GraphConvolutionModel;
  private readonly modelObjObj: GraphConvolutionModel;
  private readonly modelObjSubj: GraphConvolutionModel;

  initMoments: boolean;
  /**
     * Model holder object for the three GCN models that create query graph representations.
     * @param baseModelDir The absolute path of the directory where the three models should be saved to. Does not require trailing '/'
     */
  public constructor(baseModelDir: string) {
    this.modelSubjSubj = new GraphConvolutionModel(path.join(baseModelDir, 'gcn-model-subj-subj'));
    this.modelObjObj = new GraphConvolutionModel(path.join(baseModelDir, 'gcn-model-obj-obj'));
    this.modelObjSubj = new GraphConvolutionModel(path.join(baseModelDir, 'gcn-model-obj-subj'));

    this.initMoments = false;
  }

  public initModel(modelDir: string) {
    // Try to load weights if available, else initialise random and give warning
    this.modelSubjSubj.initModelWeights(path.join(modelDir, 'gcn-model-subj-subj'));
    this.modelObjObj.initModelWeights(path.join(modelDir, 'gcn-model-obj-obj'));
    this.modelObjSubj.initModelWeights(path.join(modelDir, 'gcn-model-obj-subj'));
  }

  public getModels(): IQueryGraphEncodingModels {
    return {
      modelSubjSubj: this.modelSubjSubj,
      modelObjObj: this.modelObjObj,
      modelObjSubj: this.modelObjSubj,
    };
  }

  public saveModels(modelDir: string) {
    this.modelSubjSubj.saveModel(path.join(modelDir, 'gcn-model-subj-subj'));
    this.modelObjObj.saveModel(path.join(modelDir, 'gcn-model-obj-obj'));
    this.modelObjSubj.saveModel(path.join(modelDir, 'gcn-model-obj-subj'));
  }

  public disposeModels(){
    this.modelSubjSubj.flushModel();
    this.modelObjObj.flushModel();
    this.modelObjSubj.flushModel();
  }
}

export interface IQueryGraphEncodingModels{
  modelSubjSubj: GraphConvolutionModel;
  modelObjObj: GraphConvolutionModel;
  modelObjSubj: GraphConvolutionModel;
}
