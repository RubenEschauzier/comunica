import { ModelTreeLSTM } from './treeLSTM';
import { ActorModelInference, IActionModelInference, IActorModelInferenceOutput, IActorModelInferenceArgs, IActorModelInferenceTest, IActorModelInferenceInternalOptions } from '@comunica/bus-model-inference';
import { IActorArgs, IActorTest } from '@comunica/core';

/**
 * A comunica Tree Lstm Q Value Model Inference Actor.
 */
export class ActorModelInferenceTreeLstmQValue extends ActorModelInference {
  model: ModelTreeLSTM
  public constructor(args: IActorModelInferenceArgs, options: IActorModelInferenceInternalOptions) {
    super(args, options);
    this.model = new ModelTreeLSTM('../config-model')
    this.loadWeights(options.weightsFileLocation);
  }

  public async test(action: IActionModelInference): Promise<IActorModelInferenceTest> {
    return {expectedPerformance: 0}; // TODO implement
  }

  public async run(action: IActionModelInference): Promise<IActorModelInferenceOutput> {
    throw new Error("TODO implement")
  }

  public async loadWeights(fileLocation: string): Promise<void> {
    try{
      await this.model.initModel();
    }
    catch(err){
      // Flush any initialisation that went through to start from fresh model
      this.model.flushModel()
      // Initialise random model
      await this.model.initModelRandom();
      console.warn("INFO: Failed to load model weights, randomly initialising model weights");
    }
  }
}
