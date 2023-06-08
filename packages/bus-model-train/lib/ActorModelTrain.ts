import * as tf from '@tensorflow/tfjs-node';
import { Actor, IAction, IActorArgs, IActorOutput, IActorTest, Mediate } from '@comunica/core';
import { MediatorModelInference } from '@comunica/mediator-model-inference';
import { ActorModelInference } from '@comunica/bus-model-inference';

/**
 * A comunica actor for model-train events.
 *
 * Actor types:
 * * Input:  IActionModelTrain:      TODO: fill in.
 * * Test:   <none>
 * * Output: IActorModelTrainOutput: TODO: fill in.
 *
 * @see IActionModelTrain
 * @see IActorModelTrainOutput
 */
export abstract class ActorModelTrain extends Actor<IActionModelTrain, IActorTest, IActorModelTrainOutput> {
  /**
  * @param args - @defaultNested {<default_bus> a <cc:components/Bus.jsonld#Bus>} bus
  */

  public readonly mediatorModelInference: MediatorModelInference<ActorModelInference, IActionModelInference, IActorTest, IActorModelInferenceOutput>;

  public includeInLogs = true;
  // public mediatorInferenceStep: MediatorRace;
  public readonly task: ITask;
  public readonly physicalName: string;
  public readonly modelWeightsFileLocation: string;
  public readonly modelConfigFileLocation: string;

  public constructor(args: IActorModelTrainArgs, options: IActorModelTrainInternalOptions) {
    super(args);
    this.task = options.task;
    this.physicalName = options.physicalName;
    this.modelWeightsFileLocation = options.modelWeightsFileLocation;
    this.modelConfigFileLocation = options.modelConfigLocation;
  }
  
  public executeOptimizationStep(): number {
    throw new Error(`${this.name} optimization step not implemented.`);
  }

  public loadConfigFromDisk(): void{
    throw new Error(`${this.name} config loading not implemented.`);
  }

  public saveWeightsToDisk(): void{
    throw new Error(`${this.name} weights saving not implemented.`);
  }

  public loadWeightsFromDisk(): void{
    throw new Error(`${this.name} weights loading not implemented.`);
  }
}

export interface IActionModelTrain extends IAction {

}

export interface IActorModelTrainOutput extends IActorOutput {

}
export interface IActorModelTrainInternalOptions {
  /**
   * The training task this actor implements
   */
  task: ITask;
  /**
   * The physical name of the model this actor implements.
   * This is used for debug and query plan logs.
   */
  physicalName: string;
  /**
   * Where the model configuration file is located
   */
  modelConfigLocation: string;

  /**
   * Where the model weights will be saved to file
   */
  modelWeightsFileLocation: string;
}


export interface IActionModelInference extends IAction {
  /**
   * The logical join type.
   */
  task: ITask;

  /**
   * Numerical input.
   */
  input: tf.Tensor;

  /**
   * Misc arguments needed for model forward pass, think of adjacency matrix for graph networks, or any hyperparameters
   */
  args: Record<string, any>;
}

export interface IActorModelInferenceOutput extends IActorOutput {
  /**
   * Tensor output of model execution
   */
  outputTensor: tf.Tensor;

  /**
   * Attached metadata about model execution
   */
  context?: Record<string, any>;
}

export type ITask = 'qValuePrediction' | 'GraphEmbedding';

export type IActorModelTrainArgs = IActorArgs<
IActionModelTrain, IActorTest, IActorModelTrainOutput>;

export type MediatorModelTrain = Mediate<
IActionModelTrain, IActorModelTrainOutput>;
