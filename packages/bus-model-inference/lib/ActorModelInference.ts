import * as tf from '@tensorflow/tfjs-node';
import { Actor, IAction, IActorArgs, IActorOutput, IActorTest, Mediate } from '@comunica/core';

/**
 * A comunica actor for model-inference events.
 *
 * Actor types:
 * * Input:  IActionModelInference:      TODO: fill in.
 * * Test:   <none>
 * * Output: IActorModelInferenceOutput: TODO: fill in.
 *
 * @see IActionModelInference
 * @see IActorModelInferenceOutput
 */
export abstract class ActorModelInference extends Actor<IActionModelInference, IActorModelInferenceTest, IActorModelInferenceOutput> {
  /**
  * If this actor will be logged in the debugger and physical query plan logger
  */
  public includeInLogs = true;
  public readonly task: ITask;
  public readonly physicalName: string;
    
  /**
  * @param args - @defaultNested {<default_bus> a <cc:components/Bus.jsonld#Bus>} bus
  */
  public constructor(args: IActorModelInferenceArgs, options: IActorModelInferenceInternalOptions) {
    super(args);
    this.task = options.task;
    this.physicalName = options.physicalName;
  }

  public async test(action: IActionModelInference): Promise<IActorModelInferenceTest> {
    if (action.task !== this.task) {
      throw new Error(`${this.name} can only handle model inference of type '${this.task}', while '${action.task}' was given.`);
    }
    // Possibly validate some shapes? Or make that model specific
    // For now we don't do expected performance, but could make that part of it
    return {expectedPerformance: 10};
  }

  public async executePlan(){
    throw new Error(`${this.name} plan execution not implemented.`);
  }

  public abstract loadWeights(fileLocation: string): void;
}

export interface IActorModelInferenceInternalOptions {
  /**
   * The prediction task of the model this actor implements
   */
  task: ITask;
  /**
   * The physical name of the model this actor implements.
   * This is used for debug and query plan logs.
   */
  physicalName: string;
  /**
   * Model weights file location
   */
  weightsFileLocation: string
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
   * Flag that indicates if the model should be reloaded from disk. Because the model in training is a different 
   * instance, if any optimization steps are made, we have to reload the model to obtain them.
   */
  reloadModelFromDisk: boolean;

  /**
   * Where the model weights should be loaded from
   */
  modelWeightsFileLocation: string;

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

export interface IActorModelInferenceTest extends IActorTest{
  /**
   * How well the model expects to perform
   */

  expectedPerformance: number;
}

export type ITask = 'qValuePrediction' | 'GraphEmbedding';


export type IActorModelInferenceArgs = IActorArgs<
IActionModelInference, IActorModelInferenceTest, IActorModelInferenceOutput>;

export type MediatorModelInference = Mediate<
IActionModelInference, IActorModelInferenceOutput>;


