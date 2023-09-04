import type * as RDF from '@rdfjs/types';
import type { IDataDestination } from './IDataDestination';
import type { IDataSource } from './IDataSource';
import type { IProxyHandler } from './IProxyHandler';
import type { SourceType } from './IQueryEngine';
import type { QueryExplainMode } from './IQueryOperationResult';
import type { Logger } from './Logger';

import * as tf from '@tensorflow/tfjs-node';
import { IQueryGraphViews, IResultSetRepresentation, IRunningMoments } from '@comunica/mediator-join-reinforcement-learning';
import { InstanceModel } from '@comunica/actor-rdf-join-inner-multi-reinforcement-learning-tree';
import { number } from 'yargs';

/**
 * Query context when a string-based query was passed.
 */
export type QueryStringContext = RDF.QueryStringContext & RDF.QuerySourceContext<SourceType> & IQueryContextCommon;
/**
 * Query context when an algebra-based query was passed.
 */
export type QueryAlgebraContext = RDF.QueryAlgebraContext & RDF.QuerySourceContext<SourceType> & IQueryContextCommon;

export type FunctionArgumentsCache = Record<string, { func?: any; cache?: FunctionArgumentsCache }>;

/**
 * Interface for a single query training step, this interface keeps track of:
 * 1. {@link ITrainEpisode.joinsMade} The order joins made during query optimization, so during backprop we can retrace
 * these steps to get gradients. The order is decided by {@link ActorRdfJoinInnerMultiReinforcementLearningTree} and added 
 * to the episode in {@link MediatorJoinReinforcementLearning.mediateWith}. 
 * 
 * 2. {@link ITrainEpisode.learnedFeatureTensor} The feature tensors used for prediction. These are initialised in 
 * {@link MediatorJoinReinforcementLearning.initialiseFeatures} when the {@link ITrainEpisode.isEmpty} flag is set to true.
 * During join optimization this is updated to reflect the added joins. Updates happen in 
 * {@link MediatorJoinReinforcementLearning.mediateWith}
 * 
 * 3. {@link ITrainEpisode.leafFeatureTensor} The starting features that represent the information in the triple pattern. 
 * Initialised in {@link MediatorJoinReinforcementLearning.initialiseFeatures} and used to recreate the optimization 
 * steps for backprop
 * 
 * 4. {@link ITrainEpisode.sharedVariables} Matrix that shows what triple patterns share variables, initialised in 
 * {@link MediatorJoinReinforcementLearning.initialiseFeatures} and updated in 
 * {@link ActorRdfJoinInnerMultiReinforcementLearningTree} when a join is executed. This is to prevent cartesian
 * products
 * 
 * 5. {@link ITrainEpisode.graphViews} Three views of query graph to use for optimization step recreation during
 * backprop. Initialised in {@link MediatorJoinReinforcementLearning.initialiseFeatures}
 * 
 * 6. {@link ITrainEpisode.isEmpty} If the train episode is empty, set to false after execution of
 * {@link MediatorJoinReinforcementLearning.initialiseFeatures} and then set to true when it is emptied after 
 * query execution.
 * 
 * The episode is emptied if various conditions are satisfied at the end of a query execution.
 */
export interface ITrainEpisode {
  joinsMade: number[][];
  learnedFeatureTensor: IResultSetRepresentation;
  leafFeatureTensor: tf.Tensor[];
  sharedVariables: number[][];
  graphViews: IQueryGraphViews
  isEmpty: boolean;
};

/**
 * TODO: Why is this needed with experiencebuffers?
 */
export interface IBatchedTrainingExamples{
  trainingExamples: Map<string, ITrainingExample>;
  leafFeatures: IResultSetRepresentation;
}

export interface ITrainingExample{
  qValue: number;
  actualExecutionTime: number;
  N: number;
}


/**
 * Common query context interface
 */
export interface IQueryContextCommon {
  // Types of these entries should be aligned with contextKeyShortcuts in ActorInitQueryBase
  // and Keys in @comunica/context-entries
  trainEpisode?: ITrainEpisode;
  batchedTrainingExamples?: IBatchedTrainingExamples;
  modelInstance?: InstanceModel;
  runningMomentsExecutionTime?: IRunningMoments;
  runningMomentsFeaturesFile?: string;
  train?: boolean;
  queryKey?: string;
  source?: IDataSource;
  // Inherited from RDF.QueryStringContext: sources
  destination?: IDataDestination;
  initialBindings?: RDF.Bindings;
  // Inherited from RDF.QueryStringContext: queryFormat?: string;
  // Inherited from RDF.QueryStringContext: baseIRI?: string;
  log?: Logger;
  datetime?: Date;
  // Inherited from RDF.QueryStringContext: queryTimestamp?: Date;
  httpProxyHandler?: IProxyHandler;
  lenient?: boolean;
  httpIncludeCredentials?: boolean;
  httpAuth?: string;
  httpTimeout?: number;
  httpBodyTimeout?: boolean;
  httpRetryCount?: number;
  httpRetryDelay?: number;
  httpRetryOnServerError?: boolean;
  fetch?: typeof fetch;
  readOnly?: boolean;
  extensionFunctionCreator?: (functionNamedNode: RDF.NamedNode)
  => ((args: RDF.Term[]) => Promise<RDF.Term>) | undefined;
  functionArgumentsCache?: FunctionArgumentsCache;
  extensionFunctions?: Record<string, (args: RDF.Term[]) => Promise<RDF.Term>>;
  explain?: QueryExplainMode;
  recoverBrokenLinks?: boolean;
  localizeBlankNodes?: boolean;
}
