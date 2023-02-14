import type * as RDF from '@rdfjs/types';
import type { IDataDestination } from './IDataDestination';
import type { IDataSource } from './IDataSource';
import type { IProxyHandler } from './IProxyHandler';
import type { SourceType } from './IQueryEngine';
import type { QueryExplainMode } from './IQueryOperationResult';
import type { Logger } from './Logger';

import * as tf from '@tensorflow/tfjs-node';
import { IResultSetRepresentation, IRunningMoments } from '@comunica/mediator-join-reinforcement-learning';
import { InstanceModel } from '@comunica/actor-rdf-join-inner-multi-reinforcement-learning-tree';

/**
 * Query context when a string-based query was passed.
 */
export type QueryStringContext = RDF.QueryStringContext & RDF.QuerySourceContext<SourceType> & IQueryContextCommon;
/**
 * Query context when an algebra-based query was passed.
 */
export type QueryAlgebraContext = RDF.QueryAlgebraContext & RDF.QuerySourceContext<SourceType> & IQueryContextCommon;

export type FunctionArgumentsCache = Record<string, { func?: any; cache?: FunctionArgumentsCache }>;

export interface ITrainEpisode {
  joinsMade: number[][];
  estimatedQValues: tf.Tensor[];
  featureTensor: IResultSetRepresentation;
  isEmpty: boolean;
};

export interface IBatchedTrainingExamples{
  trainingExamples: Map<string, ITrainingExample>;
  leafFeatures: IResultSetRepresentation;
}

export interface ITrainingExample{
  qValue: tf.Tensor;
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
