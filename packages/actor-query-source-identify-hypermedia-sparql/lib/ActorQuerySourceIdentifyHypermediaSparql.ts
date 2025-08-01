import type { MediatorHttp } from '@comunica/bus-http';
import type { MediatorMergeBindingsContext } from '@comunica/bus-merge-bindings-context';
import type {
  IActionQuerySourceIdentifyHypermedia,
  IActorQuerySourceIdentifyHypermediaOutput,
  IActorQuerySourceIdentifyHypermediaArgs,
  IActorQuerySourceIdentifyHypermediaTest,
} from '@comunica/bus-query-source-identify-hypermedia';
import {
  ActorQuerySourceIdentifyHypermedia,
} from '@comunica/bus-query-source-identify-hypermedia';
import { KeysInitQuery, KeysQueryOperation } from '@comunica/context-entries';
import type { TestResult } from '@comunica/core';
import { failTest, passTest } from '@comunica/core';
import type { ComunicaDataFactory } from '@comunica/types';
import { BindingsFactory } from '@comunica/utils-bindings-factory';
import { Factory } from 'sparqlalgebrajs';
import { QuerySourceSparql } from './QuerySourceSparql';

/**
 * A comunica SPARQL Query Source Identify Hypermedia Actor.
 */
export class ActorQuerySourceIdentifyHypermediaSparql extends ActorQuerySourceIdentifyHypermedia {
  public readonly mediatorHttp: MediatorHttp;
  public readonly mediatorMergeBindingsContext: MediatorMergeBindingsContext;
  public readonly checkUrlSuffix: boolean;
  public readonly forceHttpGet: boolean;
  public readonly cacheSize: number;
  public readonly forceSourceType: boolean;
  public readonly bindMethod: BindMethod;
  public readonly countTimeout: number;
  public readonly cardinalityCountQueries: boolean;
  public readonly cardinalityEstimateConstruction: boolean;
  public readonly forceGetIfUrlLengthBelow: number;

  public constructor(args: IActorQuerySourceIdentifyHypermediaSparqlArgs) {
    super(args, 'sparql');
  }

  public async testMetadata(
    action: IActionQuerySourceIdentifyHypermedia,
  ): Promise<TestResult<IActorQuerySourceIdentifyHypermediaTest>> {
    if (!action.forceSourceType && !this.forceSourceType && !action.metadata.sparqlService &&
      !(this.checkUrlSuffix && (action.url.endsWith('/sparql') || action.url.endsWith('/sparql/')))) {
      return failTest(`Actor ${this.name} could not detect a SPARQL service description or URL ending on /sparql.`);
    }
    return passTest({ filterFactor: 1 });
  }

  public async run(action: IActionQuerySourceIdentifyHypermedia): Promise<IActorQuerySourceIdentifyHypermediaOutput> {
    this.logInfo(action.context, `Identified ${action.url} as sparql source with service URL: ${action.metadata.sparqlService || action.url}`);

    const dataFactory: ComunicaDataFactory = action.context.getSafe(KeysInitQuery.dataFactory);
    const algebraFactory = new Factory(dataFactory);
    const isSingularSource = action.context.get(KeysQueryOperation.querySources)?.length === 1;
    const source = new QuerySourceSparql(
      (action.forceSourceType ?? this.forceSourceType) ? action.url : action.metadata.sparqlService || action.url,
      action.context,
      this.mediatorHttp,
      this.bindMethod,
      dataFactory,
      algebraFactory,
      await BindingsFactory.create(this.mediatorMergeBindingsContext, action.context, dataFactory),
      this.forceHttpGet,
      this.cacheSize,
      this.countTimeout,
      // Cardinalities can be infinity when we're querying just a single source.
      this.cardinalityCountQueries && !isSingularSource,
      this.cardinalityEstimateConstruction,
      this.forceGetIfUrlLengthBelow,
      action.metadata.defaultGraph,
      action.metadata.unionDefaultGraph,
      action.metadata.datasets,
    );
    return { source };
  }
}

export interface IActorQuerySourceIdentifyHypermediaSparqlArgs extends IActorQuerySourceIdentifyHypermediaArgs {
  /**
   * The HTTP mediator
   */
  mediatorHttp: MediatorHttp;
  /**
   * A mediator for creating binding context merge handlers
   */
  mediatorMergeBindingsContext: MediatorMergeBindingsContext;
  /**
   * If URLs ending with '/sparql' should also be considered SPARQL endpoints.
   * @default {true}
   */
  checkUrlSuffix: boolean;
  /**
   * If non-update queries should be sent via HTTP GET instead of POST
   * @default {false}
   */
  forceHttpGet: boolean;
  /**
   * The cache size for COUNT queries.
   * @range {integer}
   * @default {1024}
   */
  cacheSize?: number;
  /**
   * If provided, forces the source type of a source.
   * @default {false}
   */
  forceSourceType?: boolean;
  /**
   * The query operation for communicating bindings.
   * @default {values}
   */
  bindMethod: BindMethod;
  /**
   * Timeout in ms of how long count queries are allowed to take.
   * If the timeout is reached, an infinity cardinality is returned.
   * @default {3000}
   */
  countTimeout: number;
  /**
   * If count queries should be sent to obtain the cardinality of (sub)queries.
   * If set to false, resulting cardinalities will always be considered infinity.
   * @default {true}
   */
  cardinalityCountQueries: boolean;
  /**
   * If estimates for queries should be constructed locally from sub-query cardinalities.
   * If set to false, count queries will used for cardinality estimation at all levels.
   * @default {false}
   */
  cardinalityEstimateConstruction: boolean;
  /**
   * Force an HTTP GET instead of default POST (when forceHttpGet is false)
   * when the url length (including encoded query) is below this number.
   * @default {600}
   */
  forceGetIfUrlLengthBelow?: number;
}

export type BindMethod = 'values' | 'union' | 'filter';
