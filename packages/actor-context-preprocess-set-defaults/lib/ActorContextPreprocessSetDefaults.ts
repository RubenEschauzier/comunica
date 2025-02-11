import type {
  IActorContextPreprocessOutput,
  IActorContextPreprocessArgs,
  IActionContextPreprocess,
} from '@comunica/bus-context-preprocess';
import { ActorContextPreprocess } from '@comunica/bus-context-preprocess';
import { KeysCaches, KeysCore, KeysInitQuery, KeysQuerySourceIdentify } from '@comunica/context-entries';
import type { IAction, IActorTest, TestResult } from '@comunica/core';
import { passTestVoid } from '@comunica/core';
import type { FunctionArgumentsCache, Logger, ISourceState} from '@comunica/types';
import type * as RDF from '@rdfjs/types';
import { DataFactory } from 'rdf-data-factory';
import { LRUCache } from 'lru-cache';

/**
 * A comunica Set Defaults Context Preprocess Actor.
 */
export class ActorContextPreprocessSetDefaults extends ActorContextPreprocess {
  private readonly defaultFunctionArgumentsCache: FunctionArgumentsCache;
  private readonly withinQueryMaxCacheSize: number;
  public readonly logger: Logger;

  public constructor(args: IActorContextPreprocessSetDefaultsArgs) {
    super(args);
    this.defaultFunctionArgumentsCache = {};
  }

  public async test(_action: IAction): Promise<TestResult<IActorTest>> {
    return passTestVoid();
  }

  public async run(action: IActionContextPreprocess): Promise<IActorContextPreprocessOutput> {
    let context = action.context;

    if (action.initialize) {
      // Set default values
      context = context
        .setDefault(KeysInitQuery.queryTimestamp, new Date())
        .setDefault(KeysInitQuery.queryTimestampHighResolution, performance.now())
        .setDefault(KeysQuerySourceIdentify.sourceIds, new Map())
        .setDefault(KeysCore.log, this.logger)
        .setDefault(KeysInitQuery.functionArgumentsCache, this.defaultFunctionArgumentsCache)
        .setDefault(KeysQuerySourceIdentify.hypermediaSourcesAggregatedStores, new Map())
        .setDefault(KeysInitQuery.dataFactory, new DataFactory())
        .setDefault(KeysCaches.withinQueryStoreCache, 
          new LRUCache<string, Promise<ISourceState>>({ max: this.withinQueryMaxCacheSize }))

      // Handle default query format
      let queryFormat: RDF.QueryFormat = { language: 'sparql', version: '1.1' };
      if (context.has(KeysInitQuery.queryFormat)) {
        queryFormat = context.get(KeysInitQuery.queryFormat)!;
        if (queryFormat.language === 'graphql') {
          context = context.setDefault(KeysInitQuery.graphqlSingularizeVariables, {});
        }
      } else {
        context = context.set(KeysInitQuery.queryFormat, queryFormat);
      }
    }

    return { context };
  }
}

export interface IActorContextPreprocessSetDefaultsArgs extends IActorContextPreprocessArgs {
  /**
   * The logger of this actor
   * @default {a <npmd:@comunica/logger-void/^4.0.0/components/LoggerVoid.jsonld#LoggerVoid>}
   */
  logger: Logger;
  /**
   * The maximum number of entries in the within-query LRU cache, set to 0 to disable.
   * @range {integer}
   * @default {100}
   */
  withinQueryMaxCacheSize: number
}
