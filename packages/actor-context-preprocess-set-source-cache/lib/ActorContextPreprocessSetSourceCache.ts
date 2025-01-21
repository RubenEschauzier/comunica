import type {
  IActorContextPreprocessOutput,
  IActorContextPreprocessArgs,
  IActionContextPreprocess,
} from '@comunica/bus-context-preprocess';
import { ActorContextPreprocess } from '@comunica/bus-context-preprocess';
import { KeysCaches } from '@comunica/context-entries';
import type { IAction, IActorTest, TestResult } from '@comunica/core';
import { passTestVoid } from '@comunica/core';
import type { ISourceState, IStoreCacheEntry} from '@comunica/types';
import CachePolicy = require('http-cache-semantics');
import { LRUCache } from 'lru-cache';

/**
 * A comunica Set Source Cache Context Preprocess Actor.
 */
export class ActorContextPreprocessSetSourceCache extends ActorContextPreprocess {
  private cacheSize: number;
  private storeCacheTest: LRUCache<string, IStoreCacheEntry>;
  private policyCache: LRUCache<string, CachePolicy>;
  private storeCache: LRUCache<string, ISourceState>;

  public constructor(args: IActorContextPreprocessSetSourceCacheArgs) {
    super(args);
    this.storeCacheTest = new LRUCache<string, IStoreCacheEntry>({ max: this.cacheSize });
    this.policyCache = new LRUCache<string, CachePolicy>({ max: this.cacheSize });
    this.storeCache = new LRUCache<string, ISourceState>({ max: this.cacheSize });
  }

  public async test(_action: IAction): Promise<TestResult<IActorTest>> {
    return passTestVoid();
  }

  public async run(action: IActionContextPreprocess): Promise<IActorContextPreprocessOutput> {
    let context = action.context;
    context = context
      .setDefault(KeysCaches.storeCacheTest, this.storeCacheTest )
      .setDefault(KeysCaches.policyCache, this.policyCache )
      .setDefault(KeysCaches.storeCache, this.storeCache );
    
    return { context }
  }
}


export interface IActorContextPreprocessSetSourceCacheArgs extends IActorContextPreprocessArgs {
  /**
   * The maximum number of entries in the source cache, set to 0 to disable.
   * @range {integer}
   * @default {100}
   */
  cacheSize: number
}
