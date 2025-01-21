
import { KeysCaches } from '@comunica/context-entries';
import type { IAction, IActorTest, TestResult } from '@comunica/core';
import { passTestVoid } from '@comunica/core';
import { LRUCache } from 'lru-cache';
import { ActorInitCaches, IActionInitCaches, IActorInitCachesArgs, IActorInitCachesOutput } from '@comunica/bus-init-caches';
import CachePolicy = require('http-cache-semantics');

/**
 * A comunica Policy Cache Lru Init Caches Actor.
 */
export class ActorInitCachesPolicyCacheLru extends ActorInitCaches {
  private maxCacheSize: number;

  public constructor(args: IActorInitCachesPolicyCacheLru) {
    super(args);
  }

  public async test(_action: IAction): Promise<TestResult<IActorTest>> {
    return passTestVoid();
  }
  public async run(action: IActionInitCaches): Promise<IActorInitCachesOutput> {
    let cacheContext = action.cacheContext;
    cacheContext = cacheContext.setDefault(KeysCaches.policyCache, 
      new LRUCache<string, CachePolicy>({ max: this.maxCacheSize })
    );
    return { cacheContext };
  }
}

export interface IActorInitCachesPolicyCacheLru extends IActorInitCachesArgs {

  /**
   * The maximum number of entries in the within-query LRU cache, set to 0 to disable.
   * @range {integer}
   * @default {100}
   */
  maxCacheSize: number
}
