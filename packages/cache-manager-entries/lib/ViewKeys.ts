import type { BindingsStream, ISourceState } from '@comunica/types';
import { ViewKey } from './ViewKey';
import type { Algebra } from '@comunica/utils-algebra';

// TODO: These can probably be moved to link traversal, just leaving the key types in base comunica
export const CacheSourceStateViews = {
  /**
   * Cache for storing source states in a persistent manner over multiple queries
   */
  cacheSourceStateView:
    new ViewKey<ISourceState, { url: string }, ISourceState>('@comunica/persistent-cache-manager:sourceStateView'),

  cacheQueryView:
    new ViewKey<ISourceState, { url: string, mode: 'get' | 'query', operation: Algebra.Operation }, BindingsStream | ISourceState>('@comunica/persistent-cache-manager:cacheQuery'),
  
  cacheCountView:
    new ViewKey<
      ISourceState,
      { operation: Algebra.Operation, documents: string[] },
      number
    >('@comunica/persistent-cache-manager:cacheCount'),
};
