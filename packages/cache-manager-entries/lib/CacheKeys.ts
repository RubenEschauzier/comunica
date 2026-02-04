import type { ISourceState, ISourceStateBloomFilter } from '@comunica/types';
import { CacheKey } from './CacheKey';

export const CacheEntrySourceState = {
  /**
   * Cache for storing source states in a persistent manner over multiple queries
   */
  cacheSourceState: new CacheKey<
  ISourceState,
ISourceState,
{ headers: Headers }
  >('@comunica/persistent-cache-manager:sourceState'),
  /**
   * Cache stores indexed source states and acts as a query source. Getting from cache
   * emits the bindings to any queryBinding calls associated with that get
   */
  cacheSourceStateQuerySource: new CacheKey<ISourceState, ISourceState, { headers: Headers } >(
    '@comunica/persistent-cache-manager:sourceStateQuerySource',
  ),
  /**
   * Cache stores indexed source states with subject and object based bloomfilters
   * and acts as a query source. Getting from cache
   * emits the bindings to any queryBinding calls associated with that get 
   */
  cacheSourceStateQuerySourceBloomFilter: new CacheKey<
  ISourceStateBloomFilter,
  ISourceStateBloomFilter, 
  { headers: Headers }
  >(
    '@comunica/persistent-cache-manager:sourceStateQuerySourceBloomFilter',
  )

};
