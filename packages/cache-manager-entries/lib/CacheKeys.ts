import type { ISourceState, ISourceStateBloomFilter } from '@comunica/types';
import { CacheKey } from './CacheKey';

export const CacheEntrySourceState = {
  /**
   * Cache for storing source states in a persistent manner over multiple queries
   */
  cacheSourceStateUnIndexed: new CacheKey<
  ISourceState,
ISourceState,
{ headers: Headers }
  >('@comunica/persistent-cache-manager:sourceState'),
  /**
   * Cache stores indexed source states and acts as a query source. Getting from cache
   * emits the bindings to any queryBinding calls associated with that get
   */
  cacheSourceStateIndexed: new CacheKey<ISourceState, ISourceState, 
  { headers: Headers } 
  >(
    '@comunica/persistent-cache-manager:sourceStateQuerySource',
  ),
  cacheSourceStateIndexedDisk: new CacheKey<ISourceState, ISourceState, 
  { headers: Headers  } 
  >(
    '@comunica/persistent-cache-manager:cacheSourceStateIndexedDisk',
  ),

};

export const CacheEntryDataSummary = {
  cacheCsetCpsSummary: new CacheKey<
    ISourceState, ISourceState, { headers: Headers, updateTraverse?: boolean } 
  >(
    '@comunica/persistent-cache-manager:data-summary-cset-cp',
  ),
}