import type { AsyncIterator } from 'asynciterator';

export interface IPersistentCache<T> {
  get: (key: string) => Promise<T | undefined>;
  getMany: (keys: string[]) => Promise<(T | undefined)[]>;
  set: (key: string, value: T) => Promise<void>;
  has: (key: string) => Promise<boolean>;
  delete: (key: string) => Promise<boolean>;
  entries: () => AsyncIterator<[string, T]>;
  size: () => Promise<number>;
  serialize: () => Promise<void>;
  /**
   * Starts tracking cache performance metrics
   * @returns Metrics object which will be updated during query execution
   */
  startSession: () => ICacheMetrics;
  /**
   * Stops tracking cache performance metrics and
   * resets the metrics
   * @returns Final cache performance metrics
   */
  endSession: () => ICacheMetrics;
}

/**
 * Performance characteristics tracked during query executions
 */
export interface ICacheMetrics {
  /**
   * Number of cache hits
   */
  hits: number;
  /**
   * Number of cache misses
   */
  misses: number;
  /**
   * Number of eviction events
   */
  evictions: number;
  /**
   * Calculated size of the eviction events, depending on
   * the size of the evicted item
   */
  evictionsCalculatedSize: number;
  /**
   * The percentage of the max size of the cache evicted during the
   * tracking period
   */
  evictionPercentage: number;
}
/**
 * Interface of class that sets a value in a given cache for a given key. This can
 * be a simple URL -> ISourceState mapping or URL -> Source Data Summary. The
 * computation and logic of what to store is done in the setInCache function.
 * T = The type of data being stored (e.g., ISourceState)
 * C = The context needed to derive additional storage logic
 */
export interface ISetFn<I, S, C> {
  setInCache: (
    key: string,
    value: I,
    cache: IPersistentCache<S>,
    context: C
  ) => Promise<void>;
}

export interface ICacheRegistryEntry<I, S, C> {
  cache: IPersistentCache<S>;
  setFn: ISetFn<I, S, C>;
}

/**
 * Interface for a class that returns a view over the cache. This can be
 * just retrieving a key, a summary of the content of the cache, or
 * the content of a cache matching a certain operator (e.g. triple pattern).
 * T the output type of the constructed view
 * C the context needed to construct the view
 */
export interface ICacheView<S, C, K> {
  construct: (cache: IPersistentCache<S>, context: C) => Promise<K | undefined>;
}

