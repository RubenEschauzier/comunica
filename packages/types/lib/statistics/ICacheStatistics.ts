export interface ICacheStatistics {
  /**
   * Percentage of http requests that are cached
   */
  hitRate: number;
  /**
   * Number of cache hits that required no revalidation request
   */
  hitRateNoRevalidation: number;
  /**
   * Total number of evictions
   */
  evictions: number;
  /**
   * Total number of evicted triples
   */
  evictionsTriples: number;
  /**
   * Percentage of cache size that was evicted for a given query
   */
  evictionPercentage: number;
  testId?: number;
}
