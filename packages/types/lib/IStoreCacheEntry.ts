import CachePolicy = require('http-cache-semantics');
import type { ISourceState } from './IQuerySource';

export interface IStoreCacheEntry {
  policy: CachePolicy;
  store: ISourceState;
}
