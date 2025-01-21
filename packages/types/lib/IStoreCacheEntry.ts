import CachePolicy = require('http-cache-semantics');
import { ISourceState } from './IQuerySource';

export interface IStoreCacheEntry{
    policy: CachePolicy
    store: ISourceState
}