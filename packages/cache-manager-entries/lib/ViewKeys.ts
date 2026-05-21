import type { BindingsStream, IActionContext, ILink, ISourceState, ISourceStateBloomFilter } from '@comunica/types';
import { ViewKey } from './ViewKey';
import { Algebra } from '@comunica/utils-algebra';
import { AsyncIterator } from 'asynciterator';
import type * as RDF from '@rdfjs/types';
import { IActionQuerySourceDereferenceLink } from '@comunica/bus-query-source-dereference-link';

// TODO: These can probably be moved to link traversal, just leaving the key types in base comunica
export const CacheSourceStateViews = {
  /**
   * Cache for storing source states in a persistent manner over multiple queries
   */
  cacheSourceStateView:
    new ViewKey<
      ISourceState, 
      {       
        url: string,
        extractLinksQuadPattern?: boolean,
        action: IActionQuerySourceDereferenceLink;
      }, 
      ISourceState
    >('@comunica/persistent-cache-manager:sourceStateView'),
  /**
   * 
   */
  indexedCacheGetView:
    new ViewKey<
      ISourceState,
      { url: string, mode: 'get', action: IActionQuerySourceDereferenceLink },
      BindingsStream | AsyncIterator<RDF.Quad> | ISourceState
    >('@comunica/persistent-cache-manager:cacheQuery'),

  indexedCacheCountView:
    new ViewKey<
      ISourceState,
      { operation: Algebra.Operation },
      number
    >('@comunica/persistent-cache-manager:cacheCount'),

  indexedCacheCountViewOfflineTraversal:
    new ViewKey<
      ISourceState,
      { operation: Algebra.Operation, seeds: ILink[], query: Algebra.BaseOperation },
      number
    >('@comunica/persistent-cache-manager:cacheCount'),

};


