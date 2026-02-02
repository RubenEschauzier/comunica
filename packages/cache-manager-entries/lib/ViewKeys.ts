import type { BindingsStream, ISourceState } from '@comunica/types';
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
    new ViewKey<ISourceState, { url: string }, ISourceState>('@comunica/persistent-cache-manager:sourceStateView'),

  cacheQueryView:
    new ViewKey<
      ISourceState,
      { url: string, mode: 'get', action: IActionQuerySourceDereferenceLink } | { mode: 'queryBindings' | 'queryQuads', operation: Algebra.Operation},
      AsyncIterator<BindingsStream> | AsyncIterator<AsyncIterator<RDF.Quad>> | ISourceState
    >('@comunica/persistent-cache-manager:cacheQuery'),
  
  cacheCountView:
    new ViewKey<
      ISourceState,
      { operation: Algebra.Operation, documents: string[] },
      number
    >('@comunica/persistent-cache-manager:cacheCount'),
};


