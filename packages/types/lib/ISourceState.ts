// TODO Fix this, it shouldn't depend on a bus
import type { IActionQuerySourceDereferenceLink } from '@comunica/bus-query-source-dereference-link';
import type { ICachePolicy } from './ICachePolicy';
import type { ILink } from './ILink';
import type { MetadataBindings } from './IMetadata';
import type { IQuerySource } from './IQuerySource';
import { ScalableBloomFilter } from 'bloom-filters';

/**
 * The current state of a source.
 * This is needed for following links within a source.
 */
export interface ISourceState {
  /**
   * The link to this source.
   */
  link: ILink;
  /**
   * A source.
   */
  source: IQuerySource;
  /**
   * The source's initial metadata.
   */
  metadata: MetadataBindings;
  /**
   * All dataset identifiers that have been passed for this source.
   */
  handledDatasets: Record<string, boolean>;
  /**
   * The cache policy of the request's response.
   */
  cachePolicy?: ICachePolicy<IActionQuerySourceDereferenceLink>;
  /**
   * Any headers of the dereferenced source page
   */
  headers?: Headers | undefined;
}

export interface ISourceStateBloomFilter extends ISourceState {
  /**
   * A bloom filter used to check if an operation will have answers
   * for a given source
   */
  bloomFilter?: ScalableBloomFilter;
}