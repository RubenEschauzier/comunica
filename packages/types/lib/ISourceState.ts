// TODO Fix this, it shouldn't depend on a bus
import { IActionQuerySourceDereferenceLink } from "@comunica/bus-query-source-dereference-link";
import { ICachePolicy } from "./ICachePolicy";
import { MetadataBindings } from "./IMetadata";
import { IQuerySource } from "./IQuerySource";
import { ILink } from "./ILink";

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
