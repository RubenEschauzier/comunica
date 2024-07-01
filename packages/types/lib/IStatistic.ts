import type { ILink } from '@comunica/bus-rdf-resolve-hypermedia-links';
import { IQuerySource } from './IQuerySource';
import { EventEmitter } from 'stream';
import { Bindings } from './Bindings';

interface IStatistic{
  statisticEvents: EventEmitter;

  getEmitter: () => EventEmitter;
}

export interface IStatisticDiscoveredLinks extends IStatistic {
  /**
   * Set parent - child relation of discovered link
   * @param link The link discovered by the engine
   * @returns Boolean indicating success
   */
  setDiscoveredLink: (link: ILink, parent: ILink) => boolean;

  /**
   *
   * @returns List of all discovered [ parent - child ] relations
   */
  getDiscoveredLinks: () => Set<[string, string]>;
}

export interface IStatisticDereferencedLinks extends IStatistic {
  /**
   * Track dereference event by engine
   * @param link Link dereferenced by the engine
   * @returns Boolean indicating success
   */
  setDereferenced: (link: ILink, source: IQuerySource) => boolean;

  /**
   * Returns ordered list of dereferenced links, indicating order of
   * traversal of engine
   * @returns list of URLs
   */
  getDereferencedLinks: () => ILink[];
}
