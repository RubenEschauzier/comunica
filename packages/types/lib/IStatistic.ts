// eslint-disable-next-line
import type { EventEmitter } from 'events';
import type { ILink } from '@comunica/bus-rdf-resolve-hypermedia-links';
import type { ActionContextKey } from '@comunica/core';
import type { IActionContext } from './IActionContext';
import type { IQuerySource } from './IQuerySource';

interface IStatistic {
  statisticEvents: EventEmitter;

  getEmitter: () => EventEmitter;
}

interface IAggregateStatistic extends IStatistic {
  /**
   * The statistics this aggregate statistics aggregates over
   */
  toAggregate: ActionContextKey<IStatistic>[];

  /**
   * Attaches listeners to all statistics this class aggregates over
   */
  attachListeners: (context: IActionContext) => boolean;

  /**
   * When a new event is emitted this function is called,
   */
  onStatisticEvent: (data: any) => any;

  /**
   * Gets the underlying tracked data
   */
  getAggregateStatistic: () => any;
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
  getDiscoveredLinks: () => [string, string][];
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
