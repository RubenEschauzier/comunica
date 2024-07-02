// eslint-disable-next-line
import type { EventEmitter } from 'events';
import type { ILink } from '@comunica/bus-rdf-resolve-hypermedia-links';
import type { ActionContextKey } from '@comunica/core';
import type { IActionContext } from './IActionContext';
import type { IQuerySource } from './IQuerySource';

export interface IStatistic {
  /**
   * Query string of tracked execution
   */
  query: string, 

  statisticEvents: EventEmitter;

  getEmitter: () => EventEmitter;
}

export interface IAggregateStatistic extends IStatistic {
  /**
   * The statistics this aggregate statistic aggregates over
   */
  toAggregate: Record<string, IStatistic>;

  /**
   * Attaches listeners to all statistics this class aggregates over
   */
  attachListeners: (context: IActionContext) => boolean;

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

export interface IDiscoverEventData {
  /**
   * The edge discovered during query execution
   */
  edge: [string, string];
  /**
   * Metadata of the discovered node
   */
  metadataDiscoveredNode: Record<any, any>[];
  /**
   * Metadata of the parent of the discovered ndoe
   */
  metadataParentDiscoveredNode: Record<any, any>[];
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

/**
 * Interface describing what data will be emitted when the topology is updated
 * Will emit all data so far instead of only the update that happened to the topology
 */
export interface ITopologyEventData {
  /**
   * What type of update happened to the topology
   */
  type: 'dereference' | 'discover'
  /**
   * The edge list discovered during query execution
   */
  edgeList: Set<string>;
  /**
   * The metadata for each node
   */
  metadata: Record<string, any>[];
}
