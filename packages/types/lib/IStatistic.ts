// eslint-disable-next-line import/no-nodejs-modules
import type { EventEmitter } from 'events';
import type { ILink } from '@comunica/bus-rdf-resolve-hypermedia-links';
import type { IActionContextKey } from './IActionContext';
import type { IQuerySource } from './IQuerySource';

export interface IStatistic<T> {
  /**
   * All statistic trackers have an event emitter that makes tracked
   * information available
   */
  statisticEvents: EventEmitter;

  /**
   * Attaches a listener with callback to the statistic tracker. This can be used
   * by other parts of Comunica to use tracked statistics
   * @param cb The callback to fire when a new data event occurs. This should take
   * the data tracked by the statistics tracker as argument
   */
  addListener: (cb: (arg0: T) => void) => void;

  /**
   * Emits new data event to all listeners
   * @param data The newly observed data
   */
  emit: (data: T) => void;

}

export interface IStatisticDiscoveredLinks extends IStatistic<IDiscoverEventData> {
  /**
   * Set parent - child relation of discovered link
   * @param link The link discovered by the engine
   * @returns Boolean indicating success
   */
  setDiscoveredLink: (link: ILink, parent: ILink) => boolean;
}

export interface IStatisticDereferencedLinks extends IStatistic<ILink> {
  /**
   * Track dereference event by engine
   * @param link Link dereferenced by the engine
   * @returns Boolean indicating success
   */
  setDereferenced: (link: ILink, source: IQuerySource) => boolean;
}

export interface IDiscoverEventData {
  /**
   * The edge discovered during query execution
   */
  edge: [string, string];
  /**
   * Metadata of the discovered node
   */
  metadataChild: Record<any, any>[];
  /**
   * Metadata of the parent of the discovered ndoe
   */
  metadataParent: Record<any, any>[];
}

/**
 * Interface describing what data will be emitted when the topology is updated.
 * Will emit a notification that the topology has changed, prompting consumers of this
 * aggregate statistic to process the data.
 */
export interface ITopologyEventNotification {
  /**
   * What type of update happened to the topology
   */
  type: 'dereference' | 'discover';
}

/**
 * Interface that holds all tracked statistics
 */
export interface IStatisticsHolder {
  set: <V>(key: IActionContextKey<V>, value: V) => void;
  delete: <V>(key: IActionContextKey<V>) => void;
  get: <V>(key: IActionContextKey<V>) => V | undefined;
  has: <V>(key: IActionContextKey<V>) => boolean;
  keys: () => IActionContextKey<any>[];
}
