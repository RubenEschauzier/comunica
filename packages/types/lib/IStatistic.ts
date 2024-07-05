// eslint-disable-next-line
import type { EventEmitter } from 'events';
import type { ILink } from '@comunica/bus-rdf-resolve-hypermedia-links';
import type { IQuerySource } from './IQuerySource';
import { IActionContextKey } from './IActionContext';

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
  addListener(cb: (arg0: T) => void): void;

  /**
   * Emits new data event to all listeners
   * @param data The newly observed data
   */
  emit(data: T): void;

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
  metadataDiscoveredNode: Record<any, any>[];
  /**
   * Metadata of the parent of the discovered ndoe
   */
  metadataParentDiscoveredNode: Record<any, any>[];
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