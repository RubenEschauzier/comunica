// eslint-disable-next-line
import type { EventEmitter } from 'events';
import type { ILink } from '@comunica/bus-rdf-resolve-hypermedia-links';
import type { ActionContextKey } from '@comunica/core';
import type { IActionContext } from './IActionContext';
import type { IQuerySource } from './IQuerySource';

export interface IStatistic<T> {

  statisticEvents: EventEmitter;

  addListener(cb: (arg0: T) => void): void;

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

export interface IStatisticDereferencedLinks extends IStatistic<ILink> {
  /**
   * Track dereference event by engine
   * @param link Link dereferenced by the engine
   * @returns Boolean indicating success
   */
  setDereferenced: (link: ILink, source: IQuerySource) => boolean;
}