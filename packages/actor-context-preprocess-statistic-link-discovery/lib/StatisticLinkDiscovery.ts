import type { ILink } from '@comunica/bus-rdf-resolve-hypermedia-links';
import type { IStatisticDiscoveredLinks, Logger } from '@comunica/types';
import { EventEmitter } from 'node:events';

export class StatisticLinkDiscovery implements IStatisticDiscoveredLinks {
  // What query this statistic tracker belongs to. 
  public query: string;

  // Directed edge list
  public edgeList: Set<[string, string]>;

  // Metadata is saved as follows: First key indicates what url this metadata belongs to, while value is
  // a list of all metadata objects recorded. This list can contain multiple Records as links can be discovered from
  // multiple data sources.
  public metadata: Record<string, Record<any, any>[]>;

  // Emitter that indicates when a new discover statistic is recorded. This can be used downstream for, e.g, aggregate
  // statistic trackers
  public statisticEvents: EventEmitter;

  // Logger that emits the recorded data.
  public logger: Logger | undefined;

  public constructor(query: string, logger?: Logger) {
    this.query = query;

    this.edgeList = new Set();
    this.metadata = {};

    this.statisticEvents = new EventEmitter<IDiscoverEvent>();

    this.logger = logger;
  }

  public setDiscoveredLink(link: ILink, parent: ILink) {
    // If self-edge or duplicate edge we don't track.
    if (link.url === parent.url || this.edgeList.has([ parent.url, link.url ])) {
      return false;
    }
    this.edgeList.add([ parent.url, link.url ]);
    link.metadata = {
      ...link.metadata,
      discoveredTimestamp: Date.now(),
      discoverOrder: this.metadata.length
    }
    if (link.metadata){
      // Retain previous metadata if this link has already been discovered, and add any metadata in the passed link
      this.metadata[link.url] = this.metadata[link.url] ? [ ...this.metadata[link.url], link.metadata ] : [ link.metadata ];
    }

    // Add timestamp for discovery of link
    this.metadata[link.url] = {
      ...this.metadata[link.url], 
      
    }

    const discoverEventData: IDiscoverEventData = {
      edge: [parent.url, link.url],
      metadataDiscoveredNode: this.metadata[link.url]
    }
    this.getEmitter().emit('discoverEvent', discoverEventData);

    if (this.logger){
      this.logger.trace('Discover Event', { 
        data: JSON.stringify({
          statistic: "discoveredLinks", 
          query: this.query, 
          discoveredLinks: [...this.getDiscoveredLinks()],
          metadata: this.getMetadata()
        })
      });
    }


    return true;
  }
  
  public getDiscoveredLinks(){
    return this.edgeList
  }

  public getEmitter(): EventEmitter {
    return this.statisticEvents;
  }


  public getMetadata(){
    return this.metadata
  }
}

export interface IDiscoverEventData{
  edge: [string, string];
  metadataDiscoveredNode: Record<any, any>
}

export interface IDiscoverEvent {
  (event: string): IDiscoverEventData;
}
