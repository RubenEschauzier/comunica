// eslint-disable-next-line
import { EventEmitter } from 'events';
import type { ILink } from '@comunica/bus-rdf-resolve-hypermedia-links';
import type { IDiscoverEventData, IStatisticDiscoveredLinks, Logger } from '@comunica/types';

export class StatisticLinkDiscovery implements IStatisticDiscoveredLinks {
  // What query this statistic tracker belongs to.
  public query: string;
  public discoverEvents: number;

  // Directed edge list
  public edgeList: Set<string>;

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
    this.discoverEvents = 0;

    this.edgeList = new Set();
    this.metadata = {};

    this.statisticEvents = new EventEmitter();

    this.logger = logger;
  }

  public setDiscoveredLink(link: ILink, parent: ILink) {
    // If self-edge or duplicate edge we don't track.
    if (link.url === parent.url || this.edgeList.has(JSON.stringify([ parent.url, link.url ]))) {
      return false;
    }

    this.edgeList.add(JSON.stringify([ parent.url, link.url ]));

    link.metadata = {
      ...link.metadata,
      discoveredTimestamp: Date.now(),
      discoverOrder: this.discoverEvents,
    };

    if (link.metadata) {
      // Retain previous metadata if this link has already been discovered, and add any metadata in the passed link
      this.metadata[link.url] = this.metadata[link.url] ? [ ...this.metadata[link.url], link.metadata ] : [ link.metadata ];
    }

    const discoverEventData: IDiscoverEventData = {
      edge: [ parent.url, link.url ],
      metadataDiscoveredNode: this.metadata[link.url],
      metadataParentDiscoveredNode: this.metadata[parent.url],
    };

    this.getEmitter().emit('data', discoverEventData);

    // Increment number of discover events to track discover order
    this.discoverEvents += 1;

    if (this.logger) {
      this.logger.trace('Discover Event', {
        data: JSON.stringify({
          statistic: 'discoveredLinks',
          query: this.query,
          discoveredLinks: this.getDiscoveredLinks(),
          metadata: this.getMetadata(),
        }),
      });
    }

    return true;
  }

  public getDiscoveredLinks() {
    const edgeListArray: [string, string][] = [];

    for (const elem of this.edgeList) {
      edgeListArray.push(JSON.parse(elem));
    }
    return edgeListArray;
  }

  public getEmitter(): EventEmitter {
    return this.statisticEvents;
  }

  public getMetadata() {
    return this.metadata;
  }
}
