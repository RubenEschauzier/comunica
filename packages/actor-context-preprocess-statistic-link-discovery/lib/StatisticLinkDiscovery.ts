// eslint-disable-next-line import/no-nodejs-modules
import { EventEmitter } from 'events';
import type { ILink, IDiscoverEventData, IStatisticDiscoveredLinks } from '@comunica/types';

export class StatisticLinkDiscovery implements IStatisticDiscoveredLinks {
  // Number of discover events tracked
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
  public log: ((message: string, data?: (() => any)) => void);

  public constructor(log: (message: string, data?: (() => any)) => void) {
    this.discoverEvents = 0;

    this.edgeList = new Set();
    this.metadata = {};

    this.statisticEvents = new EventEmitter();

    this.log = log;
  }

  public setDiscoveredLink(link: ILink, parent: ILink): boolean {
    // If self-edge or duplicate edge we don't track.
    if (link.url === parent.url || this.edgeList.has(JSON.stringify([ parent.url, link.url ]))) {
      return false;
    }

    this.edgeList.add(JSON.stringify([ parent.url, link.url ]));

    const discoveredLinkMetadata = {
      ...link.metadata,
      discoveredTimestamp: Date.now(),
      discoverOrder: this.discoverEvents,
    };

    if (discoveredLinkMetadata) {
      // Retain previous metadata if this link has already been discovered, and add any metadata in the passed link
      this.metadata[link.url] = this.metadata[link.url] ?
          [ ...this.metadata[link.url], discoveredLinkMetadata ] :
          [ discoveredLinkMetadata ];
    }

    const discoverEventData: IDiscoverEventData = {
      edge: [ parent.url, link.url ],
      metadataChild: this.metadata[link.url],
      metadataParent: this.metadata[parent.url],
    };

    this.emit(discoverEventData);

    // Log info if available
    this.log('Discover Event', () => ({
      edge: [ parent.url, link.url ],
      metadataChild: this.metadata[link.url],
      metadataParent: this.metadata[parent.url],
    }));

    // Increment number of discover events to track discover order
    this.discoverEvents += 1;

    return true;
  }

  public addListener(cb: (arg0: IDiscoverEventData) => void): void {
    this.statisticEvents.addListener('data', cb);
  }

  public emit(data: IDiscoverEventData): void {
    this.statisticEvents.emit('data', data);
  }
}
