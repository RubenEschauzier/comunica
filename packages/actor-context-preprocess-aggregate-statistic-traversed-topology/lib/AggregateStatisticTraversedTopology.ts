// eslint-disable-next-line import/no-nodejs-modules
import { EventEmitter } from 'events';
import { StatisticsHolder } from '@comunica/actor-context-preprocess-set-defaults';
import { KeysTrackableStatistics } from '@comunica/context-entries';
import type {
  ILink,
  IDiscoverEventData,
  IStatistic,
  IStatisticDereferencedLinks,
  IStatisticDiscoveredLinks,
  ITopologyEventNotification,
} from '@comunica/types';

export class AggregateStatisticTraversedTopology implements IStatistic<ITopologyEventNotification> {
  // Emitter that indicates when an update to the aggregate statistic happened
  public statisticEvents: EventEmitter;
  public toAggregate: StatisticsHolder;

  // Tracked data
  public edgeList: Set<string>;
  public metadata: Record<string, Record<string, any>>;

  public log: ((message: string, data?: (() => any)) => void);

  public constructor(
    discoverStatistic: IStatisticDiscoveredLinks,
    dereferenceStatistic: IStatisticDereferencedLinks,
    log: (message: string, data?: (() => any)) => void,
  ) {
    this.toAggregate = new StatisticsHolder();
    this.toAggregate.set(KeysTrackableStatistics.discoveredLinks, discoverStatistic);
    this.toAggregate.set(KeysTrackableStatistics.dereferencedLinks, dereferenceStatistic);

    this.edgeList = new Set();
    this.metadata = {};

    this.attachListeners();
    this.statisticEvents = new EventEmitter();

    this.log = log;
  }

  public attachListeners(): boolean {
    (<IStatisticDiscoveredLinks> this.toAggregate.get(KeysTrackableStatistics.discoveredLinks))
      .addListener(this.processDiscoverEvent.bind(this));

    (<IStatisticDereferencedLinks> this.toAggregate.get(KeysTrackableStatistics.dereferencedLinks))
      .addListener(this.processDereferenceEvent.bind(this));

    return true;
  }

  public addListener(cb: (arg0: ITopologyEventNotification) => void): void {
    this.statisticEvents.addListener('data', cb);
  }

  public processDiscoverEvent(data: IDiscoverEventData): void {
    this.edgeList.add(JSON.stringify(data.edge));

    // Aggregate array of metadata entries for single node into one metadata entry with
    // array values
    const aggregatedMetadata: Record<any, any> = {};
    for (const metadata of data.metadataChild) {
      for (const key of Object.keys(metadata)) {
        aggregatedMetadata[key] = aggregatedMetadata[key] ?
            [ ...aggregatedMetadata[key], metadata[key] ] :
            [ metadata[key] ];
      }
    }

    // Merge exising child metadata with incoming child metadata.
    // On key overlap this will take values from aggregatedMetadata as that is the most recently updated
    this.metadata[data.edge[1]] = { ...this.metadata[data.edge[1]], ...aggregatedMetadata };

    // Construct update topology event object
    const topologyNotification: ITopologyEventNotification = {
      type: 'discover',
    };

    // Construct object containing all changed parts of topology
    const changedPartsTopology: ITopologyChangeDiscover = {
      edge: data.edge,
      metadataChild: this.metadata[data.edge[1]],
    };

    // Log what data changed in due to this function
    // We log the changed data in case a program outside of Comunica wants access to the tracked topology
    // as that program won't have access to this object.
    this.log('Topology Discover Event', () => changedPartsTopology);

    // Emit event indicating topology changed
    this.emit(topologyNotification);
  }

  public processDereferenceEvent(data: ILink): void {
    // When a dereference event happens, metadata is extended. On overlap most recently updated values are taken
    this.metadata[data.url] = { ...this.metadata[data.url], ...data.metadata };

    // Make shallow copies of data to emit to logger and event emitter
    const updatedTopology: ITopologyEventNotification = {
      type: 'dereference',
    };

    const changedPartsTopology: ITopologyChangeDereference = {
      dereferenced: data.url,
      metadata: this.metadata[data.url],
    };

    this.log('Topology Dereference Event', () => changedPartsTopology);

    // Emit event indicating topology changed
    this.emit(updatedTopology);
  }

  public emit(data: ITopologyEventNotification): void {
    this.statisticEvents.emit('data', data);
  }

  /**
   * Getter function for the tracked statistics. Use this where you want to use the topology
   * in other parts of Comunica. The objects returned by this function will be updated when a
   * topology event is emitted.
   * @returns Representation of tracked topolgy
   */
  public getTrackedStatistics(): ITopologyRepresentation {
    return {
      edgeList: this.edgeList,
      metadata: this.metadata,
    };
  }
}

export interface ITopologyChangeDiscover {
  edge: [string, string];
  metadataChild: Record<string, any>;
}

export interface ITopologyChangeDereference {
  dereferenced: string;
  metadata: Record<string, any>;
}

export interface ITopologyRepresentation {
  /**
   * Edgelist of topology
   */
  edgeList: Set<string>;
  /**
   * Metadata object: {url: metadata}
   */
  metadata: Record<string, Record<string, any>>;
}
