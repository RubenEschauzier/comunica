// eslint-disable-next-line import/no-nodejs-modules
import { EventEmitter } from 'events';
import type { ILink } from '@comunica/bus-rdf-resolve-hypermedia-links';
import type { IDiscoverEventData, IStatistic, IStatisticDereferencedLinks, IStatisticDiscoveredLinks }
  from '@comunica/types';
import { StatisticsHolder } from '@comunica/actor-context-preprocess-set-defaults';
import { KeysTrackableStatistics } from '@comunica/context-entries';
import { link } from 'fs';

export class AggregateStatisticTraversedTopology implements IStatistic<ITopologyEventData> {
  // Emitter that indicates when an update to the aggregate statistic happened
  public statisticEvents: EventEmitter;
  public toAggregate: StatisticsHolder;

  // Tracked data
  public edgeList: Set<string>;
  public metadata: Record<string, Record<string, any>>;
  public urlToIndex: Record<string, number>;

  public constructor(
    discoverStatistic: IStatisticDiscoveredLinks,
    dereferenceStatistic: IStatisticDereferencedLinks,
  ) {
    this.toAggregate = new StatisticsHolder();
    this.toAggregate.set(KeysTrackableStatistics.discoveredLinks, discoverStatistic);
    this.toAggregate.set(KeysTrackableStatistics.dereferencedLinks, dereferenceStatistic);

    this.attachListeners();
    this.statisticEvents = new EventEmitter();
  }

  public attachListeners(): boolean {

    (<IStatisticDiscoveredLinks> this.toAggregate.get(KeysTrackableStatistics.discoveredLinks))
      .addListener(this.processDiscoverEvent.bind(this));

    (<IStatisticDereferencedLinks> this.toAggregate.get(KeysTrackableStatistics.dereferencedLinks))
      .addListener(this.processDereferenceEvent.bind(this));

    return true;
  }

  public addListener(cb: (arg0: ITopologyEventData) => void): void {
    this.statisticEvents.addListener('data', cb);
  }

  public processDiscoverEvent(data: IDiscoverEventData): void {
    this.edgeList.add(JSON.stringify(data.edge));

    // Aggregate array of metadata entries for single node into one metadata entry with
    // array values
    const aggregatedMetadata: Record<any, any> = {}
    data.metadataDiscoveredNode.map((metadata: Record<any, any>) => {
        Object.keys(metadata).map( key => { aggregatedMetadata[key] = aggregatedMetadata[key] ? 
            [...aggregatedMetadata[key], metadata[key]] : [metadata[key]];
        })
    });
    
    // Merge exising child metadata with incoming child metadata. On key overlap this will take values from aggregatedMetadata 
    // as that is the most recently updated
    this.metadata[data.edge[1]] = {...this.metadata[data.edge[1]], ...aggregatedMetadata}

    const updatedTopology: ITopologyEventData = {
      type: 'dereference',
      edgeList: this.edgeList,
      metadata: this.metadata,
    };

    // Emit new topology for other actors that want to use this info (e.g link prioritization)
    this.emit(updatedTopology);
  }

  public processDereferenceEvent(data: ILink): void {
    // When a dereference event happens, metadata is extended. 
    this.metadata[data.url] = { ...this.metadata[data.url], ...data.metadata }
    const updatedTopology: ITopologyEventData = {
      type: 'dereference',
      edgeList: this.edgeList,
      metadata: this.metadata,
    };

    // Emit new topology for other actors that want to use this info (e.g link prioritization)
    this.emit(updatedTopology);
  }

  public emit(data: ITopologyEventData): void {
    this.statisticEvents.emit('data', data);
  }
}

/**
 * Interface describing what data will be emitted when the topology is updated
 * Will emit all data so far instead of only the update that happened to the topology
 */
export interface ITopologyEventData {
  /**
   * What type of update happened to the topology
   */
  type: 'dereference' | 'discover';
  /**
   * The edge list discovered during query execution
   */
  edgeList: Set<string>;
  /**
   * The metadata for each node
   */
  metadata: Record<string, Record<string, any>>;
}
