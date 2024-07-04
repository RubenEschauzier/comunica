// eslint-disable-next-line
import { EventEmitter } from 'events';
import type { ILink } from '@comunica/bus-rdf-resolve-hypermedia-links';
import type { IDiscoverEventData, IStatistic, 
    IStatisticDereferencedLinks, IStatisticDiscoveredLinks} from '@comunica/types';

export class AggregateStatisticTraversedTopology implements IStatistic<ITopologyEventData> {
    // Emitter that indicates when an update to the aggregate statistic happened
    public statisticEvents: EventEmitter;
    public toAggregate: Record<string, IStatistic<IDiscoverEventData | ILink>>;

    // Tracked data
    public edgeList: Set<string>;
    public metadata: Record<string, any>[];
    public urlToIndex: Record<string, number>;


    public constructor(
        discoverStatistic: IStatisticDiscoveredLinks,
        dereferenceStatistic: IStatisticDereferencedLinks
    ){
        this.toAggregate = {
            discover: discoverStatistic, 
            dereference: dereferenceStatistic
        };

        this.attachListeners();
        this.statisticEvents = new EventEmitter();
    }

    public attachListeners(): boolean {
        (<IStatisticDiscoveredLinks> this.toAggregate.discover).
        addListener(this.processDiscoverEvent);

        (<IStatisticDereferencedLinks> this.toAggregate.dereference).
        addListener(this.processDereferenceEvent);

        return true;
    }

    public addListener(cb: (arg0: ITopologyEventData) => void): void {
        this.statisticEvents.addListener('data', cb);
    }

    public processDiscoverEvent(data: IDiscoverEventData){
        // TODO: Add edge to edgelist + overwrite existing metadata from discover events here

        const updatedTopology: ITopologyEventData = {
            type: 'dereference',
            edgeList: this.edgeList,
            metadata: this.metadata
        }

        // Emit new topology for other actors that want to use this info (e.g link prioritization)
        this.emit(updatedTopology);
    }


    public processDereferenceEvent(data: ILink){
        // TODO: Add dereference metadata to existing metadata of link here

        const updatedTopology: ITopologyEventData = {
            type: 'dereference',
            edgeList: this.edgeList,
            metadata: this.metadata
        }

        // Emit new topology for other actors that want to use this info (e.g link prioritization)
        this.emit(updatedTopology);
    }


    public emit(data:  ITopologyEventData){
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
