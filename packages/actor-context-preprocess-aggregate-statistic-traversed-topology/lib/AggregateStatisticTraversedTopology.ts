// eslint-disable-next-line
import { EventEmitter } from 'events';
import type { ILink } from '@comunica/bus-rdf-resolve-hypermedia-links';
import type { IAggregateStatistic, IDiscoverEventData, IStatistic, 
    IStatisticDereferencedLinks, IStatisticDiscoveredLinks,
    ITopologyEventData} from '@comunica/types';

export class AggregateStatisticTraversedTopology implements IAggregateStatistic {
    query: string;

    // Emitter that indicates when an update to the aggregate statistic happened
    public statisticEvents: EventEmitter;

    public toAggregate: Record<string, IStatistic>;

    public edgeList: Set<string>;
    public metadata: Record<string, any>[];
    public urlToIndex: Record<string, number>;


    public constructor(
        query: string,
        discoverStatistic: IStatisticDiscoveredLinks,
        dereferenceStatistic: IStatisticDereferencedLinks
    ){
        this.query = query
        this.toAggregate = {
            discover: discoverStatistic, 
            dereference: dereferenceStatistic
        };
        this.attachListeners();

        this.statisticEvents = new EventEmitter();

    }

    public attachListeners(): boolean {
        this.toAggregate.discover.getEmitter().
            on('data', (data: IDiscoverEventData) => this.processDiscoverEvent(data));

        this.toAggregate.dereference.getEmitter().
            on('data', (data: ILink) => this.processDereferenceEvent(data));

        return true
    }


    public processDiscoverEvent(data: IDiscoverEventData){
        // TODO: Add edge to edgelist + overwrite existing metadata from discover events here

        // Emit new topology for other actors that want to use this info (e.g link prioritization)
        this.emitTopologyData('discover');
    }


    public processDereferenceEvent(data: ILink){
        // TODO: Add dereference metadata to existing metadata of link here

        // Emit new topology for other actors that want to use this info (e.g link prioritization)
        this.emitTopologyData('dereference');
    }


    public getEmitter(): EventEmitter{
        return this.statisticEvents;
    }

    public emitTopologyData(type: 'dereference' | 'discover'){
        const topologyData: ITopologyEventData = {
            type: type,
            edgeList: this.edgeList,
            metadata: this.metadata
        }
        this.getEmitter().emit('data', topologyData);
    }

    public getAggregateStatistic: () => any;

}


  