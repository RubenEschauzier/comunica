import { QuerySourceRdfJs } from '@comunica/actor-query-source-identify-rdfjs';
import type { IActorDereferenceRdfOutput, MediatorDereferenceRdf } from '@comunica/bus-dereference-rdf';
import type { MediatorQuerySourceIdentifyHypermedia } from '@comunica/bus-query-source-identify-hypermedia';
import type { IActorRdfMetadataOutput, MediatorRdfMetadata } from '@comunica/bus-rdf-metadata';
import type { MediatorRdfMetadataAccumulate } from '@comunica/bus-rdf-metadata-accumulate';
import type { MediatorRdfMetadataExtract } from '@comunica/bus-rdf-metadata-extract';
import type { MediatorRdfResolveHypermediaLinks } from '@comunica/bus-rdf-resolve-hypermedia-links';
import type { MediatorRdfResolveHypermediaLinksQueue } from '@comunica/bus-rdf-resolve-hypermedia-links-queue';
import { KeysCaches, KeysInitQuery, KeysQuerySourceIdentify } from '@comunica/context-entries';
import type {
  BindingsStream,
  ComunicaDataFactory,
  FragmentSelectorShape,
  IActionContext,
  IAggregatedStore,
  IQueryBindingsOptions,
  IQuerySource,
  MetadataBindings,
  ILink,
  ISourceState
} from '@comunica/types';
import type { BindingsFactory } from '@comunica/utils-bindings-factory';
import type * as RDF from '@rdfjs/types';
import type { AsyncIterator } from 'asynciterator';
import { TransformIterator } from 'asynciterator';
import { Readable } from 'readable-stream';
import type { Algebra } from 'sparqlalgebrajs';
import { Factory } from 'sparqlalgebrajs';
import { MediatedLinkedRdfSourcesAsyncRdfIterator } from './MediatedLinkedRdfSourcesAsyncRdfIterator';
import { StreamingStoreMetadata } from './StreamingStoreMetadata';
import CachePolicy = require('http-cache-semantics');
import { types } from 'sparqlalgebrajs/lib/algebra';
import { DataFactory } from 'rdf-data-factory';
import { ActionContext } from '@comunica/core';


// Temp placements
const DF = new DataFactory();
const algebraFactory = new Factory(DF);

export class QuerySourceHypermedia implements IQuerySource {
  public readonly referenceValue: string;
  public readonly firstUrl: string;
  public readonly forceSourceType?: string;
  public readonly aggregateStore: boolean;
  public readonly mediators: IMediatorArgs;
  public readonly logWarning: (warningMessage: string) => void;
  public readonly dataFactory: ComunicaDataFactory;
  public readonly bindingsFactory: BindingsFactory;

  private readonly maxIterators: number;

  public constructor(
    firstUrl: string,
    forceSourceType: string | undefined,
    maxIterators: number,
    aggregateStore: boolean,
    mediators: IMediatorArgs,
    logWarning: (warningMessage: string) => void,
    dataFactory: ComunicaDataFactory,
    bindingsFactory: BindingsFactory,
  ) {
    this.referenceValue = firstUrl;
    this.firstUrl = firstUrl;
    this.forceSourceType = forceSourceType;
    this.maxIterators = maxIterators;
    this.mediators = mediators;
    this.aggregateStore = aggregateStore;
    this.logWarning = logWarning;
    this.dataFactory = dataFactory;
    this.bindingsFactory = bindingsFactory;
  }

  public async getSelectorShape(context: IActionContext): Promise<FragmentSelectorShape> {
    const source = await this.getSourceCached({ url: this.firstUrl }, {}, context, this.getAggregateStore(context));
    return source.source.getSelectorShape(context);
  }

  public queryBindings(
    operation: Algebra.Operation,
    context: IActionContext,
    options?: IQueryBindingsOptions,
  ): BindingsStream {
    // Optimized match with aggregated store if enabled and started.
    const aggregatedStore: IAggregatedStore | undefined = this.getAggregateStore(context);
    if (aggregatedStore && operation.type === 'pattern' && aggregatedStore.started) {
      return new QuerySourceRdfJs(
        aggregatedStore,
        context.getSafe(KeysInitQuery.dataFactory),
        this.bindingsFactory,
      ).queryBindings(operation, context);
    }
    
    // Initialize the within query cache on first call
    if (context.getSafe(KeysCaches.withinQueryStoreCache).size === 0) {
      this.getSourceCached({ url: this.firstUrl }, {}, context, aggregatedStore)
        .catch(error => it.destroy(error));
    }

    const dataFactory: ComunicaDataFactory = context.getSafe(KeysInitQuery.dataFactory);
    const algebraFactory = new Factory(dataFactory);
    const it: MediatedLinkedRdfSourcesAsyncRdfIterator = new MediatedLinkedRdfSourcesAsyncRdfIterator(
      operation,
      options,
      context,
      this.forceSourceType,
      this.firstUrl,
      this.maxIterators,
      (link, handledDatasets) => this.getSourceCached(link, handledDatasets, context, aggregatedStore),
      aggregatedStore,
      this.mediators.mediatorMetadataAccumulate,
      this.mediators.mediatorRdfResolveHypermediaLinks,
      this.mediators.mediatorRdfResolveHypermediaLinksQueue,
      dataFactory,
      algebraFactory,
    );

    if (aggregatedStore) {
      aggregatedStore.started = true;

      // Kickstart this iterator when derived iterators are created from the aggregatedStore,
      // otherwise the traversal process will not start if this iterator is not the first one to be consumed.
      const listener = (): void => it.kickstart();
      aggregatedStore.addIteratorCreatedListener(listener);
      it.on('end', () => aggregatedStore.removeIteratorCreatedListener(listener));
    }

    return it;
  }

  public queryQuads(operation: Algebra.Operation, context: IActionContext): AsyncIterator<RDF.Quad> {
    return new TransformIterator(async() => {
      const source = await this.getSourceCached({ url: this.firstUrl }, {}, context, this.getAggregateStore(context));
      return source.source.queryQuads(operation, context);
    });
  }

  public async queryBoolean(operation: Algebra.Ask, context: IActionContext): Promise<boolean> {
    const source = await this.getSourceCached({ url: this.firstUrl }, {}, context, this.getAggregateStore(context));
    return await source.source.queryBoolean(operation, context);
  }

  public async queryVoid(operation: Algebra.Update, context: IActionContext): Promise<void> {
    const source = await this.getSourceCached({ url: this.firstUrl }, {}, context, this.getAggregateStore(context));
    return await source.source.queryVoid(operation, context);
  }

  /**
   * Resolve a source for the given URL.
   * @param link A source link.
   * @param handledDatasets A hash of dataset identifiers that have already been handled.
   * @param context The action context.
   * @param aggregatedStore An optional aggregated store.
   */
  public async getSource(
    link: ILink,
    handledDatasets: Record<string, boolean>,
    context: IActionContext,
    aggregatedStore: IAggregatedStore | undefined,
  ): Promise<ISourceState> {
    // Include context entries from link
    if (link.context) {
      context = context.merge(link.context);
    }

    // Get the RDF representation of the given document
    let url = link.url;
    let quads: RDF.Stream;
    let metadata: Record<string, any>;
    
    // Get http caches
    const storeCache = context.get(KeysCaches.storeCache);
    const policyCache = context.get(KeysCaches.policyCache);


    try {
      let policy: CachePolicy | undefined = undefined;

      if (storeCache && policyCache){
        policy = policyCache.get(link.url);
      }

      let dereferenceRdfOutput: IActorDereferenceRdfOutput = await this.mediators.mediatorDereferenceRdf
        .mediate({ context, url, validate: policy });

      // We can use cache here.
      if (dereferenceRdfOutput.isValidated){
        const cachedSource = storeCache!.get(link.url);
        if (!cachedSource){
          throw new Error("Tried to use cached entry that does not exist")
        }
        const emptyPattern = algebraFactory.createPattern(
          DF.variable('s'), DF.variable('p'), 
          DF.variable('o'), DF.variable('h'), 
        )
        const sourceQuads = cachedSource.source.queryQuads(emptyPattern, new ActionContext());
        const rdfMetadataOutput: IActorRdfMetadataOutput = await this.mediators.mediatorMetadata.mediate(
          { context, url, quads: sourceQuads, triples: dereferenceRdfOutput.metadata?.triples },
        );
        (await this.mediators.mediatorMetadataExtract.mediate({
          context,
          url,
          // The problem appears to be conflicting metadata keys here
          metadata: rdfMetadataOutput.metadata,
          headers: dereferenceRdfOutput.headers,
          requestTime: dereferenceRdfOutput.requestTime,
        }));
  
  
        aggregatedStore?.setBaseMetadata(<MetadataBindings> cachedSource.metadata, false);
        aggregatedStore?.containedSources.add(link.url);
        aggregatedStore?.import(sourceQuads).on('error', (err) => {
          console.log(`Error importing: ${err}`)
        });
    
        return cachedSource;
      }
      url = dereferenceRdfOutput.url;

      // Determine the metadata
      const rdfMetadataOutput: IActorRdfMetadataOutput = await this.mediators.mediatorMetadata.mediate(
        { context, url, quads: dereferenceRdfOutput.data, triples: dereferenceRdfOutput.metadata?.triples },
      );

      rdfMetadataOutput.data.on('error', () => {
        // Silence errors in the data stream,
        // as they will be emitted again in the metadata stream,
        // and will result in a promise rejection anyways.
        // If we don't do this, we end up with an unhandled error message
      });

      metadata = (await this.mediators.mediatorMetadataExtract.mediate({
        context,
        url,
        // The problem appears to be conflicting metadata keys here
        metadata: rdfMetadataOutput.metadata,
        headers: dereferenceRdfOutput.headers,
        requestTime: dereferenceRdfOutput.requestTime,
      })).metadata;
      quads = rdfMetadataOutput.data;

      // Optionally filter the resulting data
      if (link.transform) {
        quads = await link.transform(quads);
      }
    } catch (error: unknown) {
      // Make sure that dereference errors are only emitted once an actor really needs the read quads
      // This for example allows SPARQL endpoints that error on service description fetching to still be source-forcible
      quads = new Readable();
      quads.read = () => {
        setTimeout(() => quads.emit('error', error));
        return null;
      };
      ({ metadata } = await this.mediators.mediatorMetadataAccumulate.mediate({ context, mode: 'initialize' }));

      // Log as warning, because the quads above may not always be consumed (e.g. for SPARQL endpoints),
      // so the user would not be notified of something going wrong otherwise.
      this.logWarning(`Metadata extraction for ${url} failed: ${(<Error> error).message}`);
    }
    // Aggregate all discovered quads into a store.
    aggregatedStore?.setBaseMetadata(<MetadataBindings> metadata, false);
    aggregatedStore?.containedSources.add(link.url);
    aggregatedStore?.import(quads).on('error', (err) => {
      console.log(`Error importing: ${err}`)
    });
    

    // Determine the source
    const { source, dataset } = await this.mediators.mediatorQuerySourceIdentifyHypermedia.mediate({
      context,
      forceSourceType: link.url === this.firstUrl ? this.forceSourceType : undefined,
      handledDatasets,
      metadata,
      quads,
      url,
    });

    if (dataset) {
      // Mark the dataset as applied
      // This is needed to make sure that things like QPF search forms are only applied once,
      // and next page links are followed after that.
      handledDatasets[dataset] = true;
    }
    // If storeCache is available cache the constructed source
    storeCache?.set(link.url, { link, source, metadata: <MetadataBindings> metadata, handledDatasets });

    // In case we use local files, we add a policy to the policy cache to show we can always reuse it.
    // Local files will not pass through the http fetch actor and thus will not have a policy made for them.    
    if (url.startsWith('file://')){
      policyCache?.set(link.url, this.createAlwaysFreshPolicy())
    }
    return { link, source, metadata: <MetadataBindings> metadata, handledDatasets };
  }

  /**
   * Resolve a source for the given URL.
   * This will first try to retrieve the source from cache.
   * @param link A source ILink.
   * @param handledDatasets A hash of dataset identifiers that have already been handled.
   * @param context The action context.
   * @param aggregatedStore An optional aggregated store.
   */
  protected getSourceCached(
    link: ILink,
    handledDatasets: Record<string, boolean>,
    context: IActionContext,
    aggregatedStore: IAggregatedStore | undefined,
  ): Promise<ISourceState> {
    const withinQueryCache = context.getSafe(KeysCaches.withinQueryStoreCache);
    let source = withinQueryCache.get(link.url);
    if (source) {
      return source;
    }
    source = this.getSource(link, handledDatasets, context, aggregatedStore);
    if (link.url === this.firstUrl || aggregatedStore === undefined) {
      withinQueryCache.set(link.url, source);
    }
    return source;
  }

  public getAggregateStore(context: IActionContext): IAggregatedStore | undefined {
    let aggregatedStore: IAggregatedStore | undefined;
    if (this.aggregateStore) {
      const aggregatedStores: Map<string, IAggregatedStore> | undefined = context
        .get(KeysQuerySourceIdentify.hypermediaSourcesAggregatedStores);
      if (aggregatedStores) {
        aggregatedStore = aggregatedStores.get(this.firstUrl);
        if (!aggregatedStore) {
          aggregatedStore = new StreamingStoreMetadata(
            undefined,
            async(accumulatedMetadata, appendingMetadata) => <MetadataBindings>
              (await this.mediators.mediatorMetadataAccumulate.mediate({
                mode: 'append',
                accumulatedMetadata,
                appendingMetadata,
                context,
              })).metadata,
          );
          aggregatedStores.set(this.firstUrl, aggregatedStore);
        }
        return aggregatedStore;
      }
    }
  }

  private createAlwaysFreshPolicy(){
    return new CachePolicy(
      {
        method: "GET",
        headers: {}
      },
      {
        status: 200,
        headers: {
          "cache-control": "max-age=999999", // Keeps it fresh
        },
      }
    );
  }

  public toString(): string {
    return `QuerySourceHypermedia(${this.firstUrl})`;
  }
}

export interface IMediatorArgs {
  mediatorDereferenceRdf: MediatorDereferenceRdf;
  mediatorMetadata: MediatorRdfMetadata;
  mediatorMetadataExtract: MediatorRdfMetadataExtract;
  mediatorMetadataAccumulate: MediatorRdfMetadataAccumulate;
  mediatorQuerySourceIdentifyHypermedia: MediatorQuerySourceIdentifyHypermedia;
  mediatorRdfResolveHypermediaLinks: MediatorRdfResolveHypermediaLinks;
  mediatorRdfResolveHypermediaLinksQueue: MediatorRdfResolveHypermediaLinksQueue;
}
