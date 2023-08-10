import { BindingsFactory } from '@comunica/bindings-factory';
import type { MediatorHttp } from '@comunica/bus-http';
import type { MediatorMergeBindingFactory } from '@comunica/bus-merge-binding-factory';
import type { IActionRdfResolveHypermedia, IActorRdfResolveHypermediaOutput,
  IActorRdfResolveHypermediaTest, IActorRdfResolveHypermediaArgs } from '@comunica/bus-rdf-resolve-hypermedia';
import { ActorRdfResolveHypermedia } from '@comunica/bus-rdf-resolve-hypermedia';
import { RdfSourceSparql } from './RdfSourceSparql';

/**
 * A comunica SPARQL RDF Resolve Hypermedia Actor.
 */
export class ActorRdfResolveHypermediaSparql extends ActorRdfResolveHypermedia {
  public readonly mediatorHttp: MediatorHttp;
  public readonly mediatorMergeHandlers: MediatorMergeBindingFactory;

  public readonly checkUrlSuffix: boolean;
  public readonly forceHttpGet: boolean;
  public readonly cacheSize: number;

  public constructor(args: IActorRdfResolveHypermediaSparqlArgs) {
    super(args, 'sparql');
  }

  public async testMetadata(action: IActionRdfResolveHypermedia): Promise<IActorRdfResolveHypermediaTest> {
    if (!action.forceSourceType && !action.metadata.sparqlService &&
      !(this.checkUrlSuffix && action.url.endsWith('/sparql'))) {
      throw new Error(`Actor ${this.name} could not detect a SPARQL service description or URL ending on /sparql.`);
    }
    return { filterFactor: 1 };
  }

  public async run(action: IActionRdfResolveHypermedia): Promise<IActorRdfResolveHypermediaOutput> {
    // Create bindingsfactory with handlers
    const BF = new BindingsFactory(
      (await this.mediatorMergeHandlers.mediate({ context: action.context })).mergeHandlers,
    );
    this.logInfo(action.context, `Identified as sparql source: ${action.url}`);
    const source = new RdfSourceSparql(
      action.metadata.sparqlService || action.url,
      action.context,
      this.mediatorHttp,
      this.forceHttpGet,
      this.cacheSize,
      BF,
    );
    return { source };
  }
}

export interface IActorRdfResolveHypermediaSparqlArgs extends IActorRdfResolveHypermediaArgs {
  /**
   * The HTTP mediator
   */
  mediatorHttp: MediatorHttp;
  /**
   * If URLs ending with '/sparql' should also be considered SPARQL endpoints.
   * @default {true}
   */
  checkUrlSuffix: boolean;
  /**
   * If queries should be sent via HTTP GET instead of POST
   * @default {false}
   */
  forceHttpGet: boolean;
  /**
   * A mediator for creating binding context merge handlers
   */
  mediatorMergeHandlers: MediatorMergeBindingFactory;
  /**
   * The cache size for COUNT queries.
   * @range {integer}
   * @default {1024}
   */
  cacheSize?: number;
}
