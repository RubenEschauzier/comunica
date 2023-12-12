import type { IMergeHandler } from '@comunica/bus-merge-binding-factory';
import type * as RDF from '@rdfjs/types';
import { Map } from 'immutable';
import { DataFactory } from 'rdf-data-factory';
import { Bindings } from './Bindings';

/**
 * A Bindings factory that provides Bindings backed by immutable.js.
 */
export class BindingsFactory implements RDF.BindingsFactory {
  private readonly dataFactory: RDF.DataFactory;
  private readonly contextMergeHandlers: Record<string, IMergeHandler<any>> | undefined;

  public constructor(contextMergeHandlers?: Record<string, IMergeHandler<any>>,
    dataFactory: RDF.DataFactory = new DataFactory()) {
    this.dataFactory = dataFactory;
    this.contextMergeHandlers = contextMergeHandlers;
  }

  public bindings(entries: [RDF.Variable, RDF.Term][] = []): Bindings {
    return new Bindings(this.dataFactory,
      Map(entries.map(([ key, value ]) => [ key.value, value ])),
      this.contextMergeHandlers);
  }

  public fromBindings(bindings: Bindings): Bindings {
    return this.bindings([ ...bindings ]);
  }

  public fromRecord(record: Record<string, RDF.Term>): Bindings {
    return this.bindings(Object.entries(record).map(([ key, value ]) => [ this.dataFactory.variable!(key), value ]));
  }
}
