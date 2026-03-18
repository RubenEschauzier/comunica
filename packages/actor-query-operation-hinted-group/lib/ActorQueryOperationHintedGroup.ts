import type { IActorQueryOperationTypedMediatedArgs } from '@comunica/bus-query-operation';
import {
  ActorQueryOperationTypedMediated,
} from '@comunica/bus-query-operation';
import type { MediatorRdfJoin } from '@comunica/bus-rdf-join';
import { ActorRdfJoin } from '@comunica/bus-rdf-join';
import { KeysInitQuery } from '@comunica/context-entries';
import type { IActorTest, TestResult } from '@comunica/core';
import { passTestVoid } from '@comunica/core';
import type { IQueryOperationResult, IActionContext, IJoinEntry, ComunicaDataFactory } from '@comunica/types';
import type { Algebra } from '@comunica/utils-algebra';
import { AlgebraFactory, TypesComunica } from '@comunica/utils-algebra';
import { MetadataValidationState } from '@comunica/utils-metadata';
import { getSafeBindings } from '@comunica/utils-query-operation';
import type * as RDF from '@rdfjs/types';
import { ArrayIterator } from 'asynciterator';
import { DataFactory } from 'rdf-data-factory';

/**
 * A comunica Hinted Group Query Operation Actor.
 * Handles hinted-group algebra operations, executing joins strictly in the order
 * defined by the input array. This preserves the query structure as authored,
 * bypassing heuristic join reordering.
 */
export class ActorQueryOperationHintedGroup extends ActorQueryOperationTypedMediated<Algebra.HintedGroup> {
  public readonly mediatorJoin: MediatorRdfJoin;

  public constructor(args: IActorQueryOperationHintedGroupArgs) {
    super(args, TypesComunica.HINTED_GROUP);
    this.mediatorJoin = args.mediatorJoin;
  }

  public async testOperation(
    _operation: Algebra.HintedGroup,
    _context: IActionContext,
  ): Promise<TestResult<IActorTest>> {
    return passTestVoid();
  }

  public async runOperation(
    operationOriginal: Algebra.HintedGroup,
    context: IActionContext,
  ): Promise<IQueryOperationResult> {
    const dataFactory: ComunicaDataFactory = context.getSafe(KeysInitQuery.dataFactory);
    const algebraFactory = new AlgebraFactory(dataFactory);
    const subOperations: Algebra.Operation[] = operationOriginal.input;

    // Execute all sub-operations (including nested hinted-groups, which will recurse)
    const entries: IJoinEntry[] = (await Promise.all(subOperations
      .map(async subOperation => ({
        output: await this.mediatorQueryOperation.mediate({ operation: subOperation, context }),
        operation: subOperation,
      }))))
      .map(({ output, operation }) => ({
        output: getSafeBindings(output),
        operation,
      }));

    // Return immediately if one of the join entries has cardinality zero
    if ((await Promise.all(entries.map(entry => entry.output.metadata())))
      .some(entry => (entry.cardinality.value === 0 && entry.cardinality.type === 'exact'))) {
      for (const entry of entries) {
        entry.output.bindingsStream.close();
      }
      return {
        bindingsStream: new ArrayIterator<RDF.Bindings>([], { autoStart: false }),
        metadata: async() => ({
          state: new MetadataValidationState(),
          cardinality: { type: 'exact', value: 0 },
          variables: ActorRdfJoin.joinVariables(new DataFactory(), await ActorRdfJoin.getMetadatas(entries)),
        }),
        type: 'bindings',
      };
    }

    // If only one entry, return directly
    if (entries.length === 1) {
      return entries[0].output;
    }

    // If exactly two entries, join directly
    if (entries.length === 2) {
      return this.mediatorJoin.mediate({ type: 'inner', entries, context });
    }

    // For 3+ entries: strict left-deep join in array order (no sorting/heuristics)
    // Join first two, then join result with each subsequent entry
    let currentEntry: IJoinEntry = {
      output: getSafeBindings(await this.mediatorJoin
        .mediate({ type: 'inner', entries: [ entries[0], entries[1] ], context })),
      operation: algebraFactory.createJoin([ entries[0].operation, entries[1].operation ], false),
    };

    for (let i = 2; i < entries.length; i++) {
      currentEntry = {
        output: getSafeBindings(await this.mediatorJoin
          .mediate({ type: 'inner', entries: [ currentEntry, entries[i] ], context })),
        operation: algebraFactory.createJoin([ currentEntry.operation, entries[i].operation ], false),
      };
    }

    return currentEntry.output;
  }
}

export interface IActorQueryOperationHintedGroupArgs extends IActorQueryOperationTypedMediatedArgs {
  /**
   * A mediator for joining Bindings streams
   */
  mediatorJoin: MediatorRdfJoin;
}
