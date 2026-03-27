import type { IActorQueryOperationTypedMediatedArgs } from '@comunica/bus-query-operation';
import { ActorQueryOperationTypedMediated } from '@comunica/bus-query-operation';
import type { MediatorRdfJoin } from '@comunica/bus-rdf-join';
import { ActorRdfJoin } from '@comunica/bus-rdf-join';
import { KeysInitQuery } from '@comunica/context-entries';
import type { IActorTest, TestResult } from '@comunica/core';
import { passTestVoid } from '@comunica/core';
import type { 
  IQueryOperationResult, 
  IActionContext, 
  IJoinEntry, 
  ComunicaDataFactory,
  IPhysicalQueryPlanLogger 
} from '@comunica/types';
import type { Algebra } from '@comunica/utils-algebra';
import { AlgebraFactory, TypesComunica } from '@comunica/utils-algebra';
import { MetadataValidationState } from '@comunica/utils-metadata';
import { getSafeBindings } from '@comunica/utils-query-operation';
import type * as RDF from '@rdfjs/types';
import { ArrayIterator } from 'asynciterator';
import { DataFactory } from 'rdf-data-factory';

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

    // Establish physical query plan logging context
    const action = { operation: operationOriginal, context };
    let subContext = context;
    let parentPhysicalQueryPlanNode: any;

    if (context.has(KeysInitQuery.physicalQueryPlanLogger)) {
      parentPhysicalQueryPlanNode = context.get(KeysInitQuery.physicalQueryPlanNode);
      subContext = context.set(KeysInitQuery.physicalQueryPlanNode, action);
    }

    const physicalQueryPlanLogger: IPhysicalQueryPlanLogger | undefined = subContext.get(
      KeysInitQuery.physicalQueryPlanLogger,
    );

    if (physicalQueryPlanLogger) {
      physicalQueryPlanLogger.stashChildren(
        parentPhysicalQueryPlanNode,
        (node: any) => node.logicalOperator && node.logicalOperator.startsWith('join'),
      );
      physicalQueryPlanLogger.logOperation(
        'join', // Groups act as logical joins
        'hinted-group',
        action,
        parentPhysicalQueryPlanNode,
        this.name,
        {},
      );
    }

    // Execute sub-operations using the updated context
    const entries: IJoinEntry[] = (await Promise.all(subOperations
      .map(async subOperation => ({
        output: await this.mediatorQueryOperation.mediate({ operation: subOperation, context: subContext }),
        operation: subOperation,
      }))))
      .map(({ output, operation }) => ({
        output: getSafeBindings(output),
        operation,
      }));

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

    if (entries.length === 1) {
      return entries[0].output;
    }

    if (entries.length === 2) {
      return this.mediatorJoin.mediate({ type: 'inner', entries, context: subContext });
    }

    let currentEntry: IJoinEntry = {
      output: getSafeBindings(await this.mediatorJoin
        .mediate({ type: 'inner', entries: [ entries[0], entries[1] ], context: subContext })),
      operation: algebraFactory.createJoin([ entries[0].operation, entries[1].operation ], false),
    };

    for (let i = 2; i < entries.length; i++) {
      currentEntry = {
        output: getSafeBindings(await this.mediatorJoin
          .mediate({ type: 'inner', entries: [ currentEntry, entries[i] ], context: subContext })),
        operation: algebraFactory.createJoin([ currentEntry.operation, entries[i].operation ], false),
      };
    }

    return currentEntry.output;
  }
}

export interface IActorQueryOperationHintedGroupArgs extends IActorQueryOperationTypedMediatedArgs {
  mediatorJoin: MediatorRdfJoin;
}