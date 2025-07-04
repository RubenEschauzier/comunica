import { ActorAbstractPath } from '@comunica/actor-abstract-path';
import type { MediatorMergeBindingsContext } from '@comunica/bus-merge-bindings-context';
import type { IActorQueryOperationTypedMediatedArgs } from '@comunica/bus-query-operation';
import { KeysInitQuery } from '@comunica/context-entries';
import type {
  IQueryOperationResultBindings,
  Bindings,
  IQueryOperationResult,
  IActionContext,
  ComunicaDataFactory,
} from '@comunica/types';
import { BindingsFactory } from '@comunica/utils-bindings-factory';
import { getSafeBindings } from '@comunica/utils-query-operation';
import { BufferedIterator, MultiTransformIterator, TransformIterator } from 'asynciterator';
import { Algebra, Factory } from 'sparqlalgebrajs';

/**
 * A comunica Path OneOrMore Query Operation Actor.
 */
export class ActorQueryOperationPathOneOrMore extends ActorAbstractPath {
  public readonly mediatorMergeBindingsContext: MediatorMergeBindingsContext;

  public constructor(args: IActorQueryOperationPathOneOrMoreArgs) {
    super(args, Algebra.types.ONE_OR_MORE_PATH);
  }

  public async runOperation(operation: Algebra.Path, context: IActionContext): Promise<IQueryOperationResult> {
    const dataFactory: ComunicaDataFactory = context.getSafe(KeysInitQuery.dataFactory);
    const algebraFactory = new Factory(dataFactory);
    const bindingsFactory = await BindingsFactory.create(
      this.mediatorMergeBindingsContext,
      context,
      dataFactory,
    );

    const distinct = await this.isPathArbitraryLengthDistinct(algebraFactory, context, operation);
    if (distinct.operation) {
      return distinct.operation;
    }

    context = distinct.context;

    const predicate = <Algebra.OneOrMorePath> operation.predicate;

    if (operation.subject.termType !== 'Variable' && operation.object.termType === 'Variable') {
      const objectVar = operation.object;
      const starEval = await this.getObjectsPredicateStarEval(
        operation.subject,
        predicate.path,
        objectVar,
        operation.graph,
        context,
        false,
        algebraFactory,
        bindingsFactory,
      );
      const variables = (operation.graph.termType === 'Variable' ? [ objectVar, operation.graph ] : [ objectVar ])
        .map(variable => ({ variable, canBeUndef: false }));
      return {
        type: 'bindings',
        bindingsStream: starEval.bindingsStream,
        metadata: async() => ({ ...await starEval.metadata(), variables }),
      };
    }
    if (operation.subject.termType === 'Variable' && operation.object.termType === 'Variable') {
      // Get all the results of subjects with same predicate, but once, then fill in first variable for those
      const single = algebraFactory.createDistinct(
        algebraFactory.createPath(operation.subject, operation.predicate.path, operation.object, operation.graph),
      );
      const results = getSafeBindings(
        await this.mediatorQueryOperation.mediate({ context, operation: single }),
      );
      const subjectVar = operation.subject;
      const objectVar = operation.object;

      const termHashes = {};

      const bindingsStream: MultiTransformIterator<Bindings, Bindings> = new MultiTransformIterator(
        results.bindingsStream,
        {
          multiTransform: (bindings: Bindings) => {
            const subject = bindings.get(subjectVar);
            const object = bindings.get(objectVar);
            const graph = operation.graph.termType === 'Variable' ? bindings.get(operation.graph) : operation.graph;
            return new TransformIterator<Bindings>(
              async() => {
                const it = new BufferedIterator<Bindings>();
                await this.getSubjectAndObjectBindingsPredicateStar(
                  subjectVar,
                  objectVar,
                  subject!,
                  object!,
                  predicate.path,
                  graph!,
                  context,
                  termHashes,
                  {},
                  it,
                  { count: 0 },
                  algebraFactory,
                  bindingsFactory,
                );
                return it.map<Bindings>((item) => {
                  if (operation.graph.termType === 'Variable') {
                    item = item.set(operation.graph, graph!);
                  }
                  return item;
                });
              },
              { autoStart: false, maxBufferSize: 128 },
            );
          },
          autoStart: false,
        },
      );
      const variables = (operation.graph.termType === 'Variable' ?
          [ subjectVar, objectVar, operation.graph ] :
          [ subjectVar, objectVar ])
        .map(variable => ({ variable, canBeUndef: false }));
      return {
        type: 'bindings',
        bindingsStream,
        metadata: async() => ({ ...await results.metadata(), variables }),
      };
    }
    if (operation.subject.termType === 'Variable' && operation.object.termType !== 'Variable') {
      return <Promise<IQueryOperationResultBindings>> this.mediatorQueryOperation.mediate({
        context,
        operation: algebraFactory.createPath(
          operation.object,
          algebraFactory.createOneOrMorePath(algebraFactory.createInv(predicate.path)),
          operation.subject,
          operation.graph,
        ),
      });
    }
    // If (!sVar && !oVar)
    const variable = this.generateVariable(dataFactory);
    const results = getSafeBindings(await this.mediatorQueryOperation.mediate({
      context,
      operation: algebraFactory.createPath(operation.subject, predicate, variable, operation.graph),
    }));
    const bindingsStream = results.bindingsStream.map<Bindings>((item) => {
      if (operation.object.equals(item.get(variable))) {
        return operation.graph.termType === 'Variable' ?
          bindingsFactory.bindings([[ operation.graph, item.get(operation.graph)! ]]) :
          bindingsFactory.bindings();
      }
      return null;
    });
    return {
      type: 'bindings',
      bindingsStream,
      metadata: async() => ({
        ...await results.metadata(),
        variables: (operation.graph.termType === 'Variable' ? [ operation.graph ] : [])
          .map(variable => ({ variable, canBeUndef: false })),
      }),
    };
  }
}

export interface IActorQueryOperationPathOneOrMoreArgs extends IActorQueryOperationTypedMediatedArgs {
  /**
   * A mediator for creating binding context merge handlers
   */
  mediatorMergeBindingsContext: MediatorMergeBindingsContext;
}
