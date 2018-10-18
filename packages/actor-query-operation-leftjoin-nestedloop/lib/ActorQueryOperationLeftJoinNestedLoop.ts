import {
  ActorQueryOperation, ActorQueryOperationTypedMediated, Bindings, IActorQueryOperationOutputBindings,
  IActorQueryOperationTypedMediatedArgs,
} from "@comunica/bus-query-operation";
import { ActorRdfJoin } from "@comunica/bus-rdf-join";
import { ActionContext, IActorTest } from "@comunica/core";
import { ClonedIterator } from "asynciterator";
import { Algebra } from "sparqlalgebrajs";
import { AsyncEvaluator, ExpressionError } from 'sparqlee';

/**
 * A comunica LeftJoin NestedLoop Query Operation Actor.
 */
export class ActorQueryOperationLeftJoinNestedLoop extends ActorQueryOperationTypedMediated<Algebra.LeftJoin> {

  constructor(args: IActorQueryOperationTypedMediatedArgs) {
    super(args, 'leftjoin');
  }

  public async testOperation(pattern: Algebra.LeftJoin, context: ActionContext): Promise<IActorTest> {
    return !pattern.expression;
  }

  public async runOperation(pattern: Algebra.LeftJoin, context: ActionContext)
    : Promise<IActorQueryOperationOutputBindings> {

    const leftRaw = await this.mediatorQueryOperation.mediate({ operation: pattern.left, context });
    const left = ActorQueryOperation.getSafeBindings(leftRaw);
    const rightRaw = await this.mediatorQueryOperation.mediate({ operation: pattern.right, context });
    const right = ActorQueryOperation.getSafeBindings(rightRaw);

    const leftJoinInner = (outerItem: Bindings, innerStream: ClonedIterator<Bindings>) => {
      const joinedStream = innerStream
        .transform({
          transform: async (innerItem: Bindings, nextInner: any) => {
            const joinedBindings = ActorRdfJoin.join(outerItem, innerItem);
            if (!joinedBindings) { nextInner(); return; }
            if (!pattern.expression) {
              joinedStream._push({ joinedBindings, result: true });
              nextInner();
              return;
            }
            try {
              const evaluator = new AsyncEvaluator(pattern.expression);
              const result = await evaluator.evaluateAsEBV(joinedBindings);
              joinedStream._push({ joinedBindings, result });
            } catch (err) {
              if (!this.isExpressionError(err)) {
                bindingsStream.emit('error', err);
              }
            }
            nextInner();
          },
        });
      return joinedStream;
    };

    const leftJoinOuter = (leftItem: Bindings, nextLeft: any) => {
      const innerStream = right.bindingsStream.clone(); // TODO: Why does this even work?
      const joinedStream = leftJoinInner(leftItem, innerStream);

      // TODO: Does this even work for large streams?
      // The full inner stream is kept in memory.
      joinedStream.on('end', () => nextLeft());
      joinedStream.on('data', async ({ joinedBindings, result }) => {
        if (result) {
          bindingsStream._push(joinedBindings);
        }
      });
    };

    const transform = leftJoinOuter;
    const bindingsStream = left.bindingsStream
      .transform<Bindings>({ optional: true, transform });

    const variables = ActorRdfJoin.joinVariables({ entries: [left, right] });
    const metadata = () => Promise.all([pattern.left, pattern.right].map((entry) => entry.metadata))
      .then((metadatas) => metadatas.reduce((acc, val) => acc * val.totalItems, 1))
      .catch(() => Infinity)
      .then((totalItems) => ({ totalItems }));

    return { type: 'bindings', bindingsStream, metadata, variables };
  }

  // TODO Duplication with e.g. Extend
  // Expression errors are errors intentionally thrown because of expression-data mismatches.
  // They are distinct from other errors in the sense that their behaviour is defined by
  // the SPARQL spec.
  // In this specific case, they should be ignored (while others obviously should not).
  // This function is separate so it can more easily be mocked in tests.
  public isExpressionError(error: Error): boolean {
    return error instanceof ExpressionError;
  }
}