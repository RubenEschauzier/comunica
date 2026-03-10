import type {
  IActionOptimizeQueryOperation,
  IActorOptimizeQueryOperationOutput,
} from '@comunica/bus-optimize-query-operation';
import { ActorOptimizeQueryOperation } from '@comunica/bus-optimize-query-operation';
import { KeysInitQuery, KeysQueryOperation } from '@comunica/context-entries';
import type { IActorTest, TestResult } from '@comunica/core';
import { passTestVoid } from '@comunica/core';
import { Algebra, AlgebraFactory, algebraUtils } from '@comunica/utils-algebra';

export const HINT_SUBJECT = 'http://comunica-internal/hint';
export const HINT_PREDICATE = 'http://comunica-internal/optimizer';
export const HINT_OBJECT = 'None';

/**
 * A comunica Query Hint Join Order Fixed Optimize Query Operation Actor.
 * This actor looks for the query hint triple `comunica:hint comunica:optimizer "None" .`
 * in the query algebra, where the prefix `comunica:` resolves to `http://comunica-internal/`.
 * If found, it removes the hint triple from the algebra
 * and sets a context entry to indicate that join order should not be optimized.
 */
export class ActorOptimizeQueryOperationQueryHintJoinOrderFixed extends ActorOptimizeQueryOperation {
  public async test(_action: IActionOptimizeQueryOperation): Promise<TestResult<IActorTest>> {
    return passTestVoid();
  }

  public async run(action: IActionOptimizeQueryOperation): Promise<IActorOptimizeQueryOperationOutput> {
    const dataFactory = action.context.getSafe(KeysInitQuery.dataFactory);
    const algebraFactory = new AlgebraFactory(dataFactory);
    let hintFound = false;

    const operation = algebraUtils.mapOperation(action.operation, {
      [Algebra.Types.PATTERN]: {
        preVisitor: () => ({ continue: false }),
        transform: (pattern: Algebra.Pattern) => {
          if (this.isQueryHintJoinOrderFixed(pattern)) {
            hintFound = true;
            return algebraFactory.createNop();
          }
          return pattern;
        },
      },
    });

    const context = hintFound ?
      action.context.set(KeysQueryOperation.isJoinOrderFixed, true) :
      action.context;

    return { operation, context };
  }

  /**
   * Checks if a pattern is the query hint triple for fixing join order.
   * The hint triple has the form: `comunica:hint comunica:optimizer "None" .`
   * where the prefix `comunica:` resolves to `http://comunica-internal/`.
   */
  private isQueryHintJoinOrderFixed(pattern: Algebra.Pattern): boolean {
    return pattern.subject.termType === 'NamedNode' &&
      pattern.subject.value === HINT_SUBJECT &&
      pattern.predicate.termType === 'NamedNode' &&
      pattern.predicate.value === HINT_PREDICATE &&
      pattern.object.termType === 'Literal' &&
      pattern.object.value === HINT_OBJECT;
  }
}
