import { ActorAbstractPath } from '@comunica/actor-abstract-path';
import type { IActorQueryOperationTypedMediatedArgs } from '@comunica/bus-query-operation';
import {
  ActorQueryOperation,
} from '@comunica/bus-query-operation';
import type { IActionContext, IQueryOperationResult } from '@comunica/types';
import { Algebra } from 'sparqlalgebrajs';

/**
 * A comunica Path Nps Query Operation Actor.
 */
export class ActorQueryOperationPathNps extends ActorAbstractPath {
  public constructor(args: IActorQueryOperationTypedMediatedArgs) {
    super(args, Algebra.types.NPS);
  }

  public async runOperation(operation: Algebra.Path, context: IActionContext): Promise<IQueryOperationResult> {
    const predicate = <Algebra.Nps> operation.predicate;
    const blank = this.generateVariable(operation);

    const pattern = Object.assign(ActorAbstractPath.FACTORY
      .createPattern(operation.subject, blank, operation.object, operation.graph), { metadata: predicate.metadata });
    const output = ActorQueryOperation.getSafeBindings(
      await this.mediatorQueryOperation.mediate({ operation: pattern, context }),
    );

    // Remove the generated blank nodes from the bindings
    const bindingsStream = output.bindingsStream
      .map(bindings => predicate.iris.some(iri => iri.equals(bindings.get(blank))) ? null : bindings.delete(blank));

    return {
      type: 'bindings',
      bindingsStream,
      metadata: output.metadata,
    };
  }
}
