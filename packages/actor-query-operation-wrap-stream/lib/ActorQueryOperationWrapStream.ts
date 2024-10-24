import type { MediatorIteratorTransform } from '@comunica/bus-iterator-transform';
import type {
  IActionQueryOperation,
  IActorQueryOperationArgs,
  MediatorQueryOperation,
} from '@comunica/bus-query-operation';
import { ActorQueryOperation } from '@comunica/bus-query-operation';
import { failTest, passTest } from '@comunica/core';
import type { TestResult, IActorTest } from '@comunica/core';
import type { IQueryOperationResult, MetadataBindings, MetadataQuads } from '@comunica/types';
import { KEY_CONTEXT_WRAPPED_QUERY_OPERATION, setContextWrapped } from '@comunica/utils-query-operation';
import type * as RDF from '@rdfjs/types';
import type { AsyncIterator } from 'asynciterator';

/**
 * A comunica Wrap Stream Query Operation Actor.
 */
export class ActorQueryOperationWrapStream extends ActorQueryOperation {
  public readonly mediatorIteratorTransform: MediatorIteratorTransform;
  public readonly mediatorQueryOperation: MediatorQueryOperation;

  public constructor(args: IActorQueryOperationWrapStreamArgs) {
    super(args);
  }

  public async test(action: IActionQueryOperation): Promise<TestResult<IActorTest>> {
    if (action.context.get(KEY_CONTEXT_WRAPPED_QUERY_OPERATION) === action.operation) {
      return failTest('Unable to wrap query source multiple times');
    }
    // Ensure this is always run if not already wrapped
    return passTest({ httpRequests: Number.NEGATIVE_INFINITY });
  }

  public async run(action: IActionQueryOperation): Promise<IQueryOperationResult> {
    // Prevent infinite recursion. In consequent query operation calls this key should be set to false
    // To allow the operation to wrap ALL query operation runs
    action.context = setContextWrapped(action.operation, action.context);
    const output: IQueryOperationResult = await this.mediatorQueryOperation.mediate(action);
    switch (output.type) {
      case 'bindings': {
        const bindingIteratorTransformed = await this.mediatorIteratorTransform.mediate(
          {
            type: 'bindings',
            operation: action.operation.type,
            stream: output.bindingsStream,
            metadata: output.metadata,
            originalAction: action,
            context: action.context,
          },
        );
        output.bindingsStream =
          <AsyncIterator<RDF.Bindings>> bindingIteratorTransformed.stream;
        output.metadata =
          <() => Promise<MetadataBindings>> bindingIteratorTransformed.metadata;
        break;
      }
      case 'quads': {
        const iteratorTransformed = await this.mediatorIteratorTransform.mediate(
          {
            type: 'quads',
            operation: action.operation.type,
            stream: output.quadStream,
            metadata: output.metadata,
            context: action.context,
            originalAction: action,
          },
        );
        output.quadStream = <AsyncIterator<RDF.Quad>> iteratorTransformed.stream;
        output.metadata = <() => Promise<MetadataQuads>> iteratorTransformed.metadata;
        break;
      }
      default: {
        break;
      }
    }
    return output;
  }
}

export interface IActorQueryOperationWrapStreamArgs extends IActorQueryOperationArgs {
  /**
   * Mediator that runs all transforms defined by user over the output stream of the query operation
   */
  mediatorIteratorTransform: MediatorIteratorTransform;
  /**
   * Mediator that runs the next query operation
   */
  mediatorQueryOperation: MediatorQueryOperation;
}
