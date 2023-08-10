import { BindingsFactory } from '@comunica/bindings-factory';
import type { MediatorMergeBindingFactory } from '@comunica/bus-merge-binding-factory';
import type { IActionRdfJoin, IActorRdfJoinOutputInner, IActorRdfJoinArgs } from '@comunica/bus-rdf-join';
import { ActorRdfJoin } from '@comunica/bus-rdf-join';
import type { IMediatorTypeJoinCoefficients } from '@comunica/mediatortype-join-coefficients';
import { MetadataValidationState } from '@comunica/metadata';
import { ArrayIterator } from 'asynciterator';

/**
 * A comunica None RDF Join Actor.
 */
export class ActorRdfJoinNone extends ActorRdfJoin {
  public readonly mediatorMergeHandlers: MediatorMergeBindingFactory;

  public constructor(args: IActorRdfJoinNoneArgs) {
    super(args, {
      logicalType: 'inner',
      physicalName: 'none',
      limitEntries: 0,
    });
  }

  public async test(action: IActionRdfJoin): Promise<IMediatorTypeJoinCoefficients> {
    // Allow joining of one or zero streams
    if (action.entries.length > 0) {
      throw new Error(`Actor ${this.name} can only join zero entries`);
    }
    return await this.getJoinCoefficients();
  }

  protected async getOutput(action: IActionRdfJoin): Promise<IActorRdfJoinOutputInner> {
    const BF = new BindingsFactory(
      (await this.mediatorMergeHandlers.mediate({ context: action.context })).mergeHandlers,
    );
    return {
      result: {
        bindingsStream: new ArrayIterator([ BF.bindings() ], { autoStart: false }),
        metadata: () => Promise.resolve({
          state: new MetadataValidationState(),
          cardinality: { type: 'exact', value: 1 },
          canContainUndefs: false,
          variables: [],
        }),
        type: 'bindings',
      },
    };
  }

  protected async getJoinCoefficients(): Promise<IMediatorTypeJoinCoefficients> {
    return {
      iterations: 0,
      persistedItems: 0,
      blockingItems: 0,
      requestTime: 0,
    };
  }
}

export interface IActorRdfJoinNoneArgs extends IActorRdfJoinArgs {
  /**
   * A mediator for creating binding context merge handlers
   */
  mediatorMergeHandlers: MediatorMergeBindingFactory;
}
