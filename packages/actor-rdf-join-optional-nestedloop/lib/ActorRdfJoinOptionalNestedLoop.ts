import type { IActionRdfJoin,
  IActorRdfJoinOutputInner,
  IActorRdfJoinArgs } from '@comunica/bus-rdf-join';
import {
  ActorRdfJoin,
} from '@comunica/bus-rdf-join';
import type { IMediatorTypeJoinCoefficients } from '@comunica/mediatortype-join-coefficients';
import type { Bindings, IMetadata } from '@comunica/types';
import { NestedLoopJoin } from 'asyncjoin';

/**
 * A comunica Optional Nested Loop RDF Join Actor.
 */
export class ActorRdfJoinOptionalNestedLoop extends ActorRdfJoin {
  public constructor(args: IActorRdfJoinArgs) {
    super(args, {
      logicalType: 'optional',
      physicalName: 'nested-loop',
      limitEntries: 2,
      canHandleUndefs: true,
    });
  }

  public async getOutput(action: IActionRdfJoin): Promise<IActorRdfJoinOutputInner> {
    const join = new NestedLoopJoin<Bindings, Bindings, Bindings>(
      action.entries[0].output.bindingsStream,
      action.entries[1].output.bindingsStream,
      <any> ActorRdfJoin.joinBindings,
      { optional: true, autoStart: false },
    );
    return {
      result: {
        type: 'bindings',
        bindingsStream: join,
        variables: ActorRdfJoin.joinVariables(action),
        metadata: async() => await this.constructResultMetadata(
          action.entries,
          await ActorRdfJoin.getMetadatas(action.entries),
          action.context,
          { canContainUndefs: true },
        ),
      },
    };
  }

  protected async getJoinCoefficients(
    action: IActionRdfJoin,
    metadatas: IMetadata[],
  ): Promise<IMediatorTypeJoinCoefficients> {
    const requestInitialTimes = ActorRdfJoin.getRequestInitialTimes(metadatas);
    const requestItemTimes = ActorRdfJoin.getRequestItemTimes(metadatas);
    return {
      iterations: metadatas[0].cardinality * metadatas[1].cardinality,
      persistedItems: 0,
      blockingItems: 0,
      requestTime: requestInitialTimes[0] + metadatas[0].cardinality * requestItemTimes[0] +
        requestInitialTimes[1] + metadatas[1].cardinality * requestItemTimes[1],
    };
  }
}