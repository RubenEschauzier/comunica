import type { IAction, IActorArgs, IActorOutput, Mediate, IActorTest } from '@comunica/core';
import { Actor } from '@comunica/core';
import type { IJoinEntryWithMetadata } from '@comunica/types';

/**
 * A comunica actor for rdf-join-entries-sort events.
 *
 * Actor types:
 * * Input:  IActionRdfJoinEntriesSort:      Join entries.
 * * Test:   IActorTest:                     Test result.
 * * Output: IActorRdfJoinEntriesSortOutput: The sorted join entries.
 *
 * @see IActionRdfJoinEntriesSort
 * @see IActorTest
 * @see IActorRdfJoinEntriesSortOutput
 */
export abstract class ActorRdfJoinEntriesSort<TS = undefined>
  extends Actor<IActionRdfJoinEntriesSort, IActorRdfJoinEntriesSortTest, IActorRdfJoinEntriesSortOutput, TS> {
  /**
   * @param args -
   *   \ @defaultNested {<default_bus> a <cc:components/Bus.jsonld#Bus>} bus
   *   \ @defaultNested {Sorting join entries failed: none of the configured actors were able to sort} busFailMessage
   */
  public constructor(args: IActorRdfJoinEntriesSortArgs<TS>) {
    super(args);
  }
}

export interface IActionRdfJoinEntriesSort extends IAction {
  /**
   * The array of streams to join.
   */
  entries: IJoinEntryWithMetadata[];
}

export interface IActorRdfJoinEntriesSortTest extends IActorTest {
  /**
   * A value between 0 and 1 (both inclusive) that defines the accuracy of this actor's sorting.
   * A higher value means a higher accuracy.
   */
  accuracy: number;
}

export interface IActorRdfJoinEntriesSortOutput extends IActorOutput {
  /**
   * The array of sorted streams.
   */
  entries: IJoinEntryWithMetadata[];
}

export type IActorRdfJoinEntriesSortArgs<TS = undefined> = IActorArgs<
IActionRdfJoinEntriesSort,
IActorRdfJoinEntriesSortTest,
IActorRdfJoinEntriesSortOutput,
TS
>;

export type MediatorRdfJoinEntriesSort = Mediate<
IActionRdfJoinEntriesSort,
IActorRdfJoinEntriesSortOutput
>;
