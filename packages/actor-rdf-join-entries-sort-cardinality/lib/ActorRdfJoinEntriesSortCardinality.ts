import type { IActionRdfJoinEntriesSort, IActorRdfJoinEntriesSortOutput } from '@comunica/bus-rdf-join-entries-sort';
import { ActorRdfJoinEntriesSort } from '@comunica/bus-rdf-join-entries-sort';
import type { IActorArgs, IActorTest, TestResult } from '@comunica/core';
import { passTest, passTestVoid } from '@comunica/core';
import { IMediatorTypeAccuracy } from '@comunica/mediatortype-accuracy';

/**
 * An actor that sorts join entries by increasing cardinality.
 */
export class ActorRdfJoinEntriesSortCardinality extends ActorRdfJoinEntriesSort {
  public constructor(
    args: IActorArgs<IActionRdfJoinEntriesSort, IMediatorTypeAccuracy, IActorRdfJoinEntriesSortOutput>,
  ) {
    super(args);
  }

  public async test(_action: IActionRdfJoinEntriesSort): Promise<TestResult<IMediatorTypeAccuracy>> {
    return passTest({accuracy: .5});
  }

  public async run(action: IActionRdfJoinEntriesSort): Promise<IActorRdfJoinEntriesSortOutput> {
    const entries = [ ...action.entries ]
      .sort((entryLeft, entryRight) => entryLeft.metadata.cardinality.value - entryRight.metadata.cardinality.value);
    return { entries };
  }
}
