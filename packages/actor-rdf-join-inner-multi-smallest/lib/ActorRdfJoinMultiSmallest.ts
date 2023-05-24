import {
  ActorQueryOperation,
} from '@comunica/bus-query-operation';
import type {
  IActionRdfJoin,
  IActorRdfJoinOutputInner,
  IActorRdfJoinArgs,
  MediatorRdfJoin,
} from '@comunica/bus-rdf-join';
import { ActorRdfJoin } from '@comunica/bus-rdf-join';
import type { MediatorRdfJoinEntriesSort } from '@comunica/bus-rdf-join-entries-sort';
import type { IMediatorTypeJoinCoefficients } from '@comunica/mediatortype-join-coefficients';
import type { MetadataBindings, IJoinEntry, IActionContext, IJoinEntryWithMetadata } from '@comunica/types';
import arrayifyStream from 'arrayify-stream';
import { Factory } from 'sparqlalgebrajs';

/**
 * A Multi Smallest RDF Join Actor.
 * It accepts 3 or more streams, joins the smallest two, and joins the result with the remaining streams.
 */
export class ActorRdfJoinMultiSmallest extends ActorRdfJoin {
  public readonly mediatorJoinEntriesSort: MediatorRdfJoinEntriesSort;
  public readonly mediatorJoin: MediatorRdfJoin;

  public static readonly FACTORY = new Factory();

  public constructor(args: IActorRdfJoinMultiSmallestArgs) {
    super(args, {
      logicalType: 'inner',
      physicalName: 'multi-smallest',
      limitEntries: 3,
      limitEntriesMin: true,
    });
  }
  /**
   * Returns true if the array of entries contains entries that have common variables
   * @param entries Array of entries with metadata to check for common variables
   * @returns 
   */
  // public hasCommonVariables(entries: IJoinEntryWithMetadata[]){
  //   const variableOccurrences: Record<string, number> = {};
  //   for (const entry of entries) {
  //     for (const variable of entry.metadata.variables) {
  //       let counter = variableOccurrences[variable.value];
  //       if (!counter) {
  //         counter = 0;
  //       }
  //       variableOccurrences[variable.value] = ++counter;
  //     }
  //   }

  //   // Determine variables that occur in at least two join entries
  //   const multiOccurrenceVariables: string[] = [];
  //   for (const [ variable, count ] of Object.entries(variableOccurrences)) {
  //     if (count >= 2) {
  //       multiOccurrenceVariables.push(variable);
  //     }
  //   }
  //   return multiOccurrenceVariables.length == 0
  // }

  /**
   * Finds join indexes of lowest cardinality result sets, with priority on result sets that have common variables
   * @param entries A sorted array of entries, sorted on cardinality
   */
  public getJoinIndexes(entries: IJoinEntryWithMetadata[]){
    // Iterate over all combinations of join indexes, return the first combination that does not lead to a cartesian product
    for (let i = 0; i<entries.length; i++){
      for (let j = i+1; j<entries.length; j++){
        if (this.hasCommonVariables(entries[i], entries[j])){
          return [i,j];
        }
      }
    }
    // If all result sets are disjoint we just want the sets with lowest cardinality
    return [0,1];
  }
  /**
   * Finds join indexes of lowest product of cardinalities of result sets, with priority on result sets that have common variables
   * @param entries A sorted array of entries, sorted on cardinality
   */
  public getFirstValidJoinWithMinimalCardinalityProduct(entries: IJoinEntryWithMetadata[]){
    const cardinalityProducts: Map<number, number[]> = new Map<number, number[]>();
    // Get cardinality products
    for (let i = 0; i<entries.length; i++){
      for (let j = 0; j<entries.length; j++){
        if (i != j){
          cardinalityProducts.set(entries[i].metadata.cardinality.value*entries[j].metadata.cardinality.value, [i,j]);
        }
      }
    }
    // Sort cardinality products
    let keys: number[] = [...cardinalityProducts.keys()];
    keys.sort((a, b)=> a-b);
    // Get sorted entry indexes
    const sortedCardinalityProductsJoinIndexes = keys.map(x=>cardinalityProducts.get(x)!);
    // Iterate over sorted entry indexes and select first index that is not cartesian join
    for (const joinIndexes of sortedCardinalityProductsJoinIndexes){
      if (this.hasCommonVariables(entries[joinIndexes[0]], entries[joinIndexes[1]])){
        return joinIndexes;
      }
    }
  }

  /**
   * Function to check for common variables
   * @param entry1 
   * @param entry2 
   * @returns 
   */
  public hasCommonVariables(entry1: IJoinEntryWithMetadata, entry2: IJoinEntryWithMetadata): boolean{
    return entry1.metadata.variables.some(v => entry2.metadata.variables.includes(v));
  }

  /**
   * Order the given join entries using the join-entries-sort bus.
   * @param {IJoinEntryWithMetadata[]} entries An array of join entries.
   * @param context The action context.
   * @return {IJoinEntryWithMetadata[]} The sorted join entries.
   */
  public async sortJoinEntries(
    entries: IJoinEntryWithMetadata[],
    context: IActionContext,
  ): Promise<IJoinEntryWithMetadata[]> {
    return (await this.mediatorJoinEntriesSort.mediate({ entries, context })).entries;
  }

  protected async getOutput(action: IActionRdfJoin): Promise<IActorRdfJoinOutputInner> {
    // Determine the two smallest streams by sorting (e.g. via cardinality)
    const entries: IJoinEntry[] = await this.sortJoinEntries(
      await ActorRdfJoin.getEntriesWithMetadatas([ ...action.entries ]),
      action.context,
    );

    const entriesMetaData = await ActorRdfJoin.getEntriesWithMetadatas(entries);
    const bestJoinIndexes = this.getJoinIndexes(entriesMetaData);

    const smallestEntry1 = entries[bestJoinIndexes[0]];
    const smallestEntry2 = entries[bestJoinIndexes[1]];

    // TEST CODE:
    if (bestJoinIndexes[0] > bestJoinIndexes[1]){
      throw new Error("Invalid combination of join indexes, first element larger than second")
    }
    // END TEST CODE
    
    entries.splice(bestJoinIndexes[1], 1);
    entries.splice(bestJoinIndexes[0], 1);

    // Join the two selected streams, and then join the result with the remaining streams
    const firstEntry: IJoinEntry = {
      output: ActorQueryOperation.getSafeBindings(await this.mediatorJoin
        .mediate({ type: action.type, entries: [ smallestEntry1, smallestEntry2 ], context: action.context })),
      operation: ActorRdfJoinMultiSmallest.FACTORY
        .createJoin([ smallestEntry1.operation, smallestEntry2.operation ], false),
    };

    entries.push(firstEntry);
    return {
      result: await this.mediatorJoin.mediate({
        type: action.type,
        entries,
        context: action.context,
      }),
    };
  }

  protected async getJoinCoefficients(
    action: IActionRdfJoin,
    metadatas: MetadataBindings[],
  ): Promise<IMediatorTypeJoinCoefficients> {
    metadatas = [ ...metadatas ];
    // Determine the two smallest streams by sorting (e.g. via cardinality)
    const entriesWithMetadata = await this.sortJoinEntries(action.entries
      .map((entry, i) => ({ ...entry, metadata: metadatas[i] })), action.context);
    metadatas = entriesWithMetadata.map(entry => entry.metadata);
    const requestInitialTimes = ActorRdfJoin.getRequestInitialTimes(metadatas);
    const requestItemTimes = ActorRdfJoin.getRequestItemTimes(metadatas);

    return {
      iterations: metadatas[0].cardinality.value * metadatas[1].cardinality.value *
        metadatas.slice(2).reduce((acc, metadata) => acc * metadata.cardinality.value, 1),
      persistedItems: 0,
      blockingItems: 0,
      requestTime: requestInitialTimes[0] + metadatas[0].cardinality.value * requestItemTimes[0] +
        requestInitialTimes[1] + metadatas[1].cardinality.value * requestItemTimes[1] +
        metadatas.slice(2).reduce((sum, metadata, i) => sum + requestInitialTimes.slice(2)[i] +
          metadata.cardinality.value * requestItemTimes.slice(2)[i], 0),
    };
  }
}

export interface IActorRdfJoinMultiSmallestArgs extends IActorRdfJoinArgs {
  /**
   * The join entries sort mediator
   */
  mediatorJoinEntriesSort: MediatorRdfJoinEntriesSort;
  /**
   * A mediator for joining Bindings streams
   */
  mediatorJoin: MediatorRdfJoin;
}
