import { ActorRdfJoin, IActionRdfJoin, IActorRdfJoinArgs, IActorRdfJoinOutputInner, IActorRdfJoinTestSideData, MediatorRdfJoin } from '@comunica/bus-rdf-join';
import { MediatorRdfJoinEntriesSort } from '@comunica/bus-rdf-join-entries-sort';
import { KeysInitQuery } from '@comunica/context-entries';
import { IActorArgs, IActorTest, passTestWithSideData, TestResult } from '@comunica/core';
import { IMediatorTypeJoinCoefficients } from '@comunica/mediatortype-join-coefficients';
import { BindingsStream, ComunicaDataFactory, IActionContext, IJoinEntry, IJoinEntryWithMetadata } from '@comunica/types';
import { getSafeBindings } from '@comunica/utils-query-operation';
import { Factory } from 'sparqlalgebrajs';
import { EddieOperatorStream, JoinFunction } from './EddieOperatorStream';
import { EddieControllerStream, TimestampGenerator } from './EddieControllerStream';
import { MediatorHashBindings } from '@comunica/bus-hash-bindings';
import { Bindings } from '@comunica/utils-bindings-factory';
import { RouteFixedMinimalIndex } from './EddieRouters';
import * as RDF from '@rdfjs/types';
import { DataFactory } from 'rdf-data-factory';

/**
 * A comunica Inner Multi Stems RDF Join Actor.
 */
export class ActorRdfJoinMultiStems extends ActorRdfJoin<IActorRdfJoinMultiStemsTestSideData> {
  public readonly mediatorHashBindings: MediatorHashBindings;
  public readonly mediatorJoinEntriesSort: MediatorRdfJoinEntriesSort;
  // TEMP!
  public readonly mediatorJoin: MediatorRdfJoin;

  private readonly DF = new DataFactory();


  public constructor(args: IActorRdfJoinMultiStemsArgs) {
    super(args, {
      logicalType: 'inner',
      physicalName: 'multi-stems',
      limitEntries: 3,
      limitEntriesMin: true,
      canHandleUndefs: true,
      isLeaf: false,
    });
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
  
  protected async getOutput(
    action: IActionRdfJoin,
    sideData: IActorRdfJoinMultiStemsTestSideData,
  ): Promise<IActorRdfJoinOutputInner> {
    let { metadatas } = sideData;
    metadatas = [ ...metadatas ];

    const { hashFunction } = await this.mediatorHashBindings.mediate({ context: action.context });
    const timestampGenerator = new TimestampGenerator();
    const minimalIndexRouter = new RouteFixedMinimalIndex(action.entries.length);

    // By sorting the entries and selecting the minimal "not done" entry we route bindings
    // to the streams with smallest cardinality first.
    const sortedEntries: IJoinEntryWithMetadata[] = await this.sortJoinEntries(action.entries
      .map((entry, i) => ({ ...entry, metadata: metadatas[i] })), action.context);

    const entryJoinVariables: RDF.Variable[][][] = await this.getJoinVariables(sortedEntries);
    const stemOperators: EddieOperatorStream[] = [];
    
    for (const [i, entry] of sortedEntries.entries()) {
      stemOperators.push(
        new EddieOperatorStream(
          entry.output.bindingsStream,
          timestampGenerator,
          hashFunction,
          <JoinFunction> ActorRdfJoin.joinBindings,
          i,
          (await entry.output.metadata()).variables.map(x=>x.variable),
          entryJoinVariables[i]
        )
      )
    }

    const controllerStream = new EddieControllerStream(stemOperators, minimalIndexRouter);

    return {
      result: {
        type: 'bindings',
        bindingsStream: <BindingsStream><any>controllerStream,
        metadata: async() => await this.constructResultMetadata(
          action.entries,
          await ActorRdfJoin.getMetadatas(action.entries),
          action.context,
        ),
      }
    }
  }

  protected async getJoinCoefficients(
    action: IActionRdfJoin,
    sideData: IActorRdfJoinTestSideData,
  ): Promise<TestResult<IMediatorTypeJoinCoefficients, IActorRdfJoinMultiStemsTestSideData>> {
    let { metadatas } = sideData;
    metadatas = [ ...metadatas ];
    // Determine the two smallest streams by sorting (e.g. via cardinality)
    const sortedEntries = await this.sortJoinEntries(action.entries
      .map((entry, i) => ({ ...entry, metadata: metadatas[i] })), action.context);
    metadatas = sortedEntries.map(entry => entry.metadata);

    return passTestWithSideData({
      iterations: 0,
      persistedItems: 0,
      blockingItems: 0,
      requestTime: 0,
    }, { ...sideData, sortedEntries });
  }

  protected async getJoinVariables(entries:  IJoinEntryWithMetadata[]): Promise<RDF.Variable[][][]> {
    // For each entry, we determine the variables that overlap with other entries.
    // As some entries are joined on multiple variables, each join variable entry is an array.
    const entryJoinVariables: RDF.Variable[][][] = [];

    for (let i = 0; i < entries.length; i++) {
      const overlappingVariables: string[][] = [];
      for (let j = 0; j < entries.length; j++){
        if (i !== j){
          const variablesInner = (await entries[j].output.metadata()).variables.map(x=>x.variable.value);
          const variablesOuter = (await entries[i].output.metadata()).variables.map(x=>x.variable.value);
          const intersection = variablesOuter.filter(x => variablesInner.includes(x));
          // Ensure no duplicate entries are added
          if (intersection.length > 0 && !overlappingVariables.some(x => x.every(y => intersection.includes(y)))){
            overlappingVariables.push(intersection)
          }
        }
      }
      entryJoinVariables.push(Array.from(overlappingVariables).map(x => x.map(y=>this.DF.variable(y))));
    }
    return entryJoinVariables
  }
}


export interface IActorRdfJoinMultiStemsArgs extends IActorRdfJoinArgs<IActorRdfJoinMultiStemsTestSideData> {
  /**
   * The mediator for hashing bindings.
   */
  mediatorHashBindings: MediatorHashBindings;
  /**
   * The join entries sort mediator
   */
  mediatorJoinEntriesSort: MediatorRdfJoinEntriesSort;
  /**
   * A mediator for joining Bindings streams (TEMP)
   */
  mediatorJoin: MediatorRdfJoin;
  
}

export interface IActorRdfJoinMultiStemsTestSideData extends IActorRdfJoinTestSideData {
  sortedEntries: IJoinEntryWithMetadata[];
}
