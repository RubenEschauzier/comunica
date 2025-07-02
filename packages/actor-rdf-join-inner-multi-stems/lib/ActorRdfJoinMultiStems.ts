import type { MediatorHashBindings } from '@comunica/bus-hash-bindings';
import type { IActionRdfJoin, IActorRdfJoinArgs, IActorRdfJoinOutputInner, IActorRdfJoinTestSideData, MediatorRdfJoin } from '@comunica/bus-rdf-join';
import { ActorRdfJoin } from '@comunica/bus-rdf-join';
import type { MediatorRdfJoinEntriesSort } from '@comunica/bus-rdf-join-entries-sort';
import type { TestResult } from '@comunica/core';
import { passTestWithSideData } from '@comunica/core';
import type { IMediatorTypeJoinCoefficients } from '@comunica/mediatortype-join-coefficients';
import type { BindingsStream, ComunicaDataFactory, IActionContext, IJoinEntry, IJoinEntryWithMetadata } from '@comunica/types';
import type * as RDF from '@rdfjs/types';
import { DataFactory } from 'rdf-data-factory';
import { EddieControllerStream, TimestampGenerator } from './EddieControllerStream';
import type { JoinFunction } from './EddieOperatorStream';
import { EddieOperatorStream } from './EddieOperatorStream';
import { RouteFixedMinimalIndex } from './EddieRouters';
import { Factory } from 'sparqlalgebrajs';
import { KeysInitQuery } from '@comunica/context-entries';


/**
 * A comunica Inner Multi Stems RDF Join Actor.
 */
export class ActorRdfJoinMultiStems extends ActorRdfJoin<IActorRdfJoinMultiStemsTestSideData> {
  public readonly mediatorHashBindings: MediatorHashBindings;
  public readonly mediatorJoinEntriesSort: MediatorRdfJoinEntriesSort;
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
    
    const connectedComponents = ActorRdfJoinMultiStems.findConnectedComponentsInJoinGraph(sortedEntries);
    // Each eddie controller stream is responsible for one connected component of the join graph.
    const eddieControllerStreams = [];
    // The inputs to each controller
    const eddieEntriesInput: IJoinEntryWithMetadata[][] = [];
    for (let j = 0; j < connectedComponents.length; j++){
      const entryJoinVariables: RDF.Variable[][][] = await this.getJoinVariables(connectedComponents[j]);
      const stemOperators: EddieOperatorStream[] = [];
      const inputStreams = [];
      console.log(connectedComponents[j].map(x => x.operation));

      for (const [ i, entry ] of connectedComponents[j].entries()) {
        stemOperators.push(
          new EddieOperatorStream(
            entry.output.bindingsStream,
            timestampGenerator,
            hashFunction,
            <JoinFunction> ActorRdfJoin.joinBindings,
            i,
            (await entry.output.metadata()).variables.map(x => x.variable),
            entryJoinVariables[i],
          ),
        );
        inputStreams.push(entry);
      }
      const controllerStream = new EddieControllerStream(stemOperators, minimalIndexRouter);
      eddieControllerStreams.push(controllerStream);
      eddieEntriesInput.push(inputStreams);
    }

    // In most cases, we will have 1 connected component (no cart joins) and we just return the 
    // controller as join result
    if (eddieControllerStreams.length === 1){
      return {
        result: {
          type: 'bindings',
          bindingsStream: <BindingsStream><any>eddieControllerStreams[0],
          metadata: async() => await this.constructResultMetadata(
            action.entries,
            await ActorRdfJoin.getMetadatas(action.entries),
            action.context,
          ),
        },
      };
    }
    const dataFactory: ComunicaDataFactory = action.context.getSafe(KeysInitQuery.dataFactory);
    const algebraFactory = new Factory(dataFactory);
    
    const connectedComponentEntries = [];
    for (let i = 0; i<eddieControllerStreams.length; i++){
      // Construct metadata from the entries joined by the controller stream.
      const controllerAsEntry: IJoinEntry = {
        output: {
          bindingsStream: <BindingsStream><unknown>eddieControllerStreams[i],
          type: 'bindings',
          metadata: async() => await this.constructResultMetadata(
            eddieEntriesInput[i],
            await ActorRdfJoin.getMetadatas(eddieEntriesInput[i]),
            action.context,
          ),
        },
        operation: algebraFactory
          .createJoin(eddieEntriesInput[i].map(x=>x.operation), false),
      };
      connectedComponentEntries.push(controllerAsEntry);
    }
    return {
      result: await this.mediatorJoin.mediate({
        type: action.type,
        entries: connectedComponentEntries,
        context: action.context,
      }),
    };      
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

  protected async getJoinVariables(entries: IJoinEntryWithMetadata[]): Promise<RDF.Variable[][][]> {
    // For each entry, we determine the variables that overlap with other entries.
    // As some entries are joined on multiple variables, each join variable entry is an array.
    const entryJoinVariables: RDF.Variable[][][] = [];

    for (let i = 0; i < entries.length; i++) {
      const overlappingVariables: string[][] = [];
      for (let j = 0; j < entries.length; j++) {
        if (i !== j) {
          const variablesInner = new Set((await entries[j].output.metadata()).variables.map(x => x.variable.value));
          const variablesOuter = (await entries[i].output.metadata()).variables.map(x => x.variable.value);
          const intersection = variablesOuter.filter(x => variablesInner.has(x));
          // Ensure no duplicate entries are added
          if (intersection.length > 0 && !overlappingVariables.some(x => x.every(y => intersection.includes(y)))) {
            overlappingVariables.push(intersection);
          }
        }
      }
      entryJoinVariables.push([ ...overlappingVariables ].map(x => x.map(y => this.DF.variable(y))));
    }
    return entryJoinVariables;
  }

  /**
   * Find all connected components of the join graph using the find-union datastructure
   * @param entries 
   * @returns 
   */
  static findConnectedComponentsInJoinGraph(
    entries: IJoinEntryWithMetadata[],
  ) {
    function find(idx: number, parent: number[]): number{
      if (parent[idx] === idx) {
        return idx;
      }
      parent[idx] = find(parent[idx], parent);
      return parent[idx];
    }

    function union(a: number, b: number, parent: number[], size: number[]){
      let rootA = find(a, parent);
      let rootB = find(b, parent);
      if (rootA === rootB){
        return;
      }
      if (size[rootA] > size[rootB]){
        parent[rootB] = rootA;
        size[rootA] += size[rootB];
      }
      else{
        parent[rootA] = rootB;
        size[rootB] += size[rootA]
      }
    }

    const n = entries.length;
    const parent: number[] = Array.from({ length: n }, (_, i) => i);
    const size: number[] = Array.from({ length: n }, () => 1); 

    const variableToEntry: Map<string, number[]> = new Map();

    // Map variables to their entries to know which entries to merge
    for (let i = 0; i < n; i++){
      const entryVariables = entries[i].metadata.variables.map(variable => variable.variable.value);
      for (let j = 0; j < entryVariables.length; j++){
        if (!variableToEntry.has(entryVariables[j])){
          variableToEntry.set(entryVariables[j], []);
        }
        variableToEntry.get(entryVariables[j])!.push(i);
      }
    }

    // Union all entries associated with a variable
    for (const entryIndexes of variableToEntry.values()){
      for (let k = 1; k < entryIndexes.length; k++){
        union(entryIndexes[0], entryIndexes[k], parent, size);
      }
    }

    // Reconstruct connected components from union-find datastructure
    const connectedComponents: Map<number, IJoinEntryWithMetadata[]> = new Map();
    for (let l = 0; l<n; l++){
      const parentOfEntry = find(l, parent);
      if (!connectedComponents.has(parentOfEntry)){
        connectedComponents.set(parentOfEntry, []);
      }
      connectedComponents.get(parentOfEntry)!.push(entries[l]);
    }
    return Array.from(connectedComponents.values());
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

