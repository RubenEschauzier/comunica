import * as fs from 'node:fs';
import type { QuerySourceSkolemized } from '@comunica/actor-context-preprocess-query-source-skolemize';
import { ActorQueryOperation } from '@comunica/bus-query-operation';
import type { IActionRdfJoin, IActorRdfJoinOutputInner, IActorRdfJoinArgs, MediatorRdfJoin } from '@comunica/bus-rdf-join';
import { ActorRdfJoin } from '@comunica/bus-rdf-join';
import { KeysQueryOperation } from '@comunica/context-entries';
import type { IMediatorTypeJoinCoefficients } from '@comunica/mediatortype-join-coefficients';
import type { IJoinEntry, IQuerySourceWrapper, MetadataBindings } from '@comunica/types';
import type * as RDF from '@rdfjs/types';
import { Store, Parser } from 'n3';
import { Factory } from 'sparqlalgebrajs';
import type { ArrayIndex, ISampleResult } from './IndexBasedJoinSampler';
import { IndexBasedJoinSampler } from './IndexBasedJoinSampler';
import { JoinGraph } from './JoinGraph'
import { JoinOrderEnumerator } from './JoinOrderEnumerator';
/**
 * A comunica Inner Multi Index Sampling RDF Join Actor.
 */
export class ActorRdfJoinMultiIndexSampling extends ActorRdfJoin {
  public readonly mediatorJoin: MediatorRdfJoin;
  public static readonly FACTORY = new Factory();

  public joinSampler: IndexBasedJoinSampler;
  public estimates: Record<string, ISampleResult>;

  public samplingDone = false;
  public nSamples = 0;

  public constructor(args: IActorRdfJoinMultiIndexSampling) {
    super(args, {
      logicalType: 'inner',
      physicalName: 'multi-index-sampling',
      limitEntries: 3,
      limitEntriesMin: true,
      canHandleUndefs: true,
      isLeaf: false,
    });
    this.joinSampler = new IndexBasedJoinSampler(10000);
  }

  /**
   * Temporary function to test the possiblity of using join sampling. This will NOT be used in any 'real' tests
   * @param fileName Filename to parse
   */
  public parseFile(fileName: string) {
    const parser = new Parser();
    const store = new Store();
    const data: string = fs.readFileSync(fileName, 'utf-8');
    const testResult: RDF.Quad[] = parser.parse(
      data,
    );
    store.addQuads(testResult);
    return store;
  }

  protected async getOutput(action: IActionRdfJoin): Promise<IActorRdfJoinOutputInner> {
    // Call this only once to get cardinalities, then do dynamic programming to find optimal order, perform joins
    // return joined result + purge cardinalities.
    if (this.estimates === undefined) {
      const { store, arrayIndexes } = this.createArrayIndex(action);
      this.estimates = this.joinSampler.run(
        action.entries.map(x => x.operation),
        1_000,
        Number.POSITIVE_INFINITY,
        store,
        arrayIndexes,
      );
      const joinGraph = new JoinGraph(action.entries.map(x => x.operation))
      joinGraph.constructJoinGraphBFS(joinGraph.getEntries()[0]);
      // Seperate the enumerator from graph object to allow for easier testing
      // const enumerator: JoinOrderEnumerator = new JoinOrderEnumerator(
      //   joinGraph.getAdjencyList()
      // )
      // enumerator.enumerateCsg(action.entries.map(x => x.operation));
      const adjacencyList: Map<number, number[]> = new Map([
        [0, [1, 2, 3]],
        [1, [0, 4]],
        [2, [0, 3, 4]],
        [3, [0, 2, 4]],
        [4, [1, 2, 3]]
      ]);

      const testEnumerator: JoinOrderEnumerator = new JoinOrderEnumerator(
        adjacencyList,
        this.estimates,
        joinGraph.getEntries()
      )
      testEnumerator.search()
      // testEnumerator.enumerateCsgCmpPairs(5)
    }

    const firstEntry: IJoinEntry = {
      output: ActorQueryOperation.getSafeBindings(await this.mediatorJoin
        .mediate({ type: action.type, entries: [ action.entries[0], action.entries[1] ], context: action.context })),
      operation: ActorRdfJoinMultiIndexSampling.FACTORY
        .createJoin([ action.entries[0].operation, action.entries[1].operation ], false),
    };
    const remainingEntries: IJoinEntry[] = action.entries.slice(1);
    remainingEntries[0] = firstEntry;
    return {
      result: await this.mediatorJoin.mediate({
        type: action.type,
        entries: remainingEntries,
        context: action.context,
      }),
    };
  }

  protected async getJoinCoefficients(
    action: IActionRdfJoin,
    metadatas: MetadataBindings[],
  ): Promise<IMediatorTypeJoinCoefficients> {
    return {
      iterations: 0,
      persistedItems: 0,
      blockingItems: 0,
      requestTime: 0,
    };
  }

  protected createArrayIndex(action: IActionRdfJoin): IConstructedIndexes {
    // This is test code and highly inefficient, but parse document uri from the source into N3 store
    // and use that to get samples from adjusted index
    const querySources: IQuerySourceWrapper[] = action.context.get(KeysQueryOperation.querySources)!;
    const file: string = <string> (<QuerySourceSkolemized> querySources[0].source).referenceValue;
    const store = this.parseFile(file);

    // 'any' typing to access _graph which is a private function in typescript
    const storeWithArrayIndexes: any & Store = store;
    // Array indexes per graph
    const arrayIndexes: Record<string, ArrayIndex> = {};
    // For all stored graphs
    for (const [ graph, graphIndex ] of Object.entries(storeWithArrayIndexes._graphs)) {
      // Dictionary for a graph indexes in form of spo, pos, and osp. The final term is represented as an array
      // for efficient sampling.
      const graphArrayDict: ArrayIndex = {};
      // For stored indexes (spo, pos, osp) in graph
      for (const index of Object.keys(<Record<string, any>>graphIndex)) {
        const arrayDict = this.joinSampler.constructArrayDict(storeWithArrayIndexes._graphs[graph][index]);
        graphArrayDict[index] = arrayDict;
      }
      arrayIndexes[graph] = graphArrayDict;
    }
    return {
      store,
      arrayIndexes,
    };
  }
}

export interface IActorRdfJoinMultiIndexSampling extends IActorRdfJoinArgs {
  /**
   * A mediator for joining Bindings streams
   */
  mediatorJoin: MediatorRdfJoin;
}

export interface IConstructedIndexes {
  /**
   * N3 store
   */
  store: any;
  /**
   * Created array version of index
   */
  arrayIndexes: Record<string, ArrayIndex>;
}
