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
import { JoinGraph } from './JoinGraph';
import { JoinOrderEnumerator } from './JoinOrderEnumerator';
import type { JoinTree } from './JoinTree';

/**
 * A comunica Inner Multi Index Sampling RDF Join Actor.
 */
export class ActorRdfJoinMultiIndexSampling extends ActorRdfJoin {
  public readonly mediatorJoin: MediatorRdfJoin;
  public static readonly FACTORY = new Factory();

  public joinSampler: IndexBasedJoinSampler;
  public estimates: Map<string, ISampleResult>;

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
    // Highly inefficient, should be moved to creation of store
    const { store, arrayIndexes } = this.createArrayIndex(action);

    // Join Graph reorders entries in breadth-first style from 0 index of entries. So always use the entries of
    // the join graph to ensure number representations of entries match.
    const joinGraph = new JoinGraph(action.entries);
    joinGraph.constructJoinGraphBFS(joinGraph.getEntries()[0]);

    this.estimates = this.joinSampler.run(
      joinGraph.getEntries().map(x => x.operation),
      1_000,
      Number.POSITIVE_INFINITY,
      store,
      arrayIndexes,
    );
    // Seperate the enumerator from graph object to allow for easier testing
    const enumerator: JoinOrderEnumerator = new JoinOrderEnumerator(
      joinGraph.getAdjencyList(),
      this.estimates,
      joinGraph.getEntries().map(x => x.operation),
    );
    const joinPlan: JoinTree = enumerator.search();
    const multiJoinOutput = await joinPlan.executePlan(
      joinGraph.getEntries(),
      action,
      this.joinTwoEntries.bind(this),
    );

    return {
      result: {
        type: 'bindings',
        bindingsStream: multiJoinOutput.output.bindingsStream,
        metadata: async() => await this.constructResultMetadata(
          action.entries,
          await ActorRdfJoin.getMetadatas(action.entries),
          action.context,
        ),
      },
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

  public async joinTwoEntries(entry1: IJoinEntry, entry2: IJoinEntry, action: IActionRdfJoin): Promise<IJoinEntry> {
    return {
      output: ActorQueryOperation.getSafeBindings(await this.mediatorJoin
        .mediate({ type: action.type, entries: [ entry1, entry2 ], context: action.context })),
      operation: ActorRdfJoinMultiIndexSampling.FACTORY
        .createJoin([ entry1.operation, entry2.operation ], false),
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
