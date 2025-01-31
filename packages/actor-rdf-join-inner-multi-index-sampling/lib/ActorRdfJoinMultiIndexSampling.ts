import * as fs from 'node:fs';
import type { QuerySourceSkolemized } from '@comunica/actor-context-preprocess-query-source-skolemize';
import type { IActionRdfJoin, IActorRdfJoinOutputInner, IActorRdfJoinArgs, MediatorRdfJoin, IActorRdfJoinTestSideData } from '@comunica/bus-rdf-join';
import { ActorRdfJoin } from '@comunica/bus-rdf-join';
import type { TestResult } from '@comunica/core';
import { passTestWithSideData } from '@comunica/core';
import type { IMediatorTypeJoinCoefficients } from '@comunica/mediatortype-join-coefficients';
import type { IJoinEntry, IQuerySource } from '@comunica/types';
import { getSafeBindings, getSources } from '@comunica/utils-query-operation';
import type * as RDF from '@rdfjs/types';
import { Store, Parser } from 'n3';
import { Factory } from 'sparqlalgebrajs';
import type { Pattern } from 'sparqlalgebrajs/lib/algebra';
import type { QuerySourceHypermedia } from '../../actor-query-source-identify-hypermedia/lib';
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
    this.joinSampler = new IndexBasedJoinSampler(1000);
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
    const sourcesOperations = action.entries.map(x => getSources(x.operation));
    const sources: Record<string, IQuerySource> = {};
    sourcesOperations.map(x => x.map((sourceWrapper) => {
      const sourceString = sourceWrapper.source.toString();
      if (sources[sourceString] === undefined) {
        sources[sourceString] = sourceWrapper.source;
      }
    }));
    const sourceToSample = <QuerySourceHypermedia><unknown>
    (<QuerySourceSkolemized><unknown>sources[Object.keys(sources)[0]]).innerSource;
    const source = (await sourceToSample.sourcesState.get([ ...sourceToSample.sourcesState.keys() ][0])!).source;
    if (!source.sample) {
      throw new Error('Found source that does not support sampling');
    }
    if (!source.countQuads) {
      throw new Error('Found source that does not support quad counting');
    }

    // Join Graph reorders entries in breadth-first style from 0 index of entries. So always use the entries of
    // the join graph to ensure number representations of entries match.
    const joinGraph = new JoinGraph(action.entries);
    joinGraph.constructJoinGraphBFS(joinGraph.getEntries()[0]);

    this.estimates = await this.joinSampler.run(
      joinGraph.getEntries().map(x => <Pattern> x.operation),
      1000,
      source.sample.bind(source),
      source.countQuads.bind(source),
    );
    console.log(this.estimates);

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
    _action: IActionRdfJoin,
    sideData: IActorRdfJoinTestSideData,
  ): Promise<TestResult<IMediatorTypeJoinCoefficients, IActorRdfJoinTestSideData>> {
    return passTestWithSideData({
      iterations: 1,
      persistedItems: 0,
      blockingItems: 0,
      requestTime: 0,
    }, sideData);
  }

  public async joinTwoEntries(entry1: IJoinEntry, entry2: IJoinEntry, action: IActionRdfJoin): Promise<IJoinEntry> {
    return {
      output: getSafeBindings(await this.mediatorJoin
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
