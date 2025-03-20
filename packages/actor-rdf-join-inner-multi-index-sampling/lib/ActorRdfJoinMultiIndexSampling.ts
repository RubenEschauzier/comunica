import type { IActionRdfJoin, IActorRdfJoinOutputInner, IActorRdfJoinArgs, MediatorRdfJoin, IActorRdfJoinTestSideData } from '@comunica/bus-rdf-join';
import { ActorRdfJoin } from '@comunica/bus-rdf-join';
import { ActionContextKey, failTest, TestResult } from '@comunica/core';
import { passTestWithSideData } from '@comunica/core';
import type { IMediatorTypeJoinCoefficients } from '@comunica/mediatortype-join-coefficients';
import type { IJoinEntry, IQuerySource } from '@comunica/types';
import { getSafeBindings,  } from '@comunica/utils-query-operation';
import type * as RDF from '@rdfjs/types';
import { Factory } from 'sparqlalgebrajs';
import type { Pattern } from 'sparqlalgebrajs/lib/algebra';
import type { IEnumerationOutput } from './IndexBasedJoinSampler';
import { IndexBasedJoinSampler } from './IndexBasedJoinSampler';
import { JoinGraph } from './JoinGraph';
import { JoinOrderEnumerator } from './JoinOrderEnumerator';
import type { JoinPlan } from './JoinPlan';
import { MetadataValidationState } from '@comunica/utils-metadata';
import { MediatorExtractSources } from '@comunica/bus-extract-sources';

/**
 * A comunica Inner Multi Index Sampling RDF Join Actor.
 */
export class ActorRdfJoinMultiIndexSampling extends ActorRdfJoin {
  public readonly mediatorJoin: MediatorRdfJoin;
  public readonly mediatorExtractSources: MediatorExtractSources;

  public static readonly FACTORY = new Factory();

  public joinSampler: IndexBasedJoinSampler;
  public estimates: IEnumerationOutput;

  public samplingDone = false;
  public nSamples: number;
  public budget: number;

  public constructor(args: IActorRdfJoinMultiIndexSampling) {
    super(args, {
      logicalType: 'inner',
      physicalName: 'multi-index-sampling',
      limitEntries: 3,
      limitEntriesMin: true,
      canHandleUndefs: true,
      isLeaf: false,
    });
    this.joinSampler = new IndexBasedJoinSampler(this.budget);
  }

  protected override async getOutput(action: IActionRdfJoin): Promise<IActorRdfJoinOutputInner> {
    console.log("Start sampling")
    const sourcesUnioned = await this.mediatorExtractSources.mediate({
      operations: action.entries.map(x=>x.operation),
      context: action.context
    });
    
    if ([...Object.keys(sourcesUnioned.sources)].length === 0){
      console.warn("No sources found in sampling.")
      return {
        result: await this.mediatorJoin.mediate({
          type: action.type,
          entries: action.entries,
          context: action.context.set(KEY_NESTED_QUERY, true),
        }),
      };
    }
    const sources = [...Object.values(sourcesUnioned.sources)];

    // Source extraction found no sources to sample over. This can happen
    // during for example traversal queries. In this case use default join strategy
    const aggSourceFn = ActorRdfJoinMultiIndexSampling.aggregateSampleFn.bind(null, sources);
    const aggCountFn = ActorRdfJoinMultiIndexSampling.aggregateCountFn.bind(null, sources);

    const joinGraph = new JoinGraph(action.entries);
    joinGraph.constructJoinGraphBFS(joinGraph.getEntries()[0]);

    this.estimates = await this.joinSampler.run(
      joinGraph.getEntries().map(x => <Pattern> x.operation),
      this.nSamples,
      aggSourceFn,
      aggCountFn,
    );
    
    if (this.estimates.maxSizeEstimated === 1){
      console.warn(`${this.name}: Cardinality sampling failed, budget exhausted prematurely.`);
      return {
        result: await this.mediatorJoin.mediate({
          type: action.type,
          entries: action.entries,
          context: action.context.set(KEY_BUDGET_EXHAUSTED, true)
        }),
      };
    }

    // Seperate the enumerator from graph object to allow for easier testing
    const enumerator: JoinOrderEnumerator = new JoinOrderEnumerator(
      joinGraph.getAdjencyList(),
      this.estimates,
      joinGraph.getEntries().map(x => x.operation),
    );

    const joinPlan: JoinPlan = enumerator.search();

    const multiJoinOutput = await joinPlan.executePlan(
      joinGraph.getEntries(),
      action,
      this.joinTwoEntries.bind(this),
    );

    // This is a partial join plan, so we must join the remaining entries too
    if (joinPlan.entries.size !== action.entries.length){
      console.log(`Partial plan: ${joinPlan.entries.size}/${action.entries.length}`);
      const context = action.context.set(KEY_BUDGET_EXHAUSTED, true);
      const leftOverEntries = joinGraph.getEntries().filter((_, i) => !joinPlan.entries.has(i));
      const originalMetadataFn = multiJoinOutput.output.metadata();

      multiJoinOutput.output.metadata = async () => {
        const metadata = await originalMetadataFn;
        metadata.cardinality = { value: joinPlan.estimatedSize, type: "estimate" };
        metadata.state = new MetadataValidationState();
        return metadata;
      }
      console.log(await(multiJoinOutput.output.metadata()))
      console.log(leftOverEntries)
      leftOverEntries.push(multiJoinOutput);
      return {
        result: await this.mediatorJoin.mediate({
          type: action.type,
          entries: leftOverEntries,
          context: context,
        }),
      };
    }
    return {
      result: {
        type: 'bindings',
        bindingsStream: multiJoinOutput.output.bindingsStream,
        metadata: async() => { 
          const metadata = await this.constructResultMetadata(
            action.entries,
            await ActorRdfJoin.getMetadatas(action.entries),
            action.context,
          ) 
          metadata.cardinality = {value: joinPlan.estimatedSize, type: "estimate"}
          return metadata;
        },
      },
    };
  }

  protected async getJoinCoefficients(
    action: IActionRdfJoin,
    sideData: IActorRdfJoinTestSideData,
  ): Promise<TestResult<IMediatorTypeJoinCoefficients, IActorRdfJoinTestSideData>> {
    if (action.context.get(KEY_BUDGET_EXHAUSTED)){
      return failTest(`${this.name}: budget exhausted for join sampling.`);
    }
    if (action.context.get(KEY_NESTED_QUERY)){
      return failTest(`${this.name}: join sampling can't run for nested query calls`);
    }

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

  public static async aggregateCountFn(sources: IQuerySource[],       
    subject?: RDF.Term,
    predicate?: RDF.Term,
    object?: RDF.Term,
    graph?: RDF.Term
  ): Promise<number> {
    let count = 0;
    for (const source of sources) {
      if (!source.countQuads) {
        throw new Error("Got source without countQuads function");
      }
      count += await source.countQuads(subject, predicate, object, graph);
    }
    return count;
  }

  public static async aggregateSampleFn(
    sources: IQuerySource[],
    indexes: number[],
    subject?: RDF.Term,
    predicate?: RDF.Term,
    object?: RDF.Term,
    graph?: RDF.Term
  ): Promise<RDF.Quad[]>{
    if (indexes.length === 0){
      return [];
    }
    // Sort high to low and pop index to get first index in sequence
    indexes.sort((a,b) => b-a);
    const quads: RDF.Quad[] = []
    let currentIndex: number | undefined = indexes.pop()!;
    let searchIndex = 0;
    for (const source of sources){
      if (!source.countQuads) {
        throw new Error("Got source without countQuads function");
      }
      const sourceCount = await source.countQuads(subject, predicate, object, graph);
      const stepAhead = searchIndex + sourceCount
      if (stepAhead > currentIndex){
        // When the current source contains the index, we loop over indexes 
        // to find all indexes belonging to this source.
        const indexesToSampleFromSource = [currentIndex - searchIndex];
        while (stepAhead > currentIndex! && currentIndex !== undefined) {
          currentIndex = indexes.pop();
          if (currentIndex && stepAhead > currentIndex!){
            indexesToSampleFromSource.push(currentIndex - searchIndex);
          }
        }
        if (!source.sample){
          throw new Error("Got source without sample function");
        }
        // We sample from the source with normalized indexes. As the source 
        // sample fn assumes indexes within the range of the matching triples in 
        // the source
        quads.push(...source.sample(indexesToSampleFromSource, 
          subject, predicate, object, graph)
        );
        if (currentIndex === undefined) {
          return quads;
        }        
      }
      searchIndex += sourceCount;
    }
    return quads;
  }   
  
}


export interface IActorRdfJoinMultiIndexSampling extends IActorRdfJoinArgs {
  /**
   * A mediator for joining Bindings streams
   */
  mediatorJoin: MediatorRdfJoin;
  /**
   * A mediator for extracting sources from a given action
   */
  mediatorExtractSources: MediatorExtractSources;
  /**
   * The maximum number of entries in the source cache, set to 0 to disable.
   * @range {integer}
   * @default {10000}
   */
  budget: number
  /**
   * The maximum number of entries in the source cache, set to 0 to disable.
   * @range {integer}
   * @default {250}
   */
  nSamples: number
}

/**
 * Key that indicates to recursive calls to the join bus that there is no more sampling budget
 * for a given query. So other, non-sampling methods should be used
 */
export const KEY_BUDGET_EXHAUSTED = new ActionContextKey<boolean>(
  '@comunica/actor-rdf-join:exhausted',
);

/**
 * Key indicating join sampling should not be called due to nested query calls
 */

export const KEY_NESTED_QUERY = new ActionContextKey<boolean>(
  '@comunica/actor-rdf-join:nested'
);