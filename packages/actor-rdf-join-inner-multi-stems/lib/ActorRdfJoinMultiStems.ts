import type { MediatorHashBindings } from '@comunica/bus-hash-bindings';
import type {
  IActionRdfJoin,
  IActorRdfJoinArgs,
  IActorRdfJoinOutputInner,
  IActorRdfJoinTestSideData,
  MediatorRdfJoin,
} from '@comunica/bus-rdf-join';
import { ActorRdfJoin } from '@comunica/bus-rdf-join';
import type { MediatorRdfJoinEntriesSort } from '@comunica/bus-rdf-join-entries-sort';
import { KeysInitQuery, KeysRdfJoin, KeysStatistics } from '@comunica/context-entries';
import type { TestResult } from '@comunica/core';
import { failTest, passTestWithSideData } from '@comunica/core';
import type { IMediatorTypeJoinCoefficients } from '@comunica/mediatortype-join-coefficients';
import type {
  BindingsStream,
  ComunicaDataFactory,
  IActionContext,
  IJoinEntry,
  IJoinEntryWithMetadata,
} from '@comunica/types';
import type * as RDF from '@rdfjs/types';
import { DataFactory } from 'rdf-data-factory';
import { Factory } from 'sparqlalgebrajs';
import { EddieControllerStream, TimestampGenerator } from './EddieControllerStream';
import type { JoinFunction } from './EddieOperatorStream';
import { EddieOperatorStream } from './EddieOperatorStream';
import type { IEddieRouterFactory } from './routers/BaseRouter';

/**
 * A comunica Inner Multi Stems RDF Join Actor.
 */
export class ActorRdfJoinMultiStems extends ActorRdfJoin<IActorRdfJoinMultiStemsTestSideData> {
  public readonly mediatorHashBindings: MediatorHashBindings;
  public readonly mediatorJoinEntriesSort: MediatorRdfJoinEntriesSort;
  public readonly mediatorJoin: MediatorRdfJoin;
  public readonly routerFactory: IEddieRouterFactory;
  public readonly routerUpdateFrequency: number;

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

  public override async test(
    action: IActionRdfJoin,
  ): Promise<TestResult<IMediatorTypeJoinCoefficients, IActorRdfJoinMultiStemsTestSideData>> {
    if (action.context.get(KeysRdfJoin.joinEntriesAdaptiveJoinResult)) {
      return failTest(`${this.name} can not be invoked recursively.`);
    }
    return super.test(action);
  }

  protected async getOutput(
    action: IActionRdfJoin,
    sideData: IActorRdfJoinMultiStemsTestSideData,
  ): Promise<IActorRdfJoinOutputInner> {
    const skipLog = action.context.get(KeysStatistics.skipStatisticTracking);
    const logger = ActorRdfJoinMultiStems.getContextLogger(action.context);
    // TODO: This goes wrong with multiple separate connected components
    const snapShotLogger = action.context.get(KeysStatistics.adaptiveJoinStatistics);
    const queryString = action.context.get(KeysInitQuery.queryString);

    let { metadatas } = sideData;
    metadatas = [ ...metadatas ];

    // By sorting the entries and selecting the minimal "not done" entry we route bindings
    // to the streams with smallest cardinality first by default (before adaptivity)
    const sortedEntries: IJoinEntryWithMetadata[] = await this.sortJoinEntries(action.entries
      .map((entry, i) => ({ ...entry, metadata: metadatas[i] })), action.context);

    const { hashFunction } = await this.mediatorHashBindings.mediate({ context: action.context });
    const timestampGenerator = new TimestampGenerator();

    // TODO: Treat connected components as sub queries for derived resources, so when a derived resource comes
    // in we test containment in each component, as we don't want to deal with connecting unconnected components etc
    const connectedComponents = ActorRdfJoin.findConnectedComponentsInJoinGraph(sortedEntries);

    const eddieControllerStreams = [];
    const eddieEntriesInput: IJoinEntryWithMetadata[][] = [];
    for (const connectedComponentEntries of connectedComponents.entries) {
      const entriesJoinVariables: RDF.Variable[][][] = await this.getJoinVariables(connectedComponentEntries);
      let componentHasCartesian = false;
      for (const joinVariableEntry of entriesJoinVariables) {
        if (joinVariableEntry.length === 0) {
          componentHasCartesian = true;
        }
      }
      const stemOperators: EddieOperatorStream[] = [];
      const inputStreams = [];

      for (const [ i, entry ] of connectedComponentEntries.entries()) {
        stemOperators.push(
          new EddieOperatorStream(
            entry.output.bindingsStream,
            timestampGenerator,
            hashFunction,
            <JoinFunction> ActorRdfJoin.joinBindings,
            i,
            (await entry.output.metadata()).variables.map(x => x.variable),
            this.getComponentSubjectObjectIRIs(entry),
            entriesJoinVariables[i],
            componentHasCartesian,
          ),
        );
        inputStreams.push(entry);
      }
      const router = this.routerFactory.createRouter();

      let logContext: Record<string, any> | undefined;
      if (logger && !skipLog) {
        logContext = {
          query: queryString,
        };
      }
      const controllerStream = new EddieControllerStream(
        stemOperators,
        router,
        this.routerUpdateFrequency,
        snapShotLogger,
        logger,
        logContext,
      );
      eddieControllerStreams.push(controllerStream);
      eddieEntriesInput.push(inputStreams);
    }

    // In most cases, we will have 1 connected component (no cart joins) and we just return the
    // controller as join result
    if (eddieControllerStreams.length === 1) {
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
    for (const [ i, eddieControllerStream ] of eddieControllerStreams.entries()) {
      // Construct metadata from the entries joined by the controller stream.
      const controllerAsEntry: IJoinEntry = {
        output: {
          bindingsStream: <BindingsStream><unknown>eddieControllerStream,
          type: 'bindings',
          metadata: async() => await this.constructResultMetadata(
            eddieEntriesInput[i],
            await ActorRdfJoin.getMetadatas(eddieEntriesInput[i]),
            action.context,
          ),
        },
        operation: algebraFactory
          .createJoin(eddieEntriesInput[i].map(x => x.operation), false),
      };
      connectedComponentEntries.push(controllerAsEntry);
    }
    return {
      result: await this.mediatorJoin.mediate({
        type: action.type,
        entries: connectedComponentEntries,
        context: action.context.set(KeysRdfJoin.joinEntriesAdaptiveJoinResult, true),
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

  protected getComponentSubjectObjectIRIs(entry: IJoinEntryWithMetadata): RDF.NamedNode[] {
    const { subjects, objects } = ActorRdfJoin.getOperatorPatternSubjectObjectIRIs(entry.operation);
    const namedNodesAsString = [ ...subjects.values(), ...objects.values() ];
    return namedNodesAsString.map(x => this.DF.namedNode(x));
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
  /**
   * The routing strategy used
   */
  routerFactory: IEddieRouterFactory;
  /**
   * Update frequency of routing table, lower values mean more routing policy switches
   */
  routerUpdateFrequency: number;
}

export interface IActorRdfJoinMultiStemsTestSideData extends IActorRdfJoinTestSideData {
  sortedEntries: IJoinEntryWithMetadata[];
}

export function stringifyDepth(
  obj: unknown,
  maxDepth = 2,
  indent = 2,
): string {
  const seen = new WeakSet<object>();

  function helper(value: unknown, depth: number): unknown {
    // Stop at max depth
    if (depth > maxDepth) {
      return '...';
    }

    // Handle primitives
    if (value === null || typeof value !== 'object') {
      return value;
    }

    // Handle circular
    if (seen.has(value)) {
      if ((<any>(value)).termType) {
        return value;
      }
      return '[Circular]';
    }
    seen.add(value);

    if (Array.isArray(value)) {
      return value.map(v => helper(v, depth + 1));
    }

    // Handle objects
    const out: Record<string, unknown> = {};
    for (const [ k, v ] of Object.entries(value)) {
      out[k] = helper(v, depth + 1);
    }

    return out;
  }

  return JSON.stringify(helper(obj, 1), null, indent);
}
