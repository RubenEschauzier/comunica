import { ActionContextKey } from '@comunica/core';
import type {
  IAdaptivePlanStatistics,
  IPlanStatistic,
  IPlanSummary,
  IProducedTupleSummary,
  IStatsOperators,
  Logger,
} from '@comunica/types';
import type { Bindings } from '@comunica/utils-bindings-factory';
import { AsyncIterator } from 'asynciterator';
import type { StemsOperatorStream, ISelectivityData } from './StemsOperatorStream';
import type { IRouteTableOperation, IStemsRouter, IStemsRoutingEntry } from './routers/BaseRouter';

export class StemsControllerStream extends AsyncIterator<Bindings> {
  /**
   * The iterators performing the half-joins and producing triple pattern
   * matches
   */
  private readonly stemsIterators: StemsOperatorStream[];
  /**
   * The routing object constructing the routing table.
   */
  private readonly router: IStemsRouter;
  /**
   * Table indicating the next eddieIterator for a given 'done'
   * signature
   */
  private routingTable: Record<string, IStemsRoutingEntry[][]>;
  /**
   * Flag indicating if the all triple patterns have no new matches.
   * If this flag is set, the iterator should finish all current intermediate
   * results and end.
   */
  private endTuples: boolean;
  /**
   * Loggers for debugging
   */
  private readonly logger: Logger | undefined;
  private readonly logContext: Record<string, any> | undefined;
  private loggedEndEvent = false;
  /**
   * Contains a list of snapshots at given number of tuples processed
   */
  public snapshots: Record<number, IAdaptivePlanStatistics> | undefined;
  /**
   * Array indicating if operator stream at index i has finished
   * reading the data from its corresponding join entry
   */
  private finishedReading: number[];
  /**
   * The number of processed bindings since the last update
   */
  private bindingsSinceUpdate = 0;
  /**
   * Total bindings processed;
   */
  private bindingsTotal = 0;
  /**
   * How often the routing table should be updated.
   */
  private readonly routingUpdateFrequency: number;


  protected totalProduced: number = 0;

  public constructor(
    eddieIterators: StemsOperatorStream[],
    router: IStemsRouter,
    updateFrequency: number,
    snapshotsTracker?: Record<number, IAdaptivePlanStatistics>,
    logger?: Logger,
    logContext?: Record<string, any>,
  ) {
    super();
    this.stemsIterators = eddieIterators;
    this.finishedReading = Array.from<number>({ length: eddieIterators.length }).fill(0);
    this.router = router;
    this.routingUpdateFrequency = updateFrequency;
    this.routingTable = this.router.createRouteTable(
      this.stemsIterators.map(it => ({
        operations: it.operations,
        doneBitMask: it.doneBitMask,
        variables: it.variables,
        namedNodes: it.namedNodes,
      })),
    );

    this.endTuples = false;

    this.snapshots = snapshotsTracker;
    this.logger = logger;
    this.logContext = logContext;

    if (this.stemsIterators.some(it => it.readable)) {
      this.readable = true;
    }

    for (const [ index, it ] of this.stemsIterators.entries()) {
      it.on('readable', () => this.readable = true);
      it.on('endRead', () => {
        // When all eddiestreams finished creating new tuples,
        // we only need to process the remaining tuples in buffers
        this.finishedReading[index] = 1;
        if (this.finishedReading.every(val => val === 1)) {
          this.endTuples = true;
          const hasBufferedData = this.stemsIterators.some(op => op.readable && !op.ended);
          if (!hasBufferedData) {
            this._end();
          }
        }
      });
    }
    this.on('end', () => this._end());
  }

  get numOperators(){
    return this.stemsIterators.length;
  }

  public addOperator(
    stemsOperatorStream: StemsOperatorStream,
    metadata?: Record<string, any>,
  ) {
    const opIndex = this.stemsIterators.length;
    this.stemsIterators.push(stemsOperatorStream);
    this.finishedReading.push(0);
    // Ensure that if the other streams are ended and we get a new stream the controller
    // stream stays open
    this.endTuples = false;
    if (stemsOperatorStream.readable){
      this.readable = true;
    }
    stemsOperatorStream.on('readable', () => this.readable = true);
    stemsOperatorStream.on('endRead', () => {
      // When all eddiestreams finished creating new tuples,
      // we only need to process the remaining tuples in buffers
      this.finishedReading[opIndex] = 1;
      if (this.finishedReading.every(val => val === 1)) {
        this.endTuples = true;
        const hasBufferedData = this.stemsIterators.some(op => op.readable && !op.ended);
        if (!hasBufferedData) {
          this._end();
        }
      }
    });
    const updatedTable = this.router.addOperator(
      this.routingTable,
      stemsOperatorStream,
      metadata,
    );

    if (updatedTable) {
      this.routingTable = updatedTable;
    }
  }

  public override _end(): void {
    if (!this.loggedEndEvent && this.snapshots && this.logger) {
      const finalStatistics = this.buildStatistics();
      const totalTimeUpdatingTable = Object.values(this.snapshots)
        .reduce((acc, snapshot) => acc + snapshot.timeToUpdateRouting!, 0);
      finalStatistics.timeToUpdateRouting = totalTimeUpdatingTable;

      this.snapshots[this.bindingsTotal] = finalStatistics;
      this.logger.debug(JSON.stringify(this.snapshots, null, 2), this.logContext);
      this.loggedEndEvent = true;
    }
    for (const it of this.stemsIterators) {
      it.destroy();
    }
    super._end();
  }

  public override read(): null | Bindings {
    while (true) {
      let item: Bindings | null = null;
      let producedResults = false;
      for (const eddieIterator of this.stemsIterators) {
        item = eddieIterator.read();
        if (item === null) {
          continue;
        }
        this.bindingsSinceUpdate++;
        this.bindingsTotal++;

        producedResults = true;

        const partialResultMetadata = item.getContextEntry(stemsContextKeys.eddiesMetadata)!;
        const nextRoutes = this.routingTable[partialResultMetadata.done];

        // All done bits equal to 1 (terminal state)
        if (nextRoutes === undefined || nextRoutes.length === 0) {
          return item;
        }

        // Fast-path: When only a single route exists (e.g. pure base execution without CRs)
        if (nextRoutes.length === 1) {
          const route = nextRoutes[0];
          if (route.length > 0) {
            const nextStep = route[0];
            this.stemsIterators[nextStep.next].push({
              item,
              joinVars: nextStep.joinVars,
            });
          }
        } else {
          // Multi-route path (with CR alternatives): use bitmask to avoid per-read Set allocation
          let nextRoutesSeenMask = 0;
          const lastCr = partialResultMetadata.lastCrIndex ?? -1;

          for (const route of nextRoutes) {
            if (route.length > 0) {
              const nextStep = route[0];
              const targetOp = this.stemsIterators[nextStep.next];

              // If forbiddenBaseMask is set due to routing to a CR-based plan,
              // target operator must not overlap with forbidden operators covered by the CR
              if (partialResultMetadata.forbiddenBaseMask !== undefined &&
                  (partialResultMetadata.forbiddenBaseMask & targetOp.doneBitMask) !== 0) {
                continue;
              }

              // CRs must be executed in ascending order to prevent symmetry causing duplicate plans 
              // (CR1 \Join CR2) is same as (CR2 \join CR1) and tuples get duplicated and routed to both
              const stepCrIndex = nextStep.crIndex ?? (targetOp.isCompositeResource ? targetOp.operatorIndex : undefined);
              if (stepCrIndex !== undefined && stepCrIndex <= lastCr) {
                continue;
              }

              const targetBit = 1 << nextStep.next;
              // Enforce lazy branching on CR paths by tracking nextRouting indexes
              if ((nextRoutesSeenMask & targetBit) === 0) {
                let itemToPush = item;
                // If taking a CR alternative route where the next step is an intermediate base operator,
                // record the CR's doneBitMask in forbiddenBaseMask to prevent joining replaced base operators
                if (nextStep.crIndex !== undefined && nextStep.next !== nextStep.crIndex) {
                  const newMetadata: IStemsBindingsMetadata = {
                    ...partialResultMetadata,
                    forbiddenBaseMask: (partialResultMetadata.forbiddenBaseMask ?? 0) | (nextStep.crDoneBitMask ?? 0),
                  };
                  itemToPush = item.setContextEntry(stemsContextKeys.eddiesMetadata, newMetadata);
                }

                this.stemsIterators[nextStep.next].push({
                  item: itemToPush,
                  joinVars: nextStep.joinVars,
                });
                nextRoutesSeenMask |= targetBit;
              }
            }
          }
        }

        if (this.bindingsSinceUpdate >= this.routingUpdateFrequency) {
          const start = performance.now();
          this.routingTable = this.router.updateRouteTable(this.stemsIterators, this.routingTable);
          const end = performance.now();
          if (this.logger && this.logContext && this.snapshots) {
            const statisticsAtUpdate = this.buildStatistics();
            statisticsAtUpdate.timeToUpdateRouting = (end - start);
            this.snapshots[this.bindingsTotal] = statisticsAtUpdate;
          }

          this.bindingsSinceUpdate = 0;
        }
      }

      // If none of the eddieIterators have a match for us
      // this stream is not readable.
      if (this.endTuples && item === null) {
        this._end();
        return null;
      }

      if (this.done || !producedResults) {
        this.readable = false;
        return null;
      }
    }
  }

  public buildStatistics(): IAdaptivePlanStatistics {
    const { fullOrders, partialOrders } = this.buildPlanSummary();
    const {
      producedTuples,
      producedJoinResults,
      producedIntermediateResults,
    } = this.getProducedTuplesSummary();
    const operatorSummaries = this.getOperatorSummary(producedTuples);
    const executionStatistics: IAdaptivePlanStatistics = {
      fullOrderStatistics: fullOrders,
      partialOrderStatistics: partialOrders,
      operatorSummaries,
      totalIntermediate: producedIntermediateResults,
      fanOut: producedIntermediateResults / producedJoinResults,
      reduction: producedIntermediateResults / producedTuples.reduce((acc, curr) => acc + curr, 0),
    };
    return executionStatistics;
  }

  private getOperatorSummary(producedTuples: number[]): IStatsOperators[] {
    const operatorSummaries: IStatsOperators[] = [];
    for (let i = 0; i < this.stemsIterators.length; i++) {
      operatorSummaries.push({
        producedTuples: producedTuples[i],
        timeEmpty: this.stemsIterators[i].timeEmpty,
        timeNonEmpty: this.stemsIterators[i].timeNonEmpty,
        emptyReads: this.stemsIterators[i].nFailedReads,
        nonEmptyReads: this.stemsIterators[i].nSuccessReads,
      });
    }
    return operatorSummaries;
  }

  private getProducedTuplesSummary(): IProducedTupleSummary {
    const producedTuples: number[] = <number[]> Array.from({ length: this.stemsIterators.length }).fill(0);
    let producedJoinResults = 0;
    let producedIntermediateResults = 0;
    const orderSelectivities = this.stemsIterators.map(it => it.selectivitiesOrders);
    for (const orderSelectivity of orderSelectivities) {
      for (const [ key, value ] of Object.entries(orderSelectivity)) {
        const orderAsArray = key.split(',').map(s => Number.parseInt(s.trim(), 10));
        // Singleton orders mean it only passed one operator: the one it was produced by
        if (orderAsArray.length === 1) {
          // The in value denotes the number of tuples with that signature, which is equal
          // to the number of tuples the triple pattern associated with the operator
          // produced
          producedTuples[orderAsArray[0]] += value.in;
        }
        // Last iterator isn't in the key signature as it is implied by the index of the
        // eddies iterator
        if (orderAsArray.length === this.stemsIterators.length - 1) {
          producedJoinResults += value.out;
        } else {
          producedIntermediateResults += value.out;
        }
      }
    }
    return { producedTuples, producedJoinResults, producedIntermediateResults };
  }

  private buildPlanSummary(): IPlanSummary {
    const orderSelectivities = this.stemsIterators.map(it => it.selectivitiesOrders);
    const fullOrderSize = this.stemsIterators.length;
    const fullOrders: IPlanStatistic[] = [];
    const coveredOrders: Set<string> = new Set();

    // First find all full orders in the execution information
    for (const [ i, orderSelectivity ] of orderSelectivities.entries()) {
      const fullSelectivities = this.extractOrderSignaturesAtSize(orderSelectivity, fullOrderSize);
      for (const fullSelectivity of fullSelectivities) {
        const orderPlanStatistic = this.createPlanStatistic(fullSelectivity, orderSelectivity, i);
        fullOrders.push(orderPlanStatistic);
        coveredOrders.add(fullSelectivity);
      }
    }

    // Backtrack through the order to get the order's characteristics
    for (const fullOrder of fullOrders) {
      this.backtrackOrder(fullOrder, orderSelectivities, coveredOrders);
    }

    // After summarising all full orders, we turn to partial orders that
    // never produced results.
    const partialOrders = [];
    for (let i = fullOrderSize - 1; i > 1; i--) {
      for (let j = 0; j < orderSelectivities.length; j++) {
        const partialSignatures = this.extractOrderSignaturesAtSize(orderSelectivities[j], i);
        for (const partialSignature of partialSignatures) {
          if (!coveredOrders.has(partialSignature)) {
            const orderPlanStatistic = this.createPlanStatistic(partialSignature, orderSelectivities[j], j);
            this.backtrackOrder(orderPlanStatistic, orderSelectivities, coveredOrders);
            partialOrders.push(orderPlanStatistic);
          }
        }
      }
    }
    return { fullOrders, partialOrders };
  }

  private backtrackOrder(
    fullOrder: IPlanStatistic,
    orderSelectivities: Record<string, ISelectivityData>[],
    coveredOrders: Set<string>,
  ): void {
    let orderArray = [ ...fullOrder.order ];
    while (orderArray.length > 1) {
      // Const precedingOrderString = this.backtrackOrder(fullOrder, orderSelectivities);
      // Get preceding operator
      const currentOperator = orderArray.at(-1)!;
      const selectivitiesCurrentOperator = orderSelectivities[currentOperator];

      // Create the preceding order array (the 'state' before the current step)
      const precedingOrder = orderArray.slice(0, -1);
      const precedingOrderString = precedingOrder.join(',');
      const selectivityOrder = selectivitiesCurrentOperator[precedingOrderString];
      fullOrder.totalIntermediate += selectivityOrder.in;

      fullOrder.orderElements[precedingOrderString] = {
        orderSignature: precedingOrder,
        in: selectivityOrder.in,
        out: selectivityOrder.out,
        selectivity: selectivityOrder.out / selectivityOrder.in,
      };
      coveredOrders.add(precedingOrderString);
      orderArray = precedingOrder;
    }
  }

  private createPlanStatistic(
    key: string,
    orderSelectivitiesOperator: Record<string, ISelectivityData>,
    idxOperator: number,
  ): IPlanStatistic {
    const orderAsArray = key.split(',').map(s => Number.parseInt(s.trim(), 10));
    orderAsArray.push(idxOperator);
    const inFullOrder = orderSelectivitiesOperator[key].in;
    const outFullOrder = orderSelectivitiesOperator[key].out;

    const orderPlanStatistic: IPlanStatistic = {
      order: orderAsArray,
      orderElements: {
        [key]: {
          orderSignature: orderAsArray,
          in: inFullOrder,
          out: outFullOrder,
          selectivity: outFullOrder / inFullOrder,
        },
      },
      totalIntermediate: inFullOrder,
      totalOutput: outFullOrder,
    };
    return orderPlanStatistic;
  }

  private extractOrderSignaturesAtSize(
    selectivities: Record<string, any>,
    size: number,
  ): string[] {
    return Object.keys(selectivities).filter((key) => {
      // 1. Split the key by comma to count the hops
      const hops = key.split(',');

      // A full order contains n - 1 indices, as the last is not included
      if (hops.length !== size - 1) {
        return false;
      }
      return true;
    });
  }
}

export const stemsContextKeys = {
  eddiesMetadata: new ActionContextKey<IStemsBindingsMetadata>('metadata'),
};

export interface ITimestampGenerator {
  next: () => number;
}

export class TimestampGenerator implements ITimestampGenerator {
  private counter = 0;

  public next(): number {
    return this.counter++;
  }
}

function _bitmaskToVector(doneMask: number, totalCount: number): number[] {
  const vector: number[] = [];
  // Iterate through bit positions 0 to N-1
  for (let i = 0; i < totalCount; i++) {
    // Shift the mask right by 'i' and check if the last bit is 1
    const bit = (doneMask >> i) & 1;
    vector.push(bit);
  }

  return vector;
}

export interface IStemsBindingsMetadata {
  /**
   * Bitmask representing which query operations (triple patterns / subqueries) have been completed.
   */
  done: number;
  /**
   * Arrival timestamp from TimestampGenerator of the original source tuple that initiated this result.
   */
  timestamp: number;
  /**
   * Order of operator indexes visited during the evaluation of this intermediate result.
   */
  order: number[];
  /**
   * Exclusion bitmask.
   * If this tuple branched down an alternative route targeting a Composite Resource (CR),
   * forbiddenBaseMask contains the bitmasks of the base operators replaced by that CR.
   * Any subsequent routing step whose target operator overlaps with this mask is pruned,
   * excluding competing overlapping CRs.
   */
  forbiddenBaseMask?: number;
  /**
   * Operator index of the most recently joined Composite Resource.
   * Used to enforce canonical plan symmetry breaking (Ascending CR Rule: next CR index > lastCrIndex),
   * preventing duplicate symmetric plan evaluation (e.g. CR1 ⋈ CR2 vs CR2 ⋈ CR1).
   */
  lastCrIndex?: number;
}
