import { ActionContextKey } from '@comunica/core';
import type { Bindings } from '@comunica/utils-bindings-factory';

import { AsyncIterator } from 'asynciterator';
import type { EddieOperatorStream, ISelectivityData } from './EddieOperatorStream';
import type { IEddieRouter, IEddieRoutingEntry } from './routers/BaseRouter';
import { Logger } from '@comunica/types';

export class EddieControllerStream extends AsyncIterator<Bindings> {
  /**
   * The iterators performing the half-joins and producing triple pattern
   * matches
   */
  private readonly eddieIterators: EddieOperatorStream[];
  /**
   * The routing object constructing the routing table.
   */
  private readonly router: IEddieRouter;
  /**
   * Table indicating the next eddieIterator for a given 'done'
   * signature
   */
  private routingTable: Record<string, IEddieRoutingEntry[]>;
  /**
   * Flag indicating if the all triple patterns have no new matches.
   * If this flag is set, the iterator should finish all current intermediate
   * results and end.
   */
  private endTuples: boolean;
  /**
   * Loggers for debugging
   */
  private logger: Logger | undefined;
  private logContext: Record<string, any> | undefined;
  private loggedEndEvent: boolean = false;
  /**
   * Contains a list of snapshots at given number of tuples processed
   */
  public snapshots: Record<number, IAdaptivePlanStatistics> = {};
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

  public constructor(
    eddieIterators: EddieOperatorStream[],
    router: IEddieRouter,
    updateFrequency: number,
    logger?: Logger,
    logContext?: Record<string, any>,
  ) {
    super();
    this.eddieIterators = eddieIterators;
    this.finishedReading = Array.from<number>({ length: eddieIterators.length }).fill(0);
    this.router = router;
    this.routingUpdateFrequency = updateFrequency;
    this.routingTable = this.router.createRouteTable(this.eddieIterators.map(x => x.variables));

    this.endTuples = false;

    this.logger = logger;
    this.logContext = logContext;

    if (this.eddieIterators.some(it => it.readable)) {
      this.readable = true;
    }

    for (const [ index, it ] of this.eddieIterators.entries()) {
      it.on('readable', () => this.readable = true);
      it.on('endRead', () => {
        // When all eddiestreams finished creating new tuples,
        // we only need to process the remaining tuples in buffers
        this.finishedReading[index] = 1;
        if (this.finishedReading.every(val => val === 1)) {
          this.endTuples = true;
          const hasBufferedData = this.eddieIterators.some(op => op.readable && !op.ended);
          if (!hasBufferedData) {
            this._end();
          }
        }
      });
    }
    this.on('end', () => this._end());
  }

  public override _end(): void {
    if (!this.loggedEndEvent && this.logger && this.logContext){
      const finalStatistics = this.buildStatistics();
      const totalTimeUpdatingTable = Object.values(this.snapshots)
        .reduce((acc, snapshot) => acc + snapshot.timeToUpdateRouting!, 0);
      finalStatistics.timeToUpdateRouting = totalTimeUpdatingTable;

      this.snapshots[this.bindingsTotal] = finalStatistics;
      this.logger.debug(JSON.stringify(this.snapshots, null, 2), this.logContext);
      this.loggedEndEvent = true;
      console.log(finalStatistics);
    }
    for (const it of this.eddieIterators) {
      it.destroy();
    }
    super._end();
  }

  public override read(): null | Bindings {
    while (true) {
      let item: Bindings | null = null;
      let producedResults = false;
      for (const eddieIterator of this.eddieIterators) {
        item = eddieIterator.read();
        if (item === null) {
          continue;
        }
        this.bindingsSinceUpdate++;
        this.bindingsTotal++;

        producedResults = true;

        const partialResultMetadata = item.getContextEntry(eddiesContextKeys.eddiesMetadata)!;
        const nextEddie = this.routingTable[partialResultMetadata.done];

        // All done bits equal to 1
        if (nextEddie === undefined) {
          return item;
        }

        this.eddieIterators[nextEddie[0].next].push({ item, joinVars: nextEddie[0].joinVars });

        if (this.bindingsSinceUpdate >= this.routingUpdateFrequency) {
          const start = performance.now();
          this.routingTable = this.router.updateRouteTable(this.eddieIterators, this.routingTable);
          const end = performance.now();
          if (this.logger && this.logContext){
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

  public buildStatistics(): IAdaptivePlanStatistics{
    const { fullOrders, partialOrders } = this.buildPlanSummary();
    const {
      producedTuples,
      producedJoinResults,
      producedIntermediateResults
    } = this.getProducedTuplesSummary();
    const operatorSummaries = this.getOperatorSummary(producedTuples);
    const executionStatistics: IAdaptivePlanStatistics = {
        fullOrderStatistics: fullOrders,
        partialOrderStatistics: partialOrders,
        operatorSummaries: operatorSummaries,
        totalIntermediate: producedIntermediateResults,
        fanOut: producedIntermediateResults / producedJoinResults,
        reduction: producedIntermediateResults / producedTuples.reduce((acc, curr) => acc + curr, 0),
    }
    return executionStatistics;
  }

  private getOperatorSummary(producedTuples: number[]){
    const operatorSummaries: IStatsOperators[] = [];
    for (let i = 0; i < this.eddieIterators.length; i++){
      operatorSummaries.push({
        producedTuples: producedTuples[i],
        timeEmpty: this.eddieIterators[i].timeEmpty,
        timeNonEmpty: this.eddieIterators[i].timeNonEmpty,
        emptyReads: this.eddieIterators[i].nFailedReads,
        nonEmptyReads: this.eddieIterators[i].nSuccessReads
      })
    }
    return operatorSummaries
  }

  private getProducedTuplesSummary(){
    const producedTuples: number[] = Array(this.eddieIterators.length).fill(0);
    let producedJoinResults = 0;
    let producedIntermediateResults = 0;
    const orderSelectivities = this.eddieIterators.map(it => it.selectivitiesOrders);
    for (const orderSelectivity of orderSelectivities){
      for (const [key, value] of Object.entries(orderSelectivity)){
        const orderAsArray = key.split(',').map(s => parseInt(s.trim(), 10));
        // Singleton orders mean it only passed one operator: the one it was produced by
        if (orderAsArray.length === 1){
          // The in value denotes the number of tuples with that signature, which is equal
          // to the number of tuples the triple pattern associated with the operator 
          // produced
          producedTuples[orderAsArray[0]] += value.in;
        }
        // Last iterator isn't in the key signature as it is implied by the index of the 
        // eddies iterator
        if (orderAsArray.length === this.eddieIterators.length - 1){
          producedJoinResults += value.out;
        }
        // If its not a full result, the out value is always an intermediate result
        else {
          producedIntermediateResults += value.out
        }
      }
    }
    return { producedTuples, producedJoinResults, producedIntermediateResults };
  }

  private buildPlanSummary(){
    // TODO: This works for complete results, however it completely ignores incomplete results
    // where a tuple dies before it can reach the end and no other tuples are routed that way again.
    // TODO: Add also something that says how many each triple pattern have produced in 'build' mode.
    const orderSelectivities = this.eddieIterators.map(it => it.selectivitiesOrders);
    const fullOrderSize = this.eddieIterators.length;
    const fullOrders: IPlanStatistic[] = [];
    const coveredOrders: Set<string> = new Set();

    // First find all full orders in the execution information
    for (let i = 0; i < orderSelectivities.length; i++){
      const fullSelectivities = this.extractOrderSignaturesAtSize(orderSelectivities[i], fullOrderSize);
      for (const fullSelectivity of fullSelectivities){
        const orderPlanStatistic = this.createPlanStatistic(fullSelectivity, orderSelectivities[i], i);
        fullOrders.push(orderPlanStatistic)
        coveredOrders.add(fullSelectivity);
      }
    }

    // Backtrack through the order to get the order's characteristics
    for (const fullOrder of fullOrders){
      this.backtrackOrder(fullOrder, orderSelectivities, coveredOrders);
    }

    // After summarising all full orders, we turn to partial orders that
    // never produced results.
    const partialOrders = [];
    for (let i = fullOrderSize - 1; i > 1; i--){
      for (let j = 0; j < orderSelectivities.length; j++){
        // TODO: This goes wrong here!!
        const partialSignatures = this.extractOrderSignaturesAtSize(orderSelectivities[j], i);
        for (const partialSignature of partialSignatures){
          if (!coveredOrders.has(partialSignature)){
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
    coveredOrders: Set<string>
  ): void{
    let orderArray = [...fullOrder.order];
    while (orderArray.length > 1) {
      // const precedingOrderString = this.backtrackOrder(fullOrder, orderSelectivities);
      // Get preceding operator
      const currentOperator = orderArray[orderArray.length - 1];
      const selectivitiesCurrentOperator = orderSelectivities[currentOperator];

      //Create the preceding order array (the 'state' before the current step)
      const precedingOrder = orderArray.slice(0, orderArray.length - 1);
      const precedingOrderString = precedingOrder.join(',')
      const selectivityOrder = selectivitiesCurrentOperator[precedingOrderString];
      fullOrder.totalIntermediate += selectivityOrder.in;

      fullOrder.orderElements[precedingOrderString] = {
        orderSignature: precedingOrder,
        in: selectivityOrder.in,
        out: selectivityOrder.out,
        selectivity: selectivityOrder.out / selectivityOrder.in
      }
      coveredOrders.add(precedingOrderString);
      orderArray = precedingOrder; 
    }
  }

  private createPlanStatistic(key: string, orderSelectivitiesOperator: Record<string, ISelectivityData>, idxOperator: number){
    const orderAsArray = key.split(',').map(s => parseInt(s.trim(), 10));
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
          selectivity: outFullOrder/inFullOrder
        }
      },
      totalIntermediate: inFullOrder,
      totalOutput: outFullOrder,
    }
    return orderPlanStatistic;
  }


  private extractOrderSignaturesAtSize(
    selectivities: Record<string, any>, 
    size: number
  ): string[] {
    return Object.keys(selectivities).filter(key => {
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

export const eddiesContextKeys = {
  eddiesMetadata: new ActionContextKey<IEddieBindingsMetadata>('metadata'),
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

export interface IEddieBindingsMetadata {
  done: number;
  timestamp: number;
  order: number[];
}

/**
 * Interface for summary statistics of the current state of execution
 */
export interface IAdaptivePlanStatistics {
  /**
   * Selectivity information on full tuple routing orders, full meaning
   * all operators are included in it
   */
  fullOrderStatistics: IPlanStatistic[];
  /**
   * Selectivity information on partial tuple routing orders
   */
  partialOrderStatistics: IPlanStatistic[];
  /**
   * Operator specific statistics
   */
  operatorSummaries: IStatsOperators[];
  /**
   * Total number of intermediate results produced at a given time
   */
  totalIntermediate: number;
  /**
   * Ratio of the number of produced results compared to intermediate results
   * showing the amount of 'extra' work is done
   */
  fanOut: number;
  /**
   * Ratio of matching triples compared to intermediate results. Showing how quickly
   * selective joins are executed
   */
  reduction: number;
  /**
   * Time it took to update routing table. This can be seen as the cost of adaptive planning
   */
  timeToUpdateRouting?: number;
}

export interface IStatsOperators {
  producedTuples: number;
  timeEmpty: number;
  timeNonEmpty: number;
  emptyReads: number;
  nonEmptyReads: number;
}
/**
 * Interface denoting a given plan (or routing order) in the execution space and some statistics.
 */
export interface IPlanStatistic {
  order: number[];
  orderElements: Record<string, IOrderStatistic>;
  totalIntermediate: number;
  totalOutput: number
}

/**
 * Interface denoting statistics of a given order signature.
 */
export interface IOrderStatistic {
  orderSignature: number[]
  in: number;
  out: number;
  selectivity: number;
}