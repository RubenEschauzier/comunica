import { ActionContextKey } from '@comunica/core';
import type { Bindings } from '@comunica/utils-bindings-factory';

import { AsyncIterator } from 'asynciterator';
import type { EddieOperatorStream } from './EddieOperatorStream';
import type { IEddieRouter, IEddieRoutingEntry } from './routers/BaseRouter';
import { Logger } from '@comunica/types';

export class EddieControllerStream extends AsyncIterator<Bindings> {
  private readonly eddieIterators: EddieOperatorStream[];

  private readonly router: IEddieRouter;
  private routingTable: Record<string, IEddieRoutingEntry[]>;

  private endTuples: boolean;
  
  private logger: Logger | undefined;
  private logContext: Record<string, any> | undefined;
  private loggedEndEvent: boolean = false;
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
   * How often the routing table should be updated.
   */
  private readonly routingUpdateFrequency: number;
  /**
   * Object tracking what orders were used during query execution
   */
  private readonly orders: Record<string, number> = {};
  private nResults = 0;

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
      this.buildPlanSummary();
      // console.log(this.eddieIterators[0].selectivitiesOrders);
      this.loggedEndEvent = true;
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
        producedResults = true;

        const partialResultMetadata = item.getContextEntry(eddiesContextKeys.eddiesMetadata)!;
        const nextEddie = this.routingTable[partialResultMetadata.done];

        // Track orders used during query execution for debugging purposes
        const orderAsString = JSON.stringify(partialResultMetadata.order!);

        if (!this.orders[orderAsString]) {
          this.orders[orderAsString] = 0;
        }

        this.orders[orderAsString]++;

        // All done bits equal to 1
        if (nextEddie === undefined) {
          this.nResults++;
          return item;
        }

        this.eddieIterators[nextEddie[0].next].push({ item, joinVars: nextEddie[0].joinVars });

        if (this.bindingsSinceUpdate >= this.routingUpdateFrequency) {
          /**
           * TODO: Get current plan quality. So we map from # of tuples to the plan summary at that period
           */
          this.routingTable = this.router.updateRouteTable(this.eddieIterators, this.routingTable);
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

  private buildPlanSummary(){
    // TODO: This works for complete results, however it completely ignores incomplete results
    // where a tuple dies before it can reach the end and no other tuples are routed that way again.
    const orderSelectivities = this.eddieIterators.map(it => it.selectivitiesOrders);
    const fullOrderSize = this.eddieIterators.length;
    const fullOrders: IPlanStatistic[] = []
    // Start iterating over base of each order
    for (let i = 0; i < orderSelectivities.length; i++){
      const fullSelectivities = this.extractFullOrderSignatures(orderSelectivities[i], fullOrderSize);
      for (const fullSelectivity of fullSelectivities){
        const orderAsArray = fullSelectivity.split(',').map(s => parseInt(s.trim(), 10));
        const inFullOrder = orderSelectivities[i][fullSelectivity].in;
        const outFullOrder = orderSelectivities[i][fullSelectivity].out;

        const orderPlanStatistic: IPlanStatistic = {
          order: orderAsArray,
          orderElements: {
            [fullSelectivity]: {
              orderSignature: orderAsArray,
              in: inFullOrder,
              out: outFullOrder,
              selectivity: outFullOrder/inFullOrder
            }
          },
          totalIntermediate: inFullOrder,
          totalOutput: outFullOrder,
        }
        fullOrders.push(orderPlanStatistic)
      }
    }
    // Backtrack through the order to get the order's characteristics
    for (const fullOrder of fullOrders){
      console.log(fullOrder)
      // We update the preceding order and add current order.
      let orderArray = [...fullOrder.order];
      while (orderArray.length > 1) {
        // Get preceding operator
        const currentOperator = orderArray[orderArray.length - 1];
        const selectivitiesCurrentOperator = orderSelectivities[currentOperator];

        //Create the preceding order array (the 'state' before the current step)
        const precedingOrder = orderArray.slice(0, orderArray.length - 1);
        const precedingOrderString = precedingOrder.join(',')
        console.log(selectivitiesCurrentOperator)
        console.log(precedingOrderString)
        const selectivityOrder = selectivitiesCurrentOperator[precedingOrderString];
        fullOrder.totalIntermediate += selectivityOrder.in;

        fullOrder.orderElements[precedingOrderString] = {
          orderSignature: precedingOrder,
          in: selectivityOrder.in,
          out: selectivityOrder.out,
          selectivity: selectivityOrder.out / selectivityOrder.in
        }
        orderArray = precedingOrder; 
        console.log(`Full order updated:`)
        console.log(fullOrder)
      }
    }
    console.log("Selectivity informations from eddies:")
    console.log(orderSelectivities)
    console.log("OrderPlanStats")
    console.log(JSON.stringify(fullOrders, null, 2))
  }

  private extractFullOrderSignatures(
    selectivities: Record<string, any>, 
    totalSteMs: number
  ): string[] {
    return Object.keys(selectivities).filter(key => {
      // 1. Split the key by comma to count the hops
      const hops = key.split(',');

      // A full order contains n - 1 indices, as the last is not included
      if (hops.length !== totalSteMs - 1) {
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
 * Interface denoting a given plan in the execution space and some statistics.
 */
export interface IPlanStatistic {
  order: number[];
  orderElements: Record<string, IOrderStatistic>;
  totalIntermediate: number;
  totalOutput: number
}

export interface IOrderStatistic {
  orderSignature: number[]
  in: number;
  out: number;
  selectivity: number;
}