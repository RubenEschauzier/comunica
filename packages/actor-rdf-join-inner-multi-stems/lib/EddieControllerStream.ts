import { ActionContextKey } from '@comunica/core';
import type { Bindings } from '@comunica/utils-bindings-factory';

import { AsyncIterator } from 'asynciterator';
import type { EddieOperatorStream } from './EddieOperatorStream';
import type { IEddieRouter, IEddieRoutingEntry } from './routers/BaseRouter';

export class EddieControllerStream extends AsyncIterator<Bindings> {
  private readonly eddieIterators: EddieOperatorStream[];

  private readonly router: IEddieRouter;
  private routingTable: Record<string, IEddieRoutingEntry[]>;

  private endTuples: boolean;
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
  ) {
    super();
    this.eddieIterators = eddieIterators;
    this.finishedReading = Array.from<number>({ length: eddieIterators.length }).fill(0);
    this.router = router;
    this.routingUpdateFrequency = updateFrequency;
    this.routingTable = this.router.createRouteTable(this.eddieIterators.map(x => x.variables));

    this.endTuples = false;

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
  order?: number[];
}
