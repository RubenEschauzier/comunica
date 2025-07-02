import { ActionContextKey } from '@comunica/core';
import type { Bindings } from '@comunica/utils-bindings-factory';

import { AsyncIterator } from 'asynciterator';
import type { EddieOperatorStream } from './EddieOperatorStream';
import type { IEddieRouter, IEddieRoutingEntry } from './EddieRouters';

export class EddieControllerStream extends AsyncIterator<Bindings> {
  private readonly eddieIterators: EddieOperatorStream[];

  private readonly router: IEddieRouter;
  private readonly routingTable: Record<string, IEddieRoutingEntry[]>;

  private endTuples: boolean;
  private finishedReading: number[];

  public constructor(
    eddieIterators: EddieOperatorStream[],
    router: IEddieRouter,
  ) {
    super();
    this.eddieIterators = eddieIterators;
    this.finishedReading = Array.from<number>({ length: eddieIterators.length }).fill(0);
    this.router = router;
    this.endTuples = false;

    this.routingTable = this.router.createRouteTable(this.eddieIterators.map(x => x.variables));

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
  }

  override _end() {
    super._end();
    for (const it of this.eddieIterators) {
      it.destroy();
    }
  }

  override read() {
    while (true) {
      let item: Bindings | null = null;
      let producedResults = false;
      for (let i = 0; i < this.eddieIterators.length; i++) {
        item = this.eddieIterators[i].read();

        if (item === null) {
          continue;
        }
        producedResults = true;

        const nextEddie = this.routingTable[item.getContextEntry(eddiesContextKeys.eddiesMetadata)!.done];
        // All done bits equal to 1
        if (nextEddie === undefined) {
          return item;
        }
        this.eddieIterators[nextEddie[0].next].push({ item, joinVars: nextEddie[0].joinVars });
      }

      // If none of the eddieIterators have a match for us
      // this stream is not readable.
      if (this.endTuples && item === null) {
        this._end();
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

  next(): number {
    return this.counter++;
  }
}

export interface IEddieBindingsMetadata {
  done: number;
  timestamp: number;
  order?: number[];
}
