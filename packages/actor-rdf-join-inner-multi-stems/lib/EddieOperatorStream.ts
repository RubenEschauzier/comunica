import type { BindingsStream } from '@comunica/types';
import type { Bindings } from '@comunica/utils-bindings-factory';

import type * as RDF from '@rdfjs/types';
import { BufferedIterator } from 'asynciterator';
import type { IEddieBindingsMetadata, ITimestampGenerator } from './EddieControllerStream';
import { eddiesContextKeys } from './EddieControllerStream';

/**
 * We may want to implement AMJoin, which keeps track of a bitvector table to quickly determine join failures,
 * also look into how to route eddies tuples
 */
export class EddieOperatorStream extends BufferedIterator<Bindings> {
  /**
   * The variables that this operator can do a join on
   */
  public joinVariables: RDF.Variable[][];
  
  /** 
   * Number of 'lottery' tickets. Used for lottery scheduling
   */
  public tickets: number = 0;
  /**
   * Selectivity information per 'done' signature
   */
  public selectivities: Record<number, ISelectivityData> = {};

  /**
   * The variables present in the bindings produced by this operator.
   */
  public variables: RDF.Variable[];

  private readonly sourceIterator: BindingsStream;

  private readonly tripleMap: Map<number, Bindings[]> = new Map();
  private readonly timestampGenerator: ITimestampGenerator;

  private readonly funHash: HashFunction;
  private readonly funJoin: JoinFunction;

  private readonly bitLoc: number;


  private match: Bindings | null = null;
  private matchMetadata: IEddieBindingsMetadata | null = null; 
  private matches: Bindings[] = [];
  private matchIdx = 0;

  public constructor(
    sourceIterator: BindingsStream,
    timestampGenerator: ITimestampGenerator,
    funHash: HashFunction,
    funJoin: JoinFunction,
    bitLoc: number,
    variables: RDF.Variable[],
    joinVariables: RDF.Variable[][],
  ) {
    super();

    this.variables = variables;
    this.joinVariables = joinVariables;

    this.funHash = funHash;
    this.funJoin = funJoin;

    this.sourceIterator = sourceIterator;

    if (this.sourceIterator.readable) {
      this.readable = true;
    }

    this.sourceIterator.on('readable', () => this.readable = true);
    this.sourceIterator.on('error', error => this.destroy(error));
    this.sourceIterator.on('end', () => this.emit('endRead'));

    this.timestampGenerator = timestampGenerator;

    this.bitLoc = bitLoc;
  }

  override _end() {
    super._end();
    this.sourceIterator.destroy();
  }

  push(item: IEddieJoinEntry) {
    // Push intermediate binding to buffer. We need to cast this to any
    // as the BufferedIterator does not know about the IEddieJoinEntry type.
    this._push(<any> item);
  }

  override read() {
    while (true) {
      if (this.ended) {
        return null;
      }

      while (this.matchIdx < this.matches.length) {
        const item = this.matches[this.matchIdx++];
        const itemMetadata = item.getContextEntry(eddiesContextKeys.eddiesMetadata)!;
        // const matchMetadata = this.match!.getContextEntry(eddiesContextKeys.eddiesMetadata)!;

        // If the timestamp of the interal index is newer (so larger)
        // than the intermediate result it is not a valid join candidate
        if (itemMetadata.timestamp > this.matchMetadata!.timestamp) {
          continue;
        }
        let result = this.funJoin(item, this.match!);
        if (result !== null) {
          // Produce result, so reduce ticket count
          this.tickets -= 1;
          // Produce result so increment out
          this.selectivities[this.matchMetadata!.done].out++;

          
          // Here we can just use this.match context entries as we simply are going to
          // set a bit entry to 1, this prevents extra work copying the object. Furthmore
          // as we throw away the intermediate results, mutating the metadata is not a problem
          // as we will never try to access it to get the previous metadata state.
          const copy = { ...this.matchMetadata! };
          copy.done |= 1 << this.bitLoc;
          copy.order = [ ...copy.order!, this.bitLoc ];

          result = result.setContextEntry(eddiesContextKeys.eddiesMetadata, copy);
          return result;
        }
      }

      let itemBuffer: IEddieJoinEntry | null = null;
      let item = null;
      let joinVars: RDF.Variable[] | undefined;
      // Read from buffer first as these are intermediate results
      itemBuffer = <IEddieJoinEntry | null> super.read();
      if (itemBuffer === null) {
        item = this.sourceIterator.read();
      } else {
        item = itemBuffer.item;
        joinVars = itemBuffer.joinVars;
      }

      if (this.done || item === null) {
        this.readable = false;
        return null;
      }

      // We read from the sourceIterator. In this case we need to hash for each
      // possible join variable.
      item = <Bindings> item;

      if (!joinVars) {
        const itemMetadata: IEddieBindingsMetadata = {
          done: 1 << this.bitLoc,
          timestamp: this.timestampGenerator.next(),
          order: [ this.bitLoc ],
        };

        item = item.setContextEntry(eddiesContextKeys.eddiesMetadata, itemMetadata);

        // For each variable we can join on, we hash the item and store it in the tripleMap.
        // This allows us to quickly find matches for any intermediate result and a given
        // join variable.
        for (const joinVar of this.joinVariables) {
          const hash = this.funHash(item, joinVar);
          if (!this.tripleMap.has(hash)) {
            this.tripleMap.set(hash, []);
          }
          const result = this.tripleMap.get(hash)!;
          result.push(item);
        }

        return item;
      }
      this.matchMetadata = item.getContextEntry(eddiesContextKeys.eddiesMetadata)! 
      // We read from the buffer and got variables to join on.
      // So hash on the join variables.
      const hash = this.funHash(item, joinVars);
      this.match = item;
      this.matches = this.tripleMap.get(hash) || [];
      this.matchIdx = 0;

      this.tickets += 1;
      // Update selectivity in counter
      const matchDone = this.matchMetadata.done;
      if (!this.selectivities[matchDone]){
        this.selectivities[matchDone] = {in: 0, out: 0}
      }
      this.selectivities[matchDone].in++;
    }
  }
}

export interface IEddieJoinEntry {
  item: Bindings;
  joinVars: RDF.Variable[];
}

export interface ISelectivityData {
  in: number;
  out: number;
}

export type HashFunction = (bindings: Bindings, variables: RDF.Variable[]) => number;
export type HashFunctionVariablesSet = (item: Bindings) => number;
export type JoinFunction = (...bindings: Bindings[]) => Bindings | null;
