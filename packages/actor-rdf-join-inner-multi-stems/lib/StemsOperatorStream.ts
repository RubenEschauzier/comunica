import type { BindingsStream } from '@comunica/types';
import { Algebra } from '@comunica/utils-algebra';
import type { Bindings } from '@comunica/utils-bindings-factory';

import type * as RDF from '@rdfjs/types';
import { BufferedIterator } from 'asynciterator';
import type { IStemsBindingsMetadata, ITimestampGenerator } from './StemsControllerStream';
import { stemsContextKeys } from './StemsControllerStream';
import { bitForIndex, mergeMasks } from './utils/BitUtils';
import { DelegatedPatternsFilter } from './filters/DelegatedPatternsFilter';
import { DelegatedPatternsFilterIndex } from './filters/DelegatedPatternsFilterIndex';

/**
 * We may want to implement AMJoin, which keeps track of a bitvector table to quickly determine join failures,
 * also look into how to route eddies tuples
 */
export class StemsOperatorStream extends BufferedIterator<Bindings> {
  /**
   * The underlying SPARQL algebra operations for this operator
   */
  public readonly operations: Algebra.Operation[];
  /**
   * The bitmask of operations satisfied by this operator
   */
  public readonly doneBitMask: number;
  /**
   * The variables that this operator can do a join on
   */
  public readonly joinVariables: RDF.Variable[][];
  /**
   * One index per join variable combination, in the order of joinVariables.
   */
  private readonly keyIndexes: IKeyIndex[];
  /**
   * The index that partitions the produced tuples most finely, i.e. the one whose key covers the
   * most variables.
   */
  private readonly mostSelectiveKeyIndex: IKeyIndex | undefined;
  /**
   * Per combination of join variables a probe arrives with, the indexes whose key lies within it,
   * most selective first. A probe uses the first one whose variables it binds.
   */
  private readonly probeIndexes: Map<string, IKeyIndex[]> = new Map();
  /**
   * Indicating if this operator might have to perform a cartesian join
   */
  public canBeCartesian: boolean;
  /**
   * Hash key for cartesian product bucket. This is an illegal variable name
   * to prevent collisions
   */
  public cartesianList: Bindings[] = [];
  /**
   * Number of 'lottery' tickets. Used for lottery scheduling
   */
  public tickets = 0;
  /**
   * Selectivity information per 'done' signature
   */
  public selectivitiesSignatures: Record<number, ISelectivityData> = {};
  /**
   * Selectivity information per 'order' signature
   */
  public selectivitiesOrders: Record<string, ISelectivityData> = {};

  /**
   * Time spent with empty input queue
   */
  public timeEmpty = 0;
  /**
   * Time spent with intermediate result in input queue
   */
  public timeNonEmpty = 0;

  private startTimeEmpty: number | undefined;
  private startTimeNonEmpty: number | undefined;
  /**
   * Number of reads with no result
   */
  public nFailedReads = 0;
  /**
   * Number of reads with result
   */
  public nSuccessReads = 0;
  /**
   * Number of tuples dropped because their source document a composite resource has claimed
   * authority over (see DelegatedPatternsFilter).
   */
  public nFilteredByAuthoritativeSource = 0;
  /**
   * Number of tuples pruned because every operator this composite resource covers had already
   * produced the matching part of them (see isCoveredByProducedBaseTuples). Only ever
   * incremented on a composite resource operator.
   */
  public nPrunedAsAlreadyCovered = 0;
  /**
   * The variables present in the bindings produced by this operator
   */
  public variables: RDF.Variable[];
  /**
   * The named nodes present in the bindings produced by this operator
   */
  public namedNodes: RDF.NamedNode[];

  private readonly sourceIterator: BindingsStream;

  private readonly timestampGenerator: ITimestampGenerator;

  private readonly funHash: HashFunction;
  private readonly funJoin: JoinFunction;

  /**
   * The index of this operator within the controller's operator list
   */
  public readonly operatorIndex: number;

  /**
   * Number of tuples produced by the triple pattern underlying this operator
   */
  private nProduced = 0;

  private match: Bindings | null = null;
  private matchMetadata: IStemsBindingsMetadata | null = null;
  private matches: Bindings[] = [];
  private matchIdx = 0;

  /**
   * The base operators whose join entries this operator covers, when it is a composite resource.
   * Undefined for a plain base operator. Assigned by StemsControllerStream#addOperator when the
   * operator is attached, since that is where the operator list this indexes into lives.
   */
  public coveredOperators: StemsOperatorStream[] | undefined;

  /**
   * If this operator covers multiple join entries in the execution
   */
  public get isCompositeResource(): boolean {
    return this.coveredOperators !== undefined;
  }

  protected readonly delegatedPatternFilters = new DelegatedPatternsFilterIndex();

  public constructor(
    sourceIterator: BindingsStream,
    timestampGenerator: ITimestampGenerator,
    funHash: HashFunction,
    funJoin: JoinFunction,
    operatorIndex: number,
    doneBitMask: number,
    operations: Algebra.Operation[],
    variables: RDF.Variable[],
    namedNodes: RDF.NamedNode[],
    joinVariables: RDF.Variable[][],
    canBeCartesian: boolean,
  ) {
    super();

    this.operations = operations;
    this.doneBitMask = doneBitMask;
    this.variables = variables;
    this.namedNodes = namedNodes;
    this.joinVariables = StemsOperatorStream.normalizeJoinVariables(joinVariables);
    this.keyIndexes = this.joinVariables.map(key => ({
      variables: key,
      names: key.map(variable => variable.value),
      buckets: new Map(),
      unbound: [],
    }));
    this.mostSelectiveKeyIndex = this.keyIndexes.length === 0 ?
      undefined :
      this.keyIndexes.reduce((best, current) => current.variables.length > best.variables.length ? current : best);
    this.canBeCartesian = canBeCartesian;

    this.funHash = funHash;
    this.funJoin = funJoin;

    this.sourceIterator = sourceIterator;
    
    // this.authoritativeSourceFilter = authoritativeSourceFilter;

    // Check if already ended before setting up listeners
    if (this.sourceIterator.done) {
      // Source is already exhausted, emit endRead immediately
      setImmediate(() => this.emit('endRead'));
    }

    if (this.sourceIterator.readable) {
      this.readable = true;
    }

    this.sourceIterator.on('readable', () => this.readable = true);
    this.sourceIterator.on('error', (error: Error) => this.destroy(error));
    this.sourceIterator.on('end', () => this.emit('endRead'));

    this.timestampGenerator = timestampGenerator;

    this.operatorIndex = operatorIndex;
  }

  /**
   * Sorts the variables of every join variable combination and removes duplicate combinations.
   *
   * The hash of a combination depends on the order of its variables, so a combination is always
   * hashed in one order, whichever order a probe names it in. Every combination gets an index of its
   * own, so an equal combination twice would index every tuple twice and join it twice.
   */
  private static normalizeJoinVariables(joinVariables: RDF.Variable[][]): RDF.Variable[][] {
    const seen: Set<string> = new Set();
    const normalized: RDF.Variable[][] = [];
    for (const joinVariable of joinVariables) {
      const sorted = [ ...joinVariable ].sort((a, b) => a.value < b.value ? -1 : a.value > b.value ? 1 : 0);
      const key = StemsOperatorStream.combinationKey(sorted.map(variable => variable.value));
      if (!seen.has(key)) {
        seen.add(key);
        normalized.push(sorted);
      }
    }
    return normalized;
  }

  private static combinationKey(names: string[]): string {
    // ',' cannot occur in a SPARQL variable name, so it is a safe separator
    return names.join(',');
  }

  private static bindsAll(binding: Bindings, variables: RDF.Variable[]): boolean {
    for (const variable of variables) {
      if (!binding.has(variable)) {
        return false;
      }
    }
    return true;
  }

  /**
   * Stores a produced tuple in every index: under its hash when it binds all of the index's
   * variables, and with the index's unbound tuples otherwise, which are compatible with any value
   * of the variables they leave unbound.
   */
  private storeInIndexes(item: Bindings): void {
    for (const index of this.keyIndexes) {
      if (!StemsOperatorStream.bindsAll(item, index.variables)) {
        index.unbound.push(item);
        continue;
      }
      const hash = this.funHash(item, index.variables);
      let bucket = index.buckets.get(hash);
      if (bucket === undefined) {
        bucket = [];
        index.buckets.set(hash, bucket);
      }
      bucket.push(item);
    }
  }

  /**
   * The stored tuples an intermediate result arriving with the given join variables can join with.
   *
   * The join variables a probe arrives with need not be one of this operator's indexed
   * combinations: in a cycle they span several, and a probe may leave some of them unbound. So
   * the probe uses the most selective index whose variables it binds, which holds every compatible
   * tuple either under the probe's hash or with its unbound tuples. Without such an index, every
   * stored tuple is a candidate. The join function settles compatibility on all variables.
   */
  private candidatesFor(item: Bindings, joinVars: RDF.Variable[]): Bindings[] {
    for (const index of this.probeIndexesFor(joinVars)) {
      if (StemsOperatorStream.bindsAll(item, index.variables)) {
        const bucket = index.buckets.get(this.funHash(item, index.variables)) ?? [];
        return index.unbound.length === 0 ? bucket : [ ...bucket, ...index.unbound ];
      }
    }
    if (this.keyIndexes.length === 0) {
      return this.canBeCartesian ? this.cartesianList : [];
    }
    // Every index holds every stored tuple once, under a hash or as unbound
    const index = this.keyIndexes[0];
    return [ ...index.buckets.values(), index.unbound ].flat();
  }

  private probeIndexesFor(joinVars: RDF.Variable[]): IKeyIndex[] {
    const names = joinVars.map(variable => variable.value);
    const key = StemsOperatorStream.combinationKey(names);
    let indexes = this.probeIndexes.get(key);
    if (indexes === undefined) {
      const probed = new Set(names);
      indexes = this.keyIndexes
        .filter(index => index.names.every(name => probed.has(name)))
        .sort((a, b) => b.variables.length - a.variables.length);
      this.probeIndexes.set(key, indexes);
    }
    return indexes;
  }

  /**
   * Whether every operator this composite resource covers has already produced the part of the
   * given binding that falls within that operator's own variables.
   */
  protected isCoveredByProducedBaseTuples(binding: Bindings): boolean {
    if (this.coveredOperators === undefined || this.coveredOperators.length === 0) {
      return false;
    }
    return this.coveredOperators.every(operator => operator.hasProducedBinding(binding));
  }

  /**
   * Whether this operator has already produced a tuple agreeing with the given binding on every
   * one of this operator's variables.
   *
   * A hash hit only establishes agreement on the hashed key, so the bucket it lands in still has
   * to be verified against this operator's full variable list.
   */
  public hasProducedBinding(binding: Bindings): boolean {
    // Without a join variable there is no hash key to look under, so fall back to the cartesian
    // bucket, which holds every produced tuple when the component has a cartesian product
    if (this.mostSelectiveKeyIndex === undefined) {
      return this.canBeCartesian && this.agreesOnVariables(this.cartesianList, binding);
    }
    // Agreeing requires both to bind every variable of this operator, so neither a binding that
    // leaves a key variable unbound nor a tuple stored as unbound can agree
    if (!StemsOperatorStream.bindsAll(binding, this.mostSelectiveKeyIndex.variables)) {
      return false;
    }
    const bucket = this.mostSelectiveKeyIndex.buckets
      .get(this.funHash(binding, this.mostSelectiveKeyIndex.variables));
    if (bucket === undefined) {
      return false;
    }
    return this.agreesOnVariables(bucket, binding);
  }

  private agreesOnVariables(candidates: Bindings[], binding: Bindings): boolean {
    return candidates.some(candidate => this.variables.every((variable) => {
      const candidateTerm = candidate.get(variable);
      const bindingTerm = binding.get(variable);
      return candidateTerm !== undefined && bindingTerm !== undefined && candidateTerm.equals(bindingTerm);
    }));
  }

  public override _end(): void {
    super._end();
    this.sourceIterator.destroy();
  }

  public push(item: IEddieJoinEntry): void {
    if ((<any> this)._buffer.length === 0) {
      const now = performance.now();

      // We are finishing an 'Empty' period. Record it.
      if (this.startTimeEmpty !== undefined) {
        this.timeEmpty += now - this.startTimeEmpty;
        this.startTimeEmpty = undefined;
      }

      // Start tracking NonEmpty time
      this.startTimeNonEmpty = now;
    }
    // Push intermediate binding to buffer. We need to cast this to any
    // as the BufferedIterator does not know about the IEddieJoinEntry type.
    this._push(<any> item);
  }

  public override read(): null | Bindings {
    if (!this.startTimeEmpty && !this.startTimeNonEmpty) {
      this.startTimeEmpty = performance.now();
    }
    while (true) {
      if (this.ended) {
        return null;
      }

      while (this.matchIdx < this.matches.length) {
        const item = this.matches[this.matchIdx++];
        const itemMetadata = item.getContextEntry(stemsContextKeys.stemsMetadata)!;
        // Const matchMetadata = this.match!.getContextEntry(eddiesContextKeys.eddiesMetadata)!;

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
          this.selectivitiesSignatures[this.matchMetadata!.done].out++;
          this.selectivitiesOrders[this.matchMetadata!.order.join(',')].out++;

          // Here we can just use this.match context entries as we simply are going to
          // set a bit entry to 1, this prevents extra work copying the object. Furthmore
          // as we throw away the intermediate results, mutating the metadata is not a problem
          // as we will never try to access it to get the previous metadata state.
          const copy: IStemsBindingsMetadata = { ...this.matchMetadata! };
          copy.done = mergeMasks(copy.done, this.doneBitMask);
          copy.order = [ ...copy.order, this.operatorIndex ];

          if (this.isCompositeResource) {
            copy.crMask = mergeMasks(copy.crMask, bitForIndex(this.operatorIndex));
            copy.lastCrIndex = this.operatorIndex;
            if (copy.pendingCrIndex === this.operatorIndex) {
              copy.pendingCrIndex = undefined;
            }
          }

          result = result.setContextEntry(stemsContextKeys.stemsMetadata, copy);
          return result;
        }
      }

      let itemBuffer: IEddieJoinEntry | null = null;
      let item: Bindings | null = null;
      let joinVars: RDF.Variable[] | undefined;
      
      // Read from buffer first as these are intermediate results
      itemBuffer = <IEddieJoinEntry | null> super.read();
      if (itemBuffer === null) {
        item = <Bindings> this.sourceIterator.read();
      } else {
        item = itemBuffer.item;
        joinVars = itemBuffer.joinVars;
        if ((<any> this)._buffer.length === 0) {
          const now = performance.now();

          // We are finishing a 'NonEmpty' period. Record it.
          if (this.startTimeNonEmpty !== undefined) {
            this.timeNonEmpty += now - this.startTimeNonEmpty;
            this.startTimeNonEmpty = undefined;
          }

          // Start tracking Empty time
          this.startTimeEmpty = now;
        }
      }

      if (this.done || item === null) {
        if (item === null) {
          this.nFailedReads++;
        }
        this.readable = false;
        return null;
      }
      this.nSuccessReads++;

      // We read from the sourceIterator. In this case we need to hash for each
      // possible join variable. And possible cartesian products
      if (joinVars === undefined) {
        const crMask = this.isCompositeResource ? bitForIndex(this.operatorIndex) : 0;
        if (this.delegatedPatternFilters.shouldFilter(<Bindings> item, this.doneBitMask, crMask)){
          this.nFilteredByAuthoritativeSource++;
          continue;
        }

        if (this.isCoveredByProducedBaseTuples(item)){
          this.nPrunedAsAlreadyCovered++;
          continue;
        }

        this.nProduced++;

        const itemMetadata: IStemsBindingsMetadata = {
          done: this.doneBitMask,
          timestamp: this.timestampGenerator.next(),
          order: [ this.operatorIndex ],
          crMask,
          lastCrIndex: this.isCompositeResource ? this.operatorIndex : undefined,
        };

        item = item.setContextEntry(stemsContextKeys.stemsMetadata, itemMetadata);

        // Index the item under every join variable combination, so any intermediate result can
        // quickly find its matches
        this.storeInIndexes(item);

        if (this.canBeCartesian) {
          this.cartesianList.push(item);
        }
        return item;
      }

      this.matchMetadata = item.getContextEntry(stemsContextKeys.stemsMetadata)!;
      this.match = item;
      this.matchIdx = 0;
      if (joinVars.length > 0) {
        // We read from the buffer and got variables to join on
        this.matches = this.candidatesFor(item, joinVars);
      } else {
        this.matches = this.cartesianList;
      }

      // Update selectivities in counter
      const matchDone = this.matchMetadata.done;
      if (!this.selectivitiesSignatures[matchDone]) {
        this.selectivitiesSignatures[matchDone] = { in: 0, out: 0 };
      }
      this.selectivitiesSignatures[matchDone].in++;

      const orderKey: string = this.matchMetadata.order.join(',');
      if (!this.selectivitiesOrders[orderKey]) {
        this.selectivitiesOrders[orderKey] = { in: 0, out: 0 };
      }
      this.selectivitiesOrders[orderKey].in++;
      this.tickets += 1;
    }
  }
  public addDelegatedPatternFilter(
    delegatedPatternsFilter: DelegatedPatternsFilter,
  ){
    this.delegatedPatternFilters.add(delegatedPatternsFilter);
  }
  // /**
  //  * Adds a composite resource filter for the given domain, from the terms that must reside
  //  * within it for a base tuple to already be covered by the composite resource.
  //  * this explicitly follows the authorativeness assumption of sources.
  // */
  // public addResourceFilter(domain: string, requiredAuthoritativeTerms: RDF.Term[]){
  //   // Filters should only be added when one triple term should be checked.
  //   // Otherwise no filter means all constant terms have already been checked 
  //   // -> filter all
  //   if (requiredAuthoritativeTerms.length === 0) {
  //     throw new Error(
  //       `addResourceFilter for domain '${domain}' was called with no terms to check; ` +
  //       `this operation must have at least one authoritative term position.`,
  //     );
  //   }


  //   const domainForTermCheck = domain.endsWith('/') ? domain.slice(0, -1) : domain;

  //   const requiredAuthoritativeVars: string[] = [];
  //   for (const term of requiredAuthoritativeTerms) {
  //     if (term.termType === 'Variable') {
  //       requiredAuthoritativeVars.push(term.value);
  //       continue;
  //     }
  //     if (term.termType !== 'NamedNode' ||
  //         !this.authoritativeSourceFilter.isWithinDomain(term.value, domainForTermCheck)) {
  //       return;
  //     }
  //     // Constant term already known to reside in the domain: dropped, so we
  //     // only check the source of binding is in domain
  //   }

  //   this.authoritativeSourceFilter.registerResourceFilter(
  //     domain,
  //     requiredAuthoritativeVars,
  //   )
  // }
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

/**
 * The tuples an operator produced, indexed on one combination of join variables.
 */
interface IKeyIndex {
  /**
   * The combination, sorted by variable name.
   */
  variables: RDF.Variable[];
  names: string[];
  /**
   * The tuples binding every variable of the combination, by their hash over it.
   */
  buckets: Map<number, Bindings[]>;
  /**
   * The tuples leaving a variable of the combination unbound.
   */
  unbound: Bindings[];
}
