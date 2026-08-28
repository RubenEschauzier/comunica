import type { AsyncIterator } from 'asynciterator';
import type * as RDF from '@rdfjs/types';
import type { Bindings, BindingsStream, IJoinEntryWithMetadata } from '@comunica/types';
import type { Algebra } from '@comunica/utils-algebra';
import { IStemsRouter, ITimestampGenerator, JoinFunction, StemsControllerStream, StemsOperatorStream } from '@comunica/actor-rdf-join-inner-multi-stems';
import { HashFunction } from '@comunica/bus-hash-bindings';
import { IAdaptiveJoinComponent } from './IAdaptiveJoinController';
import equal from 'deep-equal';


export interface IAdaptiveJoinComponentSteMsArgs {
  id: number | string;
  joinEntries: IJoinEntryWithMetadata[];
  joinVariablesEntries: RDF.Variable[][][];
  stemsControllerStream: StemsControllerStream;
  router: IStemsRouter;
  timestampGenerator: ITimestampGenerator;
  hashFn: HashFunction;
  joinFn: JoinFunction;
  dataFactory: RDF.DataFactory;
  metadata?: Record<string, any>;
}
/**
 * Wrapping class for managing stems executions and dynamically adding composite sources to
 * stems execution streams.
 */
export class StemsAdaptiveJoinComponent implements IAdaptiveJoinComponent {
  public readonly id: number | string;
  public readonly operations: Algebra.Operation[];
  public readonly stemsControllerStream: StemsControllerStream;
  public readonly router: IStemsRouter;
  protected readonly joinEntries: IJoinEntryWithMetadata[];
  protected readonly joinVariablesEntries: RDF.Variable[][][];
  protected readonly timestampGenerator: ITimestampGenerator;
  protected readonly hashFn: HashFunction;
  protected readonly joinFn: JoinFunction;
  protected readonly dataFactory: RDF.DataFactory;
  protected readonly metadata?: Record<string, any>;

  public constructor(args: IAdaptiveJoinComponentSteMsArgs) {
    this.id = args.id;
    this.joinEntries = args.joinEntries;
    this.operations = args.joinEntries.map(entry => entry.operation);
    this.joinVariablesEntries = args.joinVariablesEntries;
    this.stemsControllerStream = args.stemsControllerStream;
    this.router = args.router;
    this.timestampGenerator = args.timestampGenerator;
    this.hashFn = args.hashFn;
    this.joinFn = args.joinFn;
    this.dataFactory = args.dataFactory;
    this.metadata = args.metadata;
  }

  public get ended(): boolean {
    return this.stemsControllerStream.ended;
  }

  public canCoverOperations(operations: Algebra.Operation[]): boolean {
    return operations.every(targetOp =>
      this.operations.some(op => equal(op, targetOp))
    );
  }

  public addCompositeSource(
    operations: Algebra.Operation[],
    dataStream: AsyncIterator<Bindings>,
    metadata?: Record<string, any>,
  ): boolean {
    if (this.ended) {
      return false;
    }

    // Calculate bitmask for the operations answered by this source
    const matchedIndexes = this.operations.flatMap((op, idx) =>
      operations.some(targetOp => equal(op, targetOp)) ? [idx] : []
    );

    if (matchedIndexes.length === 0) {
      return false;
    }

    // Convert to binary representation
    const setBitsMask = matchedIndexes.reduce(
      (acc: number, curr: number) => acc + (1 << curr), 0
    );

    // Identify variables required by the component for joining with the rest of the graph
    const componentJoinVariables = this.computeJoinVariablesForSubset(matchedIndexes);



    let operatorVariables: RDF.Variable[];
    operatorVariables = this.extractVariablesFromOperations(operations);


    // Instantiate a new StemsOperatorStream with the component's internal generators
    const newOperatorIndex = this.stemsControllerStream.numOperators;
    const compositeOperation = operations.length === 1 ? operations[0] : this.createCompositeOperation(operations);

    const operator = new StemsOperatorStream(
      dataStream,
      this.timestampGenerator,
      this.hashFn,
      this.joinFn,
      newOperatorIndex,
      setBitsMask,
      compositeOperation,
      operatorVariables,
      // TODO: Determine the named nodes this can join on!
      [], // Named nodes
      componentJoinVariables,
      false,
    );

    // Attach the operator to the StemsControllerStream and recalculate routes
    this.stemsControllerStream.addOperator(operator, metadata);

    return true;
  }

  /**
   * Determines overlapping join variables between this composite source
   * and all other existing operators in the connected component.
   * The RDF.Variable[][] indicates 
   * (inner): an array of variables it can join on (multiple variables can join same time)
   * (outer): The different join variables for the other entries
   * Outer array order does not matter, just indicates which variables need to be hashed.
   */
  protected computeJoinVariablesForSubset(matchedIndexes: number[]): RDF.Variable[][] {
    const matchedSet = new Set(matchedIndexes);
    const overlappingVars: RDF.Variable[][] = [];
    for (let i = 0; i < this.joinEntries.length; i++) {
      if (!matchedSet.has(i)) {
        // Find variables shared between the matched subset and outside operator i
        const shared = this.joinVariablesEntries[i].flat();
        overlappingVars.push(shared);
      }
    }
    return overlappingVars;
  }

  protected extractVariablesFromOperations(operations: Algebra.Operation[]): RDF.Variable[] {
    const vars = new Map<string, RDF.Variable>();
    for (const op of operations) {
      if (op.type === 'pattern') {
        for (const v of this.extractPatternVariables(op as Algebra.Pattern)) {
          vars.set(v.value, v);
        }
      }
    }
    return Array.from(vars.values());
  }

  protected extractPatternVariables(pattern: Algebra.Pattern): RDF.Variable[] {
    const terms = [pattern.subject, pattern.predicate, pattern.object, pattern.graph];
    return terms.filter((term): term is RDF.Variable => term.termType === 'Variable');
  }

  protected createCompositeOperation(operations: Algebra.Operation[]): Algebra.Operation {
    return {
      type: 'bgp',
      patterns: operations.filter((op): op is Algebra.Pattern => op.type === 'pattern'),
    } as any;
  }
}