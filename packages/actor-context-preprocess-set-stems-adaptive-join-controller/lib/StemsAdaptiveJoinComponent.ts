import { AsyncIterator, UnionIterator } from 'asynciterator';
import { AsyncReiterableArray } from 'asyncreiterable';
import type * as RDF from '@rdfjs/types';
import type { Bindings, BindingsStream, IJoinEntryWithMetadata } from '@comunica/types';
import { Algebra, algebraUtils } from '@comunica/utils-algebra';
import { IStemsRouter, ITimestampGenerator, JoinFunction, StemsControllerStream, StemsOperatorStream } from '@comunica/actor-rdf-join-inner-multi-stems';
import { HashFunction } from '@comunica/bus-hash-bindings';
import { IAdaptiveJoinComponent } from './IAdaptiveJoinController';
import equal = require('deep-equal');

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
  protected readonly compositeSources: Map<number, AsyncReiterableArray<AsyncIterator<Bindings>>> = new Map();
  protected finalized = false;

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

  public finalize(): void {
    this.finalized = true;
    for (const sourcesArray of this.compositeSources.values()) {
      if (!sourcesArray.isEnded()) {
        sourcesArray.push(null);
      }
    }
  }

  public canCoverOperations(operations: Algebra.Operation[]): boolean {
    return operations.every(targetOp =>
      this.operations.some(op => equal(this.stripMetadata(op), this.stripMetadata(targetOp)))
    );
  }

  public stripMetadata(operation: Algebra.Operation){
    operation.metadata = undefined;
    return operation
  }

  public addCompositeSource(
    operations: Algebra.Operation[],
    dataStream: AsyncIterator<Bindings>,
    metadata?: Record<string, any>,
  ): boolean {
    if (this.ended || this.finalized) {
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

    // If an operator for this operation array already exists, push to its stream
    const existingSource = this.compositeSources.get(setBitsMask);
    if (existingSource) {
      if (!existingSource.isEnded()) {
        existingSource.push(dataStream);
        return true;
      }
      return false;
    }

    // Identify variables required by the component for joining with the rest of the graph
    const componentJoinVariables = this.computeJoinVariablesForSubset(matchedIndexes);
    const operatorVariables = this.extractVariablesFromOperations(operations);
    const operatorNamedNodes = this.extractSubjectNamedNodesFromOperations(operations);

    // Create dynamic source holder using AsyncReiterableArray and union iterator
    const sourcesArray = AsyncReiterableArray.fromInitialEmpty<AsyncIterator<Bindings>>();
    sourcesArray.push(dataStream);
    const unionStream = new UnionIterator(sourcesArray.iterator(), { autoStart: false });
    this.compositeSources.set(setBitsMask, sourcesArray);

    // Instantiate a new StemsOperatorStream with the component's internal generators
    const newOperatorIndex = this.stemsControllerStream.numOperators;

    const operator = new StemsOperatorStream(
      unionStream,
      this.timestampGenerator,
      this.hashFn,
      this.joinFn,
      newOperatorIndex,
      setBitsMask,
      operations,
      operatorVariables,
      operatorNamedNodes,
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
      algebraUtils.visitOperation(op, {
        [Algebra.Types.PATTERN]: {
          visitor: (pattern) => {
            const terms = [pattern.subject, pattern.predicate, pattern.object, pattern.graph];
            for (const term of terms) {
              if (term.termType === 'Variable') {
                vars.set(term.value, term);
              }
            }
          },
        },
        [Algebra.Types.PATH]: {
          visitor: (path) => {
            const terms = [path.subject, path.object, path.graph];
            for (const term of terms) {
              if (term.termType === 'Variable') {
                vars.set(term.value, term);
              }
            }
          },
        },
        [Algebra.Types.VALUES]: {
          visitor: (values) => {
            for (const v of values.variables) {
              vars.set(v.value, v);
            }
          },
        },
      });
    }
    return Array.from(vars.values());
  }

  protected extractSubjectNamedNodesFromOperations(operations: Algebra.Operation[]): RDF.NamedNode[] {
    const namedNodes = new Map<string, RDF.NamedNode>();
    for (const op of operations) {
      algebraUtils.visitOperation(op, {
        [Algebra.Types.PATTERN]: {
          visitor: (pattern) => {
            if (pattern.subject.termType === 'NamedNode') {
              namedNodes.set(pattern.subject.value, pattern.subject);
            }
          },
        },
        [Algebra.Types.PATH]: {
          visitor: (path) => {
            if (path.subject.termType === 'NamedNode') {
              namedNodes.set(path.subject.value, path.subject);
            }
          },
        },
      });
    }
    return Array.from(namedNodes.values());
  }
}


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
