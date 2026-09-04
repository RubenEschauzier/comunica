import { AsyncIterator, UnionIterator } from 'asynciterator';
import { AsyncReiterableArray } from 'asyncreiterable';
import type * as RDF from '@rdfjs/types';
import type { Bindings, IJoinEntryWithMetadata } from '@comunica/types';
import { Algebra, algebraUtils } from '@comunica/utils-algebra';
import { IStemsRouter, ITimestampGenerator, JoinFunction, StemsControllerStream, StemsOperatorStream, computePairwiseJoinVariables, indexesToMask } from '@comunica/actor-rdf-join-inner-multi-stems';
import { HashFunction } from '@comunica/bus-hash-bindings';
import { IAdaptiveJoinComponent } from './IAdaptiveJoinController';
import equal = require('deep-equal');
import { AuthoritativeSourceFilter } from '@comunica/actor-rdf-join-inner-multi-stems';
import { KeysMergeBindingsContext } from '@comunica/context-entries';

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

    // Map each operation answered by this source to the index of the component operation
    // it covers (-1 when the component does not contain it). The operations are compared
    // structurally, as the algebra objects handed to us are not reference-identical to the
    // ones the join entries were built from.
    const operationToOperatorIndex = operations.map(targetOp =>
      this.operations.findIndex(op => equal(this.stripMetadata(op), this.stripMetadata(targetOp))));

    // Calculate bitmask for the operations answered by this source
    const matchedIndexes = [ ...new Set(operationToOperatorIndex.filter(idx => idx !== -1)) ].sort();

    if (matchedIndexes.length === 0) {
      return false;
    }

    // Convert to binary representation
    const setBitsMask = indexesToMask(matchedIndexes);

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

    const authoritativeSourceFilter = new AuthoritativeSourceFilter(
      (binding: Bindings) => (<any>binding).getContextEntry(KeysMergeBindingsContext.sourcesBinding) ?? [],
    );

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
      authoritativeSourceFilter,
    );

    // Attach the operator to the StemsControllerStream and recalculate routes.
    // The passed metadata contains mappings from operators to their extraction variables,
    // extended with the operator indexes covered by this source so the controller does not
    // have to rediscover them from the algebra.
    this.stemsControllerStream.addOperator(operator, {
      ...metadata,
      operationToOperatorIndex,
    });

    return true;
  }

  /**
   * Determines overlapping join variables between this composite source
   * and all other existing operators in the connected component.
   * The RDF.Variable[][] indicates
   * (inner): an array of variables it can join on (multiple variables can join same time)
   * (outer): The different join variables for the other entries
   * Outer array order does not matter, just indicates which variables need to be hashed.
   *
   * This reuses the same pairwise-intersection logic ActorRdfJoinMultiStems#getJoinVariables
   * uses for base operators: the covered entries are merged into a single variable set
   * representing the composite resource, which is then compared against every entry it does
   * not cover, exactly as if the composite resource were one join entry among the rest.
   */
  protected computeJoinVariablesForSubset(matchedIndexes: number[]): RDF.Variable[][] {
    const matchedSet = new Set(matchedIndexes);
    const compositeVariables = new Map<string, RDF.Variable>();
    const uncoveredVariableSets: RDF.Variable[][] = [];

    for (let i = 0; i < this.joinEntries.length; i++) {
      const entryVariables = this.joinEntries[i].metadata.variables.map(x => x.variable);
      if (matchedSet.has(i)) {
        for (const variable of entryVariables) {
          compositeVariables.set(variable.value, variable);
        }
      } else {
        uncoveredVariableSets.push(entryVariables);
      }
    }

    // The composite resource is placed at index 0, so its row in the result is exactly its
    // intersection with every uncovered entry, which are the only other participants
    const [ compositeJoinVariables ] = computePairwiseJoinVariables(
      [ [ ...compositeVariables.values() ], ...uncoveredVariableSets ],
    );
    return compositeJoinVariables;
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
  stemsControllerStream: StemsControllerStream;
  router: IStemsRouter;
  timestampGenerator: ITimestampGenerator;
  hashFn: HashFunction;
  joinFn: JoinFunction;
  dataFactory: RDF.DataFactory;
  metadata?: Record<string, any>;
}
