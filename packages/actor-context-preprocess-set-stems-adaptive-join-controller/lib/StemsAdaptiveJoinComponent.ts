import type { AsyncIterator } from 'asynciterator';
import type * as RDF from '@rdfjs/types';
import type { Bindings, BindingsStream, IJoinEntryWithMetadata } from '@comunica/types';
import type { Algebra } from '@comunica/utils-algebra';
import { IStemsRouter, ITimestampGenerator, JoinFunction, StemsControllerStream } from '@comunica/actor-rdf-join-inner-multi-stems';
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

  public addCompositeSource<T extends Bindings | RDF.Quad>(
    operations: Algebra.Operation[],
    dataStream: AsyncIterator<T>,
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


    // TODO: Determine if this is correct, all AI generated
    

    // 3. Adapt input stream: if QuadStream, adapt based on join requirements; if BindingsStream, use directly
    let finalBindingsStream: BindingsStream;
    let operatorVariables: RDF.Variable[];
    if (this.isQuadStream(dataStream)) {
      const starPatterns = operations.filter((op): op is Algebra.Pattern => op.type === 'pattern');
      const adapted = this.adaptQuadStreamToBindings(dataStream, starPatterns, componentJoinVariables);
      finalBindingsStream = adapted.stream;
      operatorVariables = adapted.variables;
    } else {
      finalBindingsStream = <BindingsStream><any>dataStream;
      operatorVariables = this.extractVariablesFromOperations(operations);
    }

    // 4. Instantiate a new StemsOperatorStream with the component's internal generators
    const newOperatorIndex = (this.stemsControllerStream as any).eddieIterators.length;
    const compositeOperation = operations.length === 1 ? operations[0] : this.createCompositeOperation(operations);
    const operator = new (this.stemsControllerStream as any).StemsOperatorStream(
      finalBindingsStream,
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
      false, // canBeCartesian
    );

    // 5. Attach the operator to the StemsControllerStream and recalculate routes
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

  protected isQuadStream<T>(stream: AsyncIterator<T>): stream is AsyncIterator<T & RDF.Quad> {
    // Quick runtime check: quads have 'subject', 'predicate', 'object'
    const property = (stream as any).property;
    return typeof property === 'function' && !property('variables');
  }
  /**
   * Adapts an incoming QuadStream into a BindingsStream, performing minimal extraction
   * based on the variables actually needed for joins.
   */
  protected adaptQuadStreamToBindings(
    quadStream: AsyncIterator<RDF.Quad>,
    patterns: Algebra.Pattern[],
    joinVariables: RDF.Variable[][],
  ): { stream: BindingsStream; variables: RDF.Variable[] } {
    const requiredJoinVarNames = new Set(joinVariables.flat().map(v => v.value));
    // Determine which patterns in the star contain variables we actually need to join on
    const relevantPatterns = patterns.filter(p =>
      (p.subject.termType === 'Variable' && requiredJoinVarNames.has(p.subject.value)) ||
      (p.object.termType === 'Variable' && requiredJoinVarNames.has(p.object.value))
    );
    // If only one predicate is required for joining, stream without buffering
    if (relevantPatterns.length <= 1) {
      const targetPattern = relevantPatterns[0] ?? patterns[0];
      return {
        stream: <BindingsStream><any>quadStream.map(quad => this.quadToBinding(quad, targetPattern)),
        variables: this.extractPatternVariables(targetPattern),
      };
    }
    // Otherwise, collect star properties per subject
    return {
      stream: <BindingsStream><any>new StarQuadsToBindingsIterator(quadStream, relevantPatterns, this.dataFactory),
      variables: Array.from(requiredJoinVarNames).map(name => this.dataFactory.variable(name)),
    };
  }

  protected quadToBinding(quad: RDF.Quad, pattern: Algebra.Pattern): any {
    const entries: [RDF.Variable, RDF.Term][] = [];
    if (pattern.subject.termType === 'Variable') {
      entries.push([pattern.subject, quad.subject]);
    }
    if (pattern.object.termType === 'Variable') {
      entries.push([pattern.object, quad.object]);
    }
    return new Map(entries);
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

// TODO: Is this too much??? Is this needed?

class StarQuadsToBindingsIterator extends (require('asynciterator').TransformIterator) {
  private readonly subjectBuffer: Map<string, Map<string, RDF.Term>> = new Map();
  public constructor(
    source: AsyncIterator<RDF.Quad>,
    protected readonly targetPatterns: Algebra.Pattern[],
    protected readonly dataFactory: RDF.DataFactory,
  ) {
    super(source);
  }
  protected override _transform(quad: RDF.Quad, done: () => void): void {
    const subjectKey = quad.subject.value;
    let entry = this.subjectBuffer.get(subjectKey);
    if (!entry) {
      entry = new Map();
      this.subjectBuffer.set(subjectKey, entry);
    }
    for (const pattern of this.targetPatterns) {
      if (pattern.predicate.value === quad.predicate.value && pattern.object.termType === 'Variable') {
        entry.set(pattern.object.value, quad.object);
      }
      if (pattern.subject.termType === 'Variable') {
        entry.set(pattern.subject.value, quad.subject);
      }
    }
    // Emit if all required variables for this star subject have been accumulated
    if (this.targetPatterns.every(p => p.object.termType !== 'Variable' || entry!.has(p.object.value))) {
      this._push(entry);
      this.subjectBuffer.delete(subjectKey);
    }
    done();
  }
  protected override _flush(done: () => void): void {
    this.subjectBuffer.clear();
    done();
  }
}
