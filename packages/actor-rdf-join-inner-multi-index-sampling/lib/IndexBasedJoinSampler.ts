import * as fs from 'node:fs';
import type * as RDF from '@rdfjs/types';
import { Store, Parser } from 'n3';
import type { Operation } from 'sparqlalgebrajs/lib/algebra';
import { FifoQueue } from './FifoQueue';
import { Term } from '@comunica/expression-evaluator/lib/expressions';
import { KeysCore } from '@comunica/context-entries';

/**
 * A comunica Inner Multi Index Sampling RDF Join Actor.
 */
export class IndexBasedJoinSampler {
  public budget: number;

  public constructor(budget: number) {
    this.budget = budget;
  }

  public run(n: number, store: any, arrayIndex: Record<string, ArrayIndex>, tps: Operation[]) {
    // Get all triples matching triple pattern
    const sampledTriplePatterns = this.sampleTriplePatternsQuery(Number.POSITIVE_INFINITY, store, arrayIndex, tps);
    this.bottomUpEnumeration(n, store, arrayIndex, sampledTriplePatterns, tps);
  }

  public bottomUpEnumeration(n: number, store: any, arrayIndex: Record<string, ArrayIndex>, resultSets: Record<string,string>[][], tps: Operation[]) {
    // Samples are stored by their two participating expressions like [[1], [2]] joining result set
    // 1 and 2, while [[1,2],[3]] joins result sets of join between 1,2 and result set of 3.
    const samples: Record<string, ISampleResult> = this.initializeSamplesEnumeration(resultSets);

    const problemQueue = new FifoQueue<number[][]>();
    this.joinCombinations(tps.length).map(join => problemQueue.enqueue(join));

    while (!problemQueue.isEmpty()) {
      // The combination of triple patterns we're evaluating
      const combination = problemQueue.dequeue()!;
      // The intermediate result set we use as input sample for join sampling
      const baseSample = samples[JSON.stringify(combination[0])];
      // The triple patterns already joined to produce the intermediate result set
      const tpInBaseSample = combination[0].map(x => tps[x]);
      // The triple pattern we join with intermediate result set
      const spo = this.tpToIds(tps[combination[1][0]], store);
      const joinVariable = this.determineJoinVariable(tpInBaseSample, tps[combination[1][0]]);
      if (joinVariable.joinLocation === 'c') {
        // Cross product so disregard this
        // TODO Circle back to cross products if we have exhausted all non-cross products
        continue;
      }
      const samplingOutput: ISampleOutput = this.sampleJoin(n, baseSample.sample, 
        spo[0], spo[1], spo[2], tps[combination[1][0]],
        joinVariable.joinVariable, joinVariable.joinLocation, arrayIndex);
      /**
       * Update samples by multiplying the base sample estimated cardinality with the estimated 
       * selectivity
       */
      samples[JSON.stringify(combination)] = {
        sample: samplingOutput.sample, 
        estimatedCardinality: samplingOutput.selectivity * baseSample.estimatedCardinality 
      };
      // Generate new combinations and add to queue
      this.generateNewCombinations(combination, tps.length).map(x => problemQueue.enqueue(x));
    }
    return samples;
  }

  public sampleJoin(n: number, samples: Record<string, string>[], 
    s: string | null, p: string | null, o: string | null, operation: Operation,
    joinVariable: string, joinLocation: 's'|'p'|'o', arrayIndex: Record<string, ArrayIndex>
  ): ISampleOutput {
    const operationTriplePattern: any[] = [operation.subject, 
      operation.predicate, operation.object];
    // First count the number of possible join candidates with sample
    const { counts, sampleRelations } = this.candidateCounts(samples, 
      s, p, o, joinLocation, joinVariable, arrayIndex
    );
    const sum: number = counts.reduce((acc, count) => acc += count, 0);
    const ids = this.sampleArray([ ...new Array(sum).keys() ], n);
    const joinSample: Record<string, string>[] = [];
    for (const id of ids) {
      let searchIndex = 0;
      for (const [ i, count ] of counts.entries()) {
        searchIndex += count;
        // If count > id we take the joins associated with this sample
        if (id <= searchIndex) {
          // Go back one step
          searchIndex -= count;
          // Find index
          const tripleIndex = id - (searchIndex) - 1;
          console.log(tripleIndex);
          console.log(this.lookUpIndex(
            arrayIndex[''], sampleRelations[i][0], sampleRelations[i][1], sampleRelations[i][2]
          ))
          const chosenTriple = this.lookUpIndex(
            arrayIndex[''], sampleRelations[i][0], sampleRelations[i][1], sampleRelations[i][2]
          )[tripleIndex];
          console.log(chosenTriple)
          const associatedSample = {... samples[i]};
          for (const term of chosenTriple){
            if (term === null){

            }
          }
          // Convert the variables in triple to binding
          chosenTriple.forEach((curr, k) => {
            if (this.isVariable(operationTriplePattern[k])){
              if(!associatedSample[operationTriplePattern[k].value]){
                associatedSample[operationTriplePattern[k].value] = curr;
              }
            }
          });
          joinSample.push(associatedSample);
          break;
        }
      }
    }
    const selectivity = sum / samples.length;
    return {
      sample: joinSample,
      selectivity,
    };
  }

  public candidateCounts(samples: Record<string, string>[], s: string | null, p: string | null, o: string | null,
    joinLocation: 's' | 'p' | 'o', joinVariable: string, arrayIndex: Record<string, ArrayIndex>
  ): ICandidateCounts {
    const counts: number[] = []; 
    const sampleRelations: (string | null)[][] = [];
    for (const sample of samples) {
      const sampleRelation = this.setSampleRelation(sample, s, p, o, 
        joinLocation, joinVariable);
        

      const candidateCounts: number[] = this.countTriplesJoinRelation(
        arrayIndex[''], sampleRelation[0], sampleRelation[1], sampleRelation[2]
      );
      const totalCountSample = candidateCounts.reduce((acc, count) => acc += count, 0);
      counts.push(totalCountSample);
      sampleRelations.push(sampleRelation);
    }
    return {
      counts,
      sampleRelations
    }
  }

  public sampleTriplePatternsQuery(n: number, store: any, arrayIndex: Record<string, ArrayIndex>, tps: Operation[]): 
    Record<string, string>[][] {
    const sampledTriplePatterns = tps.map(tp => {
      const spo: (null | string)[] = this.tpToIds(tp, store);
      const triplePatternKeys = Object.keys(tp);
      const triples = this.sampleIndex(spo[0], spo[1], spo[2], n, arrayIndex['']);
      
      return triples.map(triple => 
        triple.reduce((binding, curr, i) => {
          if (!spo[i]) {
            binding[tp[triplePatternKeys[i]].value] = curr;
          }
          return binding;
        }, {} as Record<string, string>)
      );
    });
    return sampledTriplePatterns;
  }

  // Take a random sample of triples matching a pattern out of an index
  public sampleIndex(s: string | null, p: string | null, o: string | null, n: number, arrayIndex: ArrayIndex) {
    // TODO check performance impact and optimize. This materializes all triples and then samples, will we could instead
    // count the triples and based on counts do random lookup without materializing all of them.
    const triples = this.lookUpIndex(arrayIndex, s, p, o);
    return this.sampleArray(triples, n);
  }

  public countTriplesJoinRelation(arrayIndex: ArrayIndex, s: string | null, p: string | null, o: string | null): number[] {
    const subTotalCandidateCounts: number[] = [];
    const nVariables = [ s, p, o ].reduce((total: number, term) => term === null ? total + 1 : total, 0);
    const indexToUse = this.determineOptimalIndex(s, p, o);

    // If no variables, return the number of triples that match
    if (nVariables === 0){
      const objsArray = this.pollIndex(arrayIndex, s, p, o, indexToUse);
      // If we find objects matching s, p
      if (objsArray) {
        const setObjs = new Set(objsArray);
        // If the passed o matches the objects matchin s,p return the triples
        if (setObjs.has(<string> o)) {
          return [1];
        }
      }
      // Else triple doesn't exist in index
      return [0];
    }
    // For one variable in triple pattern return size of array index of triple pattern
    if (nVariables === 1) {
      const indexArray = this.pollIndex(arrayIndex, s, p, o, indexToUse);
      // If no index, we have no results for this part of sample
      subTotalCandidateCounts.push(indexArray.length);
    }
    // Two variables we need to iterate over two layers of index
    else if (nVariables === 2) {
      const firstTerms = this.pollIndex(arrayIndex, s, p, o, indexToUse);
      if (firstTerms.length === 0){
        return [0];
      }
      for (const term of firstTerms) {
        // Determine the type of term we retrieve
        if (indexToUse == 'subjects') {
          // S P o index
          p = term;
        }
        if (indexToUse == 'predicates') {
          // P O s index
          o = term;
        }
        if (indexToUse == 'objects') {
          // O S p
          s = term;
        }
        const secondTerms = this.pollIndex(arrayIndex, s, p, o, indexToUse);
        subTotalCandidateCounts.push(secondTerms.length);
      }
    } 
    return subTotalCandidateCounts;
  }

  public lookUpIndex(arrayIndex: ArrayIndex, s: string | null, p: string | null, o: string | null) {
    // If we lookup a full triple, if it exists return that triple
    if (s && p && o) {
      // Abuse of the pollIndex function as this shouldn't return undefined, but in this case can
      const objs = this.pollIndex(arrayIndex, s, p, o, 'subjects');
      // If we find objects matching s, p
      if (objs) {
        const setObjs = new Set(objs);
        // If the passed o matches the objects matchin s,p return the triples
        if (setObjs.has(o)) {
          return [[ s, p, o ]];
        }
      }
      // Else its an invalid triple
      return [];
    }
    if (s) {
      if (p) {
        const objs = this.pollIndex(arrayIndex, s, p, o, 'subjects');
        const triples = objs.map(x => [ s, p, x ]);
        return triples;
      }
      if (o) {
        const preds = this.pollIndex(arrayIndex, s, p, o, 'objects');
        const triples = preds.map(x => [ s, x, o ]);
        return triples;
      }

      // If only subject is set, we first get all predicates with subject, then get all objects
      // for each subject, predicate combination
      let triples: string[][] = [];
      const preds = this.pollIndex(arrayIndex, s, p, o, 'subjects');
      for (const pred of preds) {
        const objs = this.pollIndex(arrayIndex, s, pred, o, 'subjects');
        triples = [ ...triples, ...objs.map(x => [ s, pred, x ]) ];
      }
      return triples;
    }
    if (p) {
      if (o) {
        const subs = this.pollIndex(arrayIndex, s, p, o, 'predicates');
        const triples = subs.map(x => [ x, p, o ]);
        return triples;
      }

      const objs = this.pollIndex(arrayIndex, s, p, o, 'predicates');
      let triples: string[][] = [];
      for (const obj of objs) {
        const subs = this.pollIndex(arrayIndex, s, p, obj, 'predicates');
        triples = [ ...triples, ...subs.map(x => [ x, p, obj ]) ];
      }
      return triples;
    }
    // This case only o is defined
    if (o) {
      let triples: string[][] = [];
      const subs = this.pollIndex(arrayIndex, s, p, o, 'objects');
      for (const sub of subs) {
        const preds = this.pollIndex(arrayIndex, sub, p, o, 'objects');
        triples = [ ...triples, ...preds.map(x => [ sub, x, o ]) ];
      }
      return triples;
    }

    throw new Error('Not yet implemented lookup up all triples');
  }

  public pollIndex(arrayIndex: ArrayIndex, s: string | null, p: string | null, o: string | null, indexToUse: 'subjects' | 'predicates' | 'objects'): string[] {
    const index = arrayIndex[indexToUse];
    // When an element of spo is undefined, we don't have the element in the store, thus the triple pattern will
    // always be empty
    if (s === undefined || p === undefined || o === undefined) {
      return [];
    }

    if (s === null && p === null && o === null) {
      throw new Error('Not yet implemented index poll for three variables');
    }

    if (indexToUse == 'subjects') {
      // If subject given doesn't exist in subject return empty array
      if (index[s!] === undefined) {
        return [];
      }
      // Object will always be null when using subject index
      if (p) {
        // Return triples matching subject - predicate or if predicate not in index return empty
        return index[s!][p] ? index[s!][p] : [];
      }
      // Return the predicates that match subject
      return Object.keys(index[s!]);
    }

    if (indexToUse == 'predicates') {
      if (index[p!] === undefined) {
        return [];
      }
      // Return subjects that match predicate - object
      if (o) {
        return index[p!][o] ? index[p!][o] : [];
      }
      // Return objects that match predicate
      return Object.keys(index[p!]);
    }
    if (indexToUse == 'objects') {
      if (index[o!] === undefined) {
        return [];
      }
      // Return predicates that match subjects - objects
      if (s) {
        return index[o!][s] ? index[o!][s] : [];
      }
      // Return subjects that match object
      return Object.keys(index[o!]);
    }
    throw new Error('Passed invalid index');
  }

  // Prob use built in when I find it again
  public isVariable(term: RDF.Term) {
    return term.termType == 'Variable';
  }

  public tpToIds(tp: Operation, store: any) {
    const s: string | null = this.isVariable(tp.subject) ? null : store._ids[tp.subject.value];
    const p: string | null = this.isVariable(tp.predicate) ? null : store._ids[tp.predicate.value];
    const o: string | null = this.isVariable(tp.object) ? null : store._ids[tp.object.value];
    return [ s, p, o ];
  }

  /**
   * Determines the variable name that matches in the join
   * If no variables match the two triple patterns are a cartesian join
   * @param tpsIntermediateResult All triple patterns that are 'in' the intermediate result
   * @param tp2 triple pattern that will be joined into intermediate result
   * @returns
   */
  public determineJoinVariable(tpsIntermediateResult: Operation[], tp2: Operation): IJoinVariable {
    // TODO What to do with two matching variables (cyclic join graphs, see wanderjoin paper)
    const variablesIntermediateResult: Set<string> = new Set();
    for (const termType of [ 'subject', 'predicate', 'object' ]) {
      // If the terms in intermediate result triple patterns are variables we add to set of variables
      tpsIntermediateResult.filter(term => this.isVariable(term[termType]))
        .map(x => variablesIntermediateResult.add(x[termType].value));
    }

    if (this.isVariable(tp2.subject) && variablesIntermediateResult.has(tp2.subject.value)) {
      return { joinLocation: 's', joinVariable: tp2.subject.value };
    }
    if (this.isVariable(tp2.predicate) && variablesIntermediateResult.has(tp2.predicate.value)) {
      return { joinLocation: 'p', joinVariable: tp2.predicate.value };
    }
    if (this.isVariable(tp2.object) && variablesIntermediateResult.has(tp2.object.value)) {
      return { joinLocation: 'o', joinVariable: tp2.object.value };
    }
    return { joinLocation: 'c', joinVariable: '' };
  }

  /**
   * Generate all join combinations size 2
   * @param n the number of triple patterns
   */
  public joinCombinations(n: number): number[][][] {
    const combinations = [];
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        combinations.push([[ i ], [ j ]]);
      }
    }
    return combinations;
  }

  public initializeSamplesEnumeration(resultSets: Record<string,string>[][]) {
    const samples: Record<string, ISampleResult> = {};
    for (const [ i, resultSet ] of resultSets.entries()) {
      samples[JSON.stringify([ i ])] = {sample: resultSet, estimatedCardinality: resultSet.length };
    }
    return samples;
  }

  /**
   * Generate all combination arrays possible for adding a single index to the combination
   * @param combination
   * @param n
   * @returns
   */
  public generateNewCombinations(combination: number[][], n: number): number[][][] {
    if (n <= 0) {
      throw new Error('Received non-positive n');
    }
    if (combination.flat() && (combination.length !== 2 || combination[0].length === 0 || combination[1].length === 0)) {
      throw new Error('Combination should contain atleast two elements with length >= 1');
    }
    const used = new Set(combination.flat());
    return [ ...new Array(n).keys() ].filter(i => !used.has(i)).map(x => [ combination.flat(), [ x ]]);
  }

  /**
   * Fill s, p, o triple pattern with the value from the sample relation
   * @param sample The sample triple we join with our triple pattern
   * @param s subject of to join triple pattern
   * @param p predicate of to join triple pattern
   * @param o object of to join triple pattern
   * @param joinLocation What variable the join should be executed over
   * @returns
   */
  public setSampleRelation(sample: Record<string, string>, s: string | null, p: string | null, o: string | null, 
    joinLocation: 's' | 'p' | 'o', joinVariable: string): (string | null)[] {
    if (Object.keys(sample).length === 0){
      throw new Error('Sample should contain atleast a binding')
    }
    if (joinLocation === 's') {
      return [ sample[joinVariable], p, o ];
    }
    if (joinLocation === 'p') {
      return [ s, sample[joinVariable], o ];
    }
    if (joinLocation === 'o') {
      return [ s, p, sample[joinVariable] ];
    }
    throw new Error('Invalid join variable');
  }

  /**
   * Given a triple pattern, determines what index to use for lookup
   * @param s Subject of triple pattern
   * @param p Predicate of triple pattern
   * @param o Object of triple pattern
   * @returns string of index to use
   */
  public determineOptimalIndex(s: string | null, p: string | null, o: string | null):
  'subjects' | 'predicates' | 'objects' {
    if (s) {
      if (p) {
        return 'subjects';
      }
      if (o) {
        return 'objects';
      }

      return 'subjects';
    }
    if (p) {
      if (o) {
        return 'predicates';
      }

      return 'predicates';
    }
    // This case only o is defined
    if (o) {
      return 'objects';
    }
    throw new Error('Cant determine index of triple pattern with only variables');
  }

  /**
   * Inefficient code to construct indexes as follows: s p [o], p o [s], o s [p]
   */
  public constructArrayDict(storeIndex: Record<string, any>) {
    const termArrayDict: Record<string, Record<string, string[]>> = {};
    // Iterate over first layer of index
    for (const key1 in storeIndex) {
      const innerTerms: Record<string, string[]> = {};
      // Second layer
      for (const key2 in storeIndex[key1]) {
        const terms: string[] = Object.keys(storeIndex[key1][key2]);
        innerTerms[key2] = terms;
      }
      termArrayDict[key1] = innerTerms;
    }
    return termArrayDict;
  }


  public sampleArray<T>(array: T[], n: number): T[] {
    if (n > array.length) {
      return array;
    }

    const result = [];
    const arrCopy = [ ...array ]; // Create a copy of the array to avoid mutating the original one

    for (let i = 0; i < n; i++) {
      const randomIndex = Math.floor(Math.random() * (arrCopy.length - i)) + i;

      // Swap the randomly selected element to the position i
      [ arrCopy[i], arrCopy[randomIndex] ] = [ arrCopy[randomIndex], arrCopy[i] ];

      // Push the selected element to the result array
      result.push(arrCopy[i]);
    }

    return result;
  }

  /**
   * Temporary function to test the possiblity of using join sampling. This will NOT be used in any 'real' tests
   * @param fileName Filename to parse
   */
  public parseFile(fileName: string) {
    const parser = new Parser();
    const store = new Store();
    const data: string = fs.readFileSync(fileName, 'utf-8');
    const testResult: RDF.Quad[] = parser.parse(
      data,
    );
    store.addQuads(testResult);
    return store;
  }
}

export type ArrayIndex = Record<string, Record<string, Record<string, string[]>>>;

export interface ISampleOutput {
  /**
   *
   */
  sample: Record<string,string>[];
  /**
   *
   */
  selectivity: number;
}

export interface ISampleResult {
  /**
   * 
   */
  sample: Record<string,string>[];
  /**
   * Estimated cardanility associated with join combination
   */
  estimatedCardinality: number;
}

export interface IJoinVariable {
  /**
   * What location the join variable occurs in the to join triple pattern
   */
  joinLocation: "s"|"p"|"o"|"c";
  /**
   * The string of the variable
   */
  joinVariable: string;
}

export interface ICandidateCounts {
  /**
   * Number of candidates per sampled binding given as input
   */
  counts: number[];
  /**
   * Triple pattern used to look up candidates in index
   */
  sampleRelations: (string | null)[][]
}
