import { Factory } from 'sparqlalgebrajs';
import * as RDF from '@rdfjs/types';
import { Store, Parser } from 'n3';
import * as fs from 'fs';
import { Operation } from 'sparqlalgebrajs/lib/algebra';
import {FifoQueue} from './FifoQueue'

/**
 * A comunica Inner Multi Index Sampling RDF Join Actor.
 */
export class IndexBasedJoinSampler {
  public budget: number;

  public constructor(budget: number) {
    this.budget = budget;
  }
  
  // Prob use built in when I find it again
  public isVariable(term: RDF.Term){
    return term.constructor.name == "Variable";
  }

  public tpToIds(tp: Operation, store: any){
    const s: string | null = this.isVariable(tp.subject) ? null : store._ids[tp.subject.value];
    const p: string | null = this.isVariable(tp.predicate) ? null : store._ids[tp.predicate.value];
    const o: string | null = this.isVariable(tp.object) ? null : store._ids[tp.object.value];
    return [s, p, o]
  }

  /**
   * Determines what in the second triple pattern will be joined. 
   * If no variables match the two triple patterns are a cartesian join
   * @param tp1 
   * @param tp2 
   * @returns 
   */
  public determineJoinVariable(tpsIntermediateResult: Operation[], tp2: Operation): "s"|"p"|"o"|"c" {
    // TODO What to do with two matching variables
    const variablesIntermediateResult: Record<string, Set<string>> = {};
    variablesIntermediateResult.subject = new Set(
      tpsIntermediateResult.filter(term => this.isVariable(term.subject))
      .map(x => x.subject.value)
    );
    variablesIntermediateResult.predicate = new Set(
      tpsIntermediateResult.filter(term => this.isVariable(term.predicate))
      .map(x => x.predicate.value)
    );
    variablesIntermediateResult.object = new Set(
      tpsIntermediateResult.filter(term => this.isVariable(term.object))
      .map(x => x.object.value)
    );
    if (this.isVariable(tp2.subject) && variablesIntermediateResult.subject.has(tp2.subject.value) ){
      return "s"
    }
    if (this.isVariable(tp2.predicate) && variablesIntermediateResult.predicate.has(tp2.predicate.value)){
      return "p"
    }
    if (this.isVariable(tp2.object) && variablesIntermediateResult.object.has(tp2.object.value)){
      return "o"
    }
    return "c"
  }

  public run(n: number, store: any, arrayIndex: Record<string, ArrayIndex> , tps: Operation[]){
    // Get all triples matching triple pattern
    const sampledTriplePatterns = this.sampleTriplePatternsQuery(Number.POSITIVE_INFINITY, store, arrayIndex, tps);
    this.bottomUpEnumeration(n, store, arrayIndex, sampledTriplePatterns, tps);
  } 
  /**
   * Generate all join combinations size 2
   * @param n the number of triple patterns
   */
  public joinCombinations(n: number): number[][][]{
    const combinations = []
    for (let i = 0; i < n; i++){
      for (let j = i+1; j < n; j++){
        combinations.push([[i],[j]]);
      } 
    }
    return combinations;
  }

  public initializeSamplesEnumeration(resultSets: string[][][]){
    const samples: Record<string, string[][]> = {};
    for (let i = 0; i < resultSets.length; i++){
      samples[JSON.stringify([i])] = resultSets[i];
    }
    return samples
  }

  /**
   * Generate all combination arrays possible for adding a single index to the combination
   * @param combination 
   * @param n 
   * @returns 
   */
  public generateNewCombinations(combination: number[][], n: number): number[][][]{
    const used = new Set(combination.flat());
    return [...Array(n).keys()].filter(i => !used.has(i)).map(x=>[combination.flat(), [x]]);   
  }

  public bottomUpEnumeration(n: number, store: any, arrayIndex: Record<string, ArrayIndex>, resultSets: string[][][], tps: Operation[]){
    const cardinalities: Record<string, number> = {}
    // Samples are stored by their two participating expressions like [[1], [2]] joining result set 
    // 1 and 2, while [[1,2],[3]] joins result sets of join between 1,2 and result set of 3.
    const samples: Record<string, string[][]> = this.initializeSamplesEnumeration(resultSets);
    
    const problemQueue = new FifoQueue<number[][]>();
    this.joinCombinations(tps.length).map(join => problemQueue.enqueue(join));

    while (!problemQueue.isEmpty()){
      // The combination of triple patterns we're evaluating
      const combination = problemQueue.dequeue()!;
      // The intermediate result set we use as input sample for join sampling
      const baseSample = samples[JSON.stringify(combination[0])];
      // The triple patterns already joined to produce the intermediate result set
      const tpInBaseSample = combination[0].map(x=>tps[x]);
      // The triple pattern we join with intermediate result set
      const spo = this.tpToIds(tps[combination[1][0]], store);
      const joinVariable = this.determineJoinVariable(tpInBaseSample, tps[combination[1][0]])
      console.log(tpInBaseSample)
      console.log(tps[combination[1][0]])
      console.log(joinVariable);
      if (joinVariable == "c"){
        // Cross product so disregard this 
        // TODO Circle back to cross products if we have exhausted all non-cross products
        continue;
      }
      const samplingOutput: ISampleOutput = this.sampleJoin(n, baseSample, spo[0], spo[1], spo[2], joinVariable, arrayIndex);
      samples[JSON.stringify(combination)] = samplingOutput.sample;
      // Generate new combinations and add to queue
      this.generateNewCombinations(combination, tps.length).map(x => problemQueue.enqueue(x));
      console.log(samplingOutput.sample);
    }
  }

  public sampleJoin(n: number, samples: string[][], s: string | null, p: string | null, o: string | null, 
    joinVariable: 's'|'p'|'o', arrayIndex: Record<string, ArrayIndex>): ISampleOutput{
      // First count the number of possible join candidates with sample
      const counts: number[] = [], subcounts: number[][] = [];
      const sampleRelations: (string | null)[][] = [];
      for (const sample of samples){
        const sampleRelation = this.setSampleRelation(sample, s, p, o, joinVariable);
        console.log(sampleRelation)

        const candidateCounts: number[] = this.countTriplesJoinRelation(arrayIndex[''], 
          sampleRelation[0], sampleRelation[1], sampleRelation[2]
        );
        const totalCountSample = candidateCounts.reduce((acc, count) => acc += count, 0);
        counts.push(totalCountSample);
        subcounts.push(candidateCounts);
        sampleRelations.push(sampleRelation);
      }
      const sum: number = counts.reduce((acc, count) => acc += count, 0);
      const ids = this.sampleArray([...Array(sum).keys()], n);
      
      const joinSample: string[][] = []
      for (const id of ids){
        let searchIndex = 0;
        for (let i = 0; i < counts.length; i++){
          searchIndex += counts[i];
          // If count > id we take the joins associated with this sample
          if (id < searchIndex){
            // Go back one step
            searchIndex -= counts[i]
            // Find index
            const tripleIndex = id - (searchIndex)
            const triples = this.lookUpIndex(sampleRelations[i][0], sampleRelations[i][1], sampleRelations[i][2], arrayIndex['']);
            joinSample.push(triples[tripleIndex]);
            break;
          }
        }
      }
      // TODO validate if this is correct
      const selectivity = joinSample.length / samples.length;
      return {
        sample: joinSample,
        selectivity: selectivity
      }
  }

  public setSampleRelation(sample: string[], s: string | null, p: string | null, o: string | null, 
    joinVariable: 's'|'p'|'o'): (string | null)[]{
      if (joinVariable == 's'){
        return [sample[0], p, o];
      }
      if (joinVariable == 'p'){
        return [s, sample[1], o];
      }
      if (joinVariable == 'o'){
        return [s, p, sample[2]];
      }
      else{
        throw new Error('Invalid join variable')
      }
  }

  public countTriplesJoinRelation(arrayIndex: ArrayIndex, 
    s: string | null, p: string | null, o: string | null
  ): number[]{
    const subTotalCandidateCounts: number[] = [];
    const nVariables = [s, p, o].reduce((total: number, term) => term === null ? total + 1: total, 0);
    const indexToUse = this.determineOptimalIndex(s, p, o);

    // If we only count for one variable in triple pattern return size of array index of triple pattern
    if (nVariables === 1){
      const indexArray = this.pollIndex(arrayIndex, s, p , o, indexToUse);
      // If no index, we have no results for this part of sample
      if (!indexArray){
        subTotalCandidateCounts.push(0);      
      }
      else{
        subTotalCandidateCounts.push(indexArray.length);      
      }
    }
    // Two variables we need to iterate over two layers of index
    else if (nVariables === 2){
      const firstTerms = this.pollIndex(arrayIndex, s, p, o , indexToUse);
      for (const term in firstTerms){
        // Determine the type of term we retrieve
        if (indexToUse == "subjects"){
          // s P o index 
          p = term;
        }
        if (indexToUse == "predicates"){
          // p O s index
          o = term;
        }
        if (indexToUse == "objects"){
          // o S p
          s = term;
        }
        const secondTerms = this.pollIndex(arrayIndex, s, p, o, indexToUse);
        subTotalCandidateCounts.push(secondTerms.length);
      }
    }
    else{
      throw new Error("Join relation should not have 3 variables");
    }
    return subTotalCandidateCounts
  }

  public sampleTriplePatternsQuery(n: number, store: any, arrayIndex: Record<string, ArrayIndex>, tps: Operation[]){
    const sampledTriplePatterns = [];
    for (const tp of tps){
      const spo = this.tpToIds(tp, store);
      sampledTriplePatterns.push(this.sampleIndex(spo[0], spo[1], spo[2], n, arrayIndex['']));
    }
    return sampledTriplePatterns
  }
  /**
   * Given a triple pattern, determines what index to use for lookup
   * @param s Subject of triple pattern
   * @param p Predicate of triple pattern
   * @param o Object of triple pattern
   * @returns string of index to use
   */
  public determineOptimalIndex(s: string | null, p: string | null, o: string | null): 
  "subjects"|"predicates"|"objects" {
    if (s){
      if (p){
        return "subjects";
      }
      if (o){
        return "objects";
      }
      else{
        return "subjects";
      }
    }
    if (p){
      if (o){
        return "predicates";
      }
      else{
        return "predicates";
      }
    }
    // This case only o is defined
    if (o){
      return "objects";
    }
    else 
      throw new Error("Cant determine index of triple pattern with only variables")
  }

  public lookUpIndex(s: string | null, p: string | null, o: string | null, arrayIndex: ArrayIndex){
    // If we lookup a full triple, if it exists return that triple
    if (s && p && o){
      // Abuse of the pollIndex function as this shouldn't return undefined, but in this case can
      const objs = this.pollIndex(arrayIndex, s, p, o, "subjects");
      // If we find objects matching s, p
      if (objs){
        const setObjs = new Set(objs);
        // If the passed o matches the objects matchin s,p return the triples
        if (setObjs.has(o)){
          return [[s, p, o]];
        }
      }
      // Else its an invalid triple
      return [];
    }
    if (s){
      if (p){
        const objs = this.pollIndex(arrayIndex, s, p, o, "subjects");
        const triples = objs.map(x=>[s, p, x]);
        return triples;
      }
      if (o){
        const preds = this.pollIndex(arrayIndex, s, p, o, "objects");
        const triples = preds.map(x=>[s, x, o]);
        return triples;
      }
      else{
        // If only subject is set, we first get all predicates with subject, then get all objects
        // for each subject, predicate combination
        let triples: string[][] = [];
        const preds = this.pollIndex(arrayIndex, s, p, o, "subjects");
        for (const pred of preds){
          const objs = this.pollIndex(arrayIndex, s, pred, o, "subjects")
          triples = [...triples, ...objs.map(x => [s, pred, x])];
        }
        return triples;
      }
    }
    if (p){
      if (o){
        const subs = this.pollIndex(arrayIndex, s, p, o, "predicates")
        const triples = subs.map(x=>[x, p, o]);
        return triples;
      }
      else{
        const objs = this.pollIndex(arrayIndex, s, p, o, "predicates");
        let triples: string[][] = [];
        for (const obj of objs){
          const subs = this.pollIndex(arrayIndex, s, p, obj, "predicates");
          triples = [...triples, ...subs.map(x => [x, p, obj])];
        }
        return triples;
      }
    }
    // This case only o is defined
    if (o){
      let triples: string[][] = [];
      const subs = this.pollIndex(arrayIndex, s, p, o, "objects")
      for (const sub of subs){
        const preds = this.pollIndex(arrayIndex, sub, p, o, "objects")
        triples = [...triples, ...preds.map(x => [sub, x, o])];
      }
      return triples;
    }
    else{
      throw Error("Not yet lookup up all triples")
    }
  }

  // Take a random sample of triples matching a pattern out of an index
  public sampleIndex(s: string | null, p: string | null, o: string | null, n: number, arrayIndex: ArrayIndex){
    const triples = this.lookUpIndex(s, p, o, arrayIndex);
    return this.sampleArray(triples, n);
  }

  public pollIndex(arrayIndex: ArrayIndex, s: string | null, p: string | null, o: string | null, 
    indexToUse: "subjects" | "predicates" | "objects"
  ): string[]{
    const index = arrayIndex[indexToUse];
    if (indexToUse == "subjects"){
      // object will always be null when using subject index
      if (p){
        // Return triples matching subject - predicate
        return index[s!][p];
      }
      // Return the predicates that match subject
      return Object.keys(index[s!]);
    }

    else if (indexToUse == "predicates"){
      // Return subjects that match predicate - object
      if (o) {
        return index[p!][o];
      }
      // Return objects that match predicate
      return Object.keys(index[p!]);
    }
    else if (indexToUse == "objects"){
      // Return predicates that match subjects - objects
      if (s){
        return index[o!][s];
      }
      // Return subjects that match object
      return Object.keys(index[o!]);
    }
    else{
      throw new Error("Passed invalid index");
    }  
  }

  /**
   * Inefficient code to construct indexes as follows: s p [o], p o [s], o s [p]
   */
  public constructArrayDict(storeIndex: Record<string, any>){
    const termArrayDict: Record<string, Record<string, string[]>> = {};
    // Iterate over first layer of index
    for (const key1 in storeIndex){
      const innerTerms: Record<string, string[]> = {};
      // Second layer
      for (const key2 in storeIndex[key1]){
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
    const arrCopy = [...array]; // Create a copy of the array to avoid mutating the original one

    for (let i = 0; i < n; i++) {
        const randomIndex = Math.floor(Math.random() * (arrCopy.length - i)) + i;
        
        // Swap the randomly selected element to the position i
        [arrCopy[i], arrCopy[randomIndex]] = [arrCopy[randomIndex], arrCopy[i]];

        // Push the selected element to the result array
        result.push(arrCopy[i]);
    }

    return result;
  }

  /**
   * Temporary function to test the possiblity of using join sampling. This will NOT be used in any 'real' tests
   * @param fileName Filename to parse
   */
  public parseFile(fileName: string){
    const parser = new Parser();
    const store = new Store();
    const data: string = fs.readFileSync(fileName, 'utf-8');
    const testResult: RDF.Quad[] = parser.parse(
      data,
      );
    store.addQuads(testResult);
    return store
  }
}

export type ArrayIndex = Record<string, Record<string, Record<string, string[]>>>;

export interface ISampleOutput{
  /**
   * 
   */
  sample: string[][],
  /**
   * 
   */
  selectivity: number
}