import { IndexBasedJoinSampler } from '../lib/IndexBasedJoinSampler';
import '@comunica/jest';

describe('IndexBasedJoinSampler', () => {
  let joinSampler: any;
  let spyOnMathRandom: any;

  beforeEach(() => {
    spyOnMathRandom = jest.spyOn(global.Math, 'random').mockReturnValue(0.4);
    joinSampler = new IndexBasedJoinSampler(0);
  });

  describe('sample array', () => {
    let array: number[];
    beforeEach(() => {
      array = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ];
    });

    it('should correctly sample', () => {
      expect(joinSampler.sampleArray(array, 2)).toEqual([ 5, 1 ]);
    });

    it('should correctly sample on edge random number (0)', () => {
      spyOnMathRandom = jest.spyOn(global.Math, 'random').mockReturnValue(0);
      expect(joinSampler.sampleArray(array, 2)).toEqual([ 1, 2 ]);
    });

    it('should correctly sample on edge random number (.9999)', () => {
      spyOnMathRandom = jest.spyOn(global.Math, 'random').mockReturnValue(0.9999);
      expect(joinSampler.sampleArray(array, 2)).toEqual([ 10, 1 ]);
    });

    it('should sample uniformly', () => {
      // Function runs chi-squared test for uniform sampling distribution with alpha = .95
      spyOnMathRandom.mockRestore();
      const n_runs = 10_000;

      // eslint-disable-next-line
      const occurences: number[] = new Array(10).fill(0);
      for (let i = 0; i < n_runs; i++) {
        const sample = joinSampler.sampleArray(array, 4);
        for (const value of sample) {
          occurences[value]++;
        }
      }
      // Chi squared for uniform distribution
      const chiSquared = occurences.map(x => ((x - n_runs / 10) ^ 2) / (n_runs / 10))
        .reduce((acc, curr) => acc += curr, 0);
      // We have 10 categories (9 df), confidence of 0.05, chi squared 16.919
      expect(chiSquared > 16.9);
    });

    it('should return full array when n > array.length', () => {
      expect(joinSampler.sampleArray(array, 11)).toEqual([ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ]);
    });

    it('should return empty array if n = 0', () => {
      expect(joinSampler.sampleArray(array, 0)).toEqual([]);
    });
    it('should return empty array if n < 0', () => {
      expect(joinSampler.sampleArray(array, -1)).toEqual([]);
    });
  });
  describe('determine optimal index', () => {
    it('should throw an error when all parameters are null', () => {
      expect(() => {
        joinSampler.determineOptimalIndex(null, null, null);
      }).toThrow('Cant determine index of triple pattern with only variables');
    });

    it('should return "subjects" when only s is defined', () => {
      expect(joinSampler.determineOptimalIndex('s1', null, null)).toBe('subjects');
    });

    it('should return "predicates" when only p is defined', () => {
      expect(joinSampler.determineOptimalIndex(null, 'pValue', null)).toBe('predicates');
    });

    it('should return "objects" when only o is defined', () => {
      expect(joinSampler.determineOptimalIndex(null, null, 'oValue')).toBe('objects');
    });

    it('should return "subjects" when s and p are defined but o is null', () => {
      expect(joinSampler.determineOptimalIndex('s1', 'pValue', null)).toBe('subjects');
    });

    it('should return "subjects" when s and o are defined but p is null', () => {
      expect(joinSampler.determineOptimalIndex('s1', null, 'oValue')).toBe('objects');
    });

    it('should return "predicates" when p and o are defined but s is null', () => {
      expect(joinSampler.determineOptimalIndex(null, 'pValue', 'oValue')).toBe('predicates');
    });

    it('should return "subjects" when all parameters are defined', () => {
      expect(joinSampler.determineOptimalIndex('sValue', 'pValue', 'oValue')).toBe('subjects');
    });
  });
  describe('set sample relation', () => {
    let sample: Record<string, string>;
    beforeEach(() => {
      sample = {'v0': 'sub', 'v1': 'pred', 'v2': 'obj'};
    })
    it('should correctly fill subject when joinVariable is "s"', () => {
      const result = joinSampler.setSampleRelation(sample, null, 'pred', 'obj', 's', 'v0');
      expect(result).toEqual([ 'sub', 'pred', 'obj' ]);
    });

    it('should correctly fill predicate when joinVariable is "p"', () => {
      const result = joinSampler.setSampleRelation(sample, 'sub', null, 'obj', 'p', 'v1');
      expect(result).toEqual([ 'sub', 'pred', 'obj' ]);
    });

    it('should correctly fill object when joinVariable is "o"', () => {
      const result = joinSampler.setSampleRelation(sample, 'sub', 'pred', null, 'o', 'v2');
      expect(result).toEqual([ 'sub', 'pred', 'obj' ]);
    });

    it('should handle empty sample correctly', () => {
      const sample: Record<string,string> = {};
      expect(() => {
        joinSampler.setSampleRelation(sample, 'sub', 'pred', 'obj', 's', 'v0');
      }).toThrow(
        new Error('Sample should contain atleast a binding'),
      );
    });

    it('should handle null values for parameters', () => {
      const result = joinSampler.setSampleRelation(sample, null, null, null, 's', 'v0');
      expect(result).toEqual([ 'sub', null, null ]);
    });

    it('should throw an error for an invalid joinVariable', () => {
      expect(() => {
        joinSampler.setSampleRelation(sample, 'sub', 'pred', 'obj', 'invalid', 'v0');
      }).toThrow('Invalid join variable');
    });
  });
  describe('generateNewCombinations', () => {
    it('should throw an error when combination does not contain exactly 2 subarrays', () => {
      const combination = [[ 0 ], [ 1 ], [ 2 ]];
      const n = 3;
      expect(() => {
        joinSampler.generateNewCombinations(combination, n);
      }).toThrow('Combination should contain atleast two elements with length >= 1');
    });

    it('should throw an error when the first subarray in combination is empty', () => {
      const combination = [[], [ 1 ]];
      const n = 3;
      expect(() => {
        joinSampler.generateNewCombinations(combination, n);
      }).toThrow('Combination should contain atleast two elements with length >= 1');
    });

    it('should throw an error when the second subarray in combination is empty', () => {
      const combination = [[ 0 ], []];
      const n = 3;
      expect(() => {
        joinSampler.generateNewCombinations(combination, n);
      }).toThrow('Combination should contain atleast two elements with length >= 1');
    });

    // Basic Functionality Test
    it('should generate new combinations correctly for a valid input', () => {
      const combination = [[ 0 ], [ 1 ]];
      const n = 3;
      const result = joinSampler.generateNewCombinations(combination, n);
      expect(result).toEqual([
        [[ 0, 1 ], [ 2 ]],
      ]);
    });

    it('should generate new combinations correctly for a valid input', () => {
      const combination = [[ 0 ], [ 1 ]];
      const n = 4;
      const result = joinSampler.generateNewCombinations(combination, n);
      expect(result).toEqual([
        [[ 0, 1 ], [ 2 ]],
        [[ 0, 1 ], [ 3 ]],
      ]);
    });

    // Edge Case Test
    it('should return an empty array when n is 0', () => {
      const combination = [[ 0 ], [ 1 ]];
      const n = 0;
      expect(() => {
        joinSampler.generateNewCombinations(combination, n);
      }).toThrow('Received non-positive n');
    });

    it('should return an empty array when all numbers from 0 to n-1 are used', () => {
      const combination = [[ 0, 1 ], [ 2 ]];
      const n = 3;
      const result = joinSampler.generateNewCombinations(combination, n);
      expect(result).toEqual([]);
    });
  });
  describe('initializeSamplesEnumeration', () => {
    it('should initialize samples correctly for a non-empty resultSets', () => {
      const resultSets = [
        [[ 'a1', 'b1' ], [ 'c1', 'd1' ]],
        [[ 'a2', 'b2' ], [ 'c2', 'd2' ]],
      ];
      const result = joinSampler.initializeSamplesEnumeration(resultSets);

      expect(result).toEqual({
        '[0]': {sample: [[ 'a1', 'b1' ], [ 'c1', 'd1' ]], estimatedCardinality: 2},
        '[1]': {sample: [[ 'a2', 'b2' ], [ 'c2', 'd2' ]], estimatedCardinality: 2}
      });
    });

    it('should handle resultSets with a single resultSet', () => {
      const resultSets = [
        [[ 'a1', 'b1' ], [ 'c1', 'd1' ]],
      ];
      const result = joinSampler.initializeSamplesEnumeration(resultSets);

      expect(result).toEqual({
        '[0]': {sample: [[ 'a1', 'b1' ], [ 'c1', 'd1' ]], estimatedCardinality: 2},
      });
    });

    it('should handle resultSets with multiple empty arrays', () => {
      const resultSets = [
        [],
        [[ 'a2', 'b2' ], [ 'c2', 'd2' ]],
        [],
      ];
      const result = joinSampler.initializeSamplesEnumeration(resultSets);

      expect(result).toEqual({
        '[0]': {sample: [], estimatedCardinality: 0},
        '[1]': {sample:[[ 'a2', 'b2' ], [ 'c2', 'd2' ]], estimatedCardinality: 2},
        '[2]': {sample:[], estimatedCardinality: 0},
      });
    });

    it('should create unique keys even with similar content in resultSets', () => {
      const resultSets = [
        [[ 'a1', 'b1' ]],
        [[ 'a1', 'b1' ]],
      ];
      const result = joinSampler.initializeSamplesEnumeration(resultSets);

      expect(result).toEqual({
        '[0]': {sample: [[ 'a1', 'b1' ]], estimatedCardinality: 1 } ,
        '[1]': {sample: [[ 'a1', 'b1' ]], estimatedCardinality: 1}
      });
    });
  });
  describe('joinCombinations', () => {
    it('should create all valid size 2 combinations (without order)', () => {
      const n = 4;
      expect(joinSampler.joinCombinations(n)).toEqual([
        [[ 0 ], [ 1 ]],
        [[ 0 ], [ 2 ]],
        [[ 0 ], [ 3 ]],
        [[ 1 ], [ 2 ]],
        [[ 1 ], [ 3 ]],
        [[ 2 ], [ 3 ]],
      ]);
    });
    it('should return empty when n = 0', () => {
      const n = 0;
      expect(joinSampler.joinCombinations(n)).toEqual([]);
    });
  });
  describe('determineJoinVariable', () => {
    let operation1: any;
    let operation2: any;
    let operation3: any;
    let operation4: any;
    let operation5: any;
    let operation6: any;
    beforeEach(() => {
      operation1 = {
        subject: { termType: 'Variable', value: 'v0' },
        predicate: {
          termType: 'NamedNode',
          value: 'p0',
        },
        object: { termType: 'Variable', value: 'v1' },
      };
      operation2 = {
        subject: { termType: 'Variable', value: 'v0' },
        predicate: {
          termType: 'NamedNode',
          value: 'p1',
        },
        object: { termType: 'Variable', value: 'v2' },
      };
      operation3 = {
        subject: { termType: 'Variable', value: 'v1' },
        predicate: {
          termType: 'NamedNode',
          value: 'p2',
        },
        object: { termType: 'Variable', value: 'v2' },
      };
      operation4 = {
        subject: {
          termType: 'NamedNode',
          value: 's1',
        },
        predicate: { termType: 'Variable', value: 'p0' },
        object: { termType: 'Variable', value: 'v1' },
      };
      operation5 = {
        subject: {
          termType: 'NamedNode',
          value: 's2',
        },
        predicate: { termType: 'Variable', value: 'p0' },
        object: {
          termType: 'NamedNode',
          value: 'o1',
        },
      };
      operation6 = {
        subject: { termType: 'Variable', value: 'v2' },
        predicate: {
          termType: 'NamedNode',
          value: 'p2',
        },
        object: { termType: 'Variable', value: 'v3' },
      };
    });

    it('should work with two triple patterns in star', () => {
      expect(joinSampler.determineJoinVariable([ operation1 ], operation2)).toEqual({joinLocation: 's', joinVariable: 'v0'});
    });
    it('should work with two triple patterns in path (object)', () => {
      expect(joinSampler.determineJoinVariable([ operation1 ], operation3)).toEqual({joinLocation: 's', joinVariable: 'v1'});
    });
    it('should work with two triple patterns in path (subject)', () => {
      expect(joinSampler.determineJoinVariable([ operation3 ], operation1)).toEqual({joinLocation: 'o', joinVariable: 'v1'});
    });
    it('should work with two triple patterns with predicate join', () => {
      expect(joinSampler.determineJoinVariable([ operation4 ], operation5)).toEqual({joinLocation: 'p', joinVariable: 'p0'});
    });
    it('should correctly identify cartesian joins', () => {
      expect(joinSampler.determineJoinVariable([ operation1 ], operation6)).toEqual({joinLocation: 'c', joinVariable: ''});
      expect(joinSampler.determineJoinVariable([ operation5 ], operation6)).toEqual({joinLocation: 'c', joinVariable: ''});
    });
    it('should work with multiple triple patterns in intermediate result', () => {
      expect(joinSampler.determineJoinVariable([ operation1, operation2 ], operation6)).toEqual({joinLocation: 's', joinVariable: 'v2'});
    });
  });

  describe('tpToIds', () => {
    let store: any;
    beforeEach(() => {
      store = {
        _ids: {
          s0: '0',
          p0: '1',
          o0: '2',
        },
      };
    });
    it('should correctly identify subject id', () => {
      const triplePattern = {
        subject: {
          termType: 'NamedNode',
          value: 's0',
        },
        predicate: { termType: 'Variable', value: 'p0' },
        object: { termType: 'Variable', value: 'o0' },
      };
      expect(joinSampler.tpToIds(triplePattern, store)).toEqual([ '0', null, null ]);
    });
    it('should correctly identify predicate id', () => {
      const triplePattern = {
        subject: { termType: 'Variable', value: 'v0' },
        predicate: {
          termType: 'NamedNode',
          value: 'p0',
        },
        object: { termType: 'Variable', value: 'v1' },
      };
      expect(joinSampler.tpToIds(triplePattern, store)).toEqual([ null, '1', null ]);
    });
    it('should correctly identify object id', () => {
      const triplePattern = {
        subject: { termType: 'Variable', value: 's0' },
        predicate: { termType: 'Variable', value: 'p0' },
        object: {
          termType: 'NamedNode',
          value: 'o0',
        },
      };
      expect(joinSampler.tpToIds(triplePattern, store)).toEqual([ null, null, '2' ]);
    });
    it('should return undefined when term is not in store', () => {
      const triplePattern = {
        subject: { termType: 'Variable', value: 'v0' },
        predicate: {
          termType: 'NamedNode',
          value: 'p1',
        },
        object: { termType: 'Variable', value: 'v1' },
      };
      expect(joinSampler.tpToIds(triplePattern, store)).toEqual([ null, undefined, null ]);
    });
  });
  describe('index-based functions', () => {
    let sampleArrayIndex: any;

    beforeEach(() => {
      // Sample data structure for arrayIndex
      sampleArrayIndex = {
        subjects: {
          s1: {
            p1: [ 'o1', 'o2' ],
            p2: [ 'o3' ],
          },
          s2: {
            p1: [ 'o4' ],
          },
          s3: {
            p1: [ 'o2' ],
          },
        },
        predicates: {
          p1: {
            o1: [ 's1' ],
            o2: [ 's3', 's1' ],
            o4: [ 's2' ],
          },
          p2: {
            o3: [ 's1' ],
          },
        },
        objects: {
          o1: {
            s1: [ 'p1' ],
          },
          o2: {
            s1: [ 'p1' ],
            s3: [ 'p1' ],
          },
          o3: {
            s1: [ 'p2' ],
          },
          o4: {
            s2: [ 'p1' ],
          },
        },
      };
    });
    describe('pollIndex', () => {
      it('should return an empty array if any of s, p, or o is undefined', () => {
        expect(joinSampler.pollIndex(sampleArrayIndex, undefined, 'p1', 'o1', 'subjects')).toEqual([]);
        expect(joinSampler.pollIndex(sampleArrayIndex, 's1', undefined, 'o1', 'predicates')).toEqual([]);
        expect(joinSampler.pollIndex(sampleArrayIndex, 's1', 'p1', undefined, 'objects')).toEqual([]);
      });

      it('should return empty array for non-existant subject, with given predicate', () => {
        expect(joinSampler.pollIndex(sampleArrayIndex, 's4', 'p1', null, 'subjects')).toEqual([]);
      });
      it('should return empty array for non-existant subject, with null predicate', () => {
        expect(joinSampler.pollIndex(sampleArrayIndex, 's5', null, null, 'subjects')).toEqual([]);
      });
      it('should return empty array for existing subject, with non-existing predicate', () => {
        expect(joinSampler.pollIndex(sampleArrayIndex, 's1', 'predicate3', null, 'subjects')).toEqual([]);
      });

      it('should return empty array for non-existant predicate, with null object', () => {
        expect(joinSampler.pollIndex(sampleArrayIndex, 'predicate3', null, null, 'predicates')).toEqual([]);
      });
      it('should return empty array for non-existant predicate, with given object', () => {
        expect(joinSampler.pollIndex(sampleArrayIndex, 'predicate3', 'o1', null, 'predicates')).toEqual([]);
      });
      it('should return empty array for existant predicate, with non-existant object', () => {
        expect(joinSampler.pollIndex(sampleArrayIndex, 'p1', null, 'object5', 'subjects')).toEqual([]);
      });

      it('should return empty array for non-existant object, with null subject', () => {
        expect(joinSampler.pollIndex(sampleArrayIndex, 'object5', null, null, 'objects')).toEqual([]);
      });
      it('should return empty array for non-existant object, with given subject', () => {
        expect(joinSampler.pollIndex(sampleArrayIndex, 'object5', 'subjec1', null, 'objects')).toEqual([]);
      });
      it('should return empty array for existing object, with non-existant subject', () => {
        expect(joinSampler.pollIndex(sampleArrayIndex, 'o1', 's5', null, 'subjects')).toEqual([]);
      });

      it('should return predicates that match the given subject when indexToUse is subjects and no predicate is specified', () => {
        expect(joinSampler.pollIndex(sampleArrayIndex, 's1', null, null, 'subjects')).toEqual([ 'p1', 'p2' ]);
      });

      it('should return triples matching subject-predicate when indexToUse is subjects and both subject and predicate are provided', () => {
        expect(joinSampler.pollIndex(sampleArrayIndex, 's1', 'p1', null, 'subjects')).toEqual([ 'o1', 'o2' ]);
      });

      it('should return objects that match a given predicate when indexToUse is predicates and no object is provided', () => {
        expect(joinSampler.pollIndex(sampleArrayIndex, null, 'p1', null, 'predicates')).toEqual([ 'o1', 'o2', 'o4' ]);
      });

      it('should return subjects that match a given predicate-object pair when indexToUse is predicates and both predicate and object are provided', () => {
        expect(joinSampler.pollIndex(sampleArrayIndex, null, 'p1', 'o1', 'predicates')).toEqual([ 's1' ]);
      });

      it('should return subjects that match the given object when indexToUse is objects and no subject is provided', () => {
        expect(joinSampler.pollIndex(sampleArrayIndex, null, null, 'o1', 'objects')).toEqual([ 's1' ]);
      });

      it('should return predicates that match the subject-object pair when indexToUse is objects and both subject and object are provided', () => {
        expect(joinSampler.pollIndex(sampleArrayIndex, 's1', null, 'o1', 'objects')).toEqual([ 'p1' ]);
      });

      it('throws error when s, p, and o are all null', () => {
        // Expect the function to throw an error when called with null arguments
        expect(() => joinSampler.pollIndex(sampleArrayIndex, null, null, null, 'subjects')).toThrow(
          'Not yet implemented index poll for three variables',
        );
      });
      it('throws error when invalid index passed', () => {
        // Expect the function to throw an error when called with null arguments
        expect(() => joinSampler.pollIndex(sampleArrayIndex, 's1', null, null, 'fail')).toThrow(
          'Passed invalid index',
        );
      });
    });

    describe('lookUpIndex', () => {
      it('should return an empty array if any of s, p, or o is undefined', () => {
        expect(joinSampler.lookUpIndex(sampleArrayIndex, undefined, 'p1', 'o1')).toEqual([]);
        expect(joinSampler.lookUpIndex(sampleArrayIndex, 's1', undefined, 'o1')).toEqual([]);
        expect(joinSampler.lookUpIndex(sampleArrayIndex, 's1', 'p1', undefined)).toEqual([]);
      });

      it('should return empty array for non-existant subject, with given predicate', () => {
        expect(joinSampler.lookUpIndex(sampleArrayIndex, 's4', 'p1', null)).toEqual([]);
      });
      it('should return empty array for non-existant subject, with null predicate', () => {
        expect(joinSampler.lookUpIndex(sampleArrayIndex, 's5', null, null)).toEqual([]);
      });
      it('should return empty array for existing subject, with non-existing predicate', () => {
        expect(joinSampler.lookUpIndex(sampleArrayIndex, 's1', 'predicate3', null)).toEqual([]);
      });

      it('should return empty array for non-existant predicate, with null object', () => {
        expect(joinSampler.lookUpIndex(sampleArrayIndex, 'predicate3', null, null)).toEqual([]);
      });
      it('should return empty array for non-existant predicate, with given object', () => {
        expect(joinSampler.lookUpIndex(sampleArrayIndex, 'predicate3', 'o1', null)).toEqual([]);
      });
      it('should return empty array for existant predicate, with non-existant object', () => {
        expect(joinSampler.lookUpIndex(sampleArrayIndex, 'p1', null, 'object5')).toEqual([]);
      });

      it('should return empty array for non-existant object, with null subject', () => {
        expect(joinSampler.lookUpIndex(sampleArrayIndex, 'object5', null, null)).toEqual([]);
      });
      it('should return empty array for non-existant object, with given subject', () => {
        expect(joinSampler.lookUpIndex(sampleArrayIndex, 'object5', 'subjec1', null)).toEqual([]);
      });
      it('should return empty array for existing object, with non-existant subject', () => {
        expect(joinSampler.lookUpIndex(sampleArrayIndex, 'o1', 's5', null)).toEqual([]);
      });
      it('should return triples that match the given subject ', () => {
        expect(joinSampler.lookUpIndex(sampleArrayIndex, 's1', null, null)).toEqual(
          [[ 's1', 'p1', 'o1' ], [ 's1', 'p1', 'o2' ], [ 's1', 'p2', 'o3' ]],
        );
      });
      it('should return triples matching subject-predicate with subj & pred', () => {
        expect(joinSampler.lookUpIndex(sampleArrayIndex, 's1', 'p1', null)).toEqual(
          [[ 's1', 'p1' ,'o1'],['s1','p1','o2']]);
      });
      it('should return triples that match a given predicate', () => {
        expect(joinSampler.lookUpIndex(sampleArrayIndex, null, 'p1', null)).toEqual(
          [[ 's1', 'p1', 'o1' ],
        ['s3', 'p1', 'o2'],
      ['s1', 'p1', 'o2'],
    ['s2', 'p1', 'o4']]);
      });
      it('should return triples matching a given predicate-object pair', () => {
        expect(joinSampler.lookUpIndex(sampleArrayIndex, null, 'p1', 'o1')).toEqual(
          [[ 's1', 'p1', 'o1' ]]);
      });
      it('should return triples that match the given object', () => {
        expect(joinSampler.lookUpIndex(sampleArrayIndex, null, null, 'o1')).toEqual(
          [[ 's1', 'p1', 'o1' ]]);
      });
      it('should return predicates that match the subject-object pair', () => {
        expect(joinSampler.lookUpIndex(sampleArrayIndex, 's1', null, 'o1')).toEqual(
          [[ 's1', 'p1', 'o1' ]]);
      });
      it('throws error when s, p, and o are all null', () => {
        // Expect the function to throw an error when called with null arguments
        expect(() => joinSampler.lookUpIndex(sampleArrayIndex, null, null, null)).toThrow(
          'Not yet implemented lookup up all triples',
        );
      });
      it('should return triples that match a given triple (not pattern)', () => {
        expect(joinSampler.lookUpIndex(sampleArrayIndex, 's1', 'p1', 'o1')).toEqual(
          [[ 's1', 'p1', 'o1' ]]);
      });
      it('should empty array when no triple matches given triple (not pattern)', () => {
        expect(joinSampler.lookUpIndex(sampleArrayIndex, 's1', 'p3', 'o1')).toEqual(
          []);
      })
    });
    describe('countTriplesJoinRelation', () => {
      it('should return the number of triples that match specific triple', () => {
        expect(joinSampler.countTriplesJoinRelation(sampleArrayIndex, 's1', 'p1', 'o1'))
          .toEqual([1]);
      });
      it('should return an empty array when a triple is not in the index', () => {
        expect(joinSampler.countTriplesJoinRelation(sampleArrayIndex, 's6', 'p1', 'o1'))
          .toEqual([0]);
      });
      it('should return the number of triples matching predicate - object', () => {
        expect(joinSampler.countTriplesJoinRelation(sampleArrayIndex, null, 'p1', 'o2'))
          .toEqual([2]);
      });
      it('should return the number of triples matching predicate - object without result', () => {
        expect(joinSampler.countTriplesJoinRelation(sampleArrayIndex, null, 'p4', 'o2'))
          .toEqual([0]);
      });
      it('should return the number of triples matching subject - object', () => {
        expect(joinSampler.countTriplesJoinRelation(sampleArrayIndex, 's1', null, 'o2'))
          .toEqual([1]);
      });
      it('should return the number of triples matching subject - object without result', () => {
        expect(joinSampler.countTriplesJoinRelation(sampleArrayIndex, 's7', null, 'o2'))
          .toEqual([0]);
      });
      it('should return the number of triples matching subject - predicate', () => {
        expect(joinSampler.countTriplesJoinRelation(sampleArrayIndex, 's1', 'p2', null))
        .toEqual([1]);
      });
      it('should return the number of triples matching subject - predicate without result', () => {
        expect(joinSampler.countTriplesJoinRelation(sampleArrayIndex, 's1', 'p3', null))
        .toEqual([0]);
      });
      it('should return the number of triples matching subject - predicate without result', () => {
        expect(joinSampler.countTriplesJoinRelation(sampleArrayIndex, 's1', 'p3', null))
        .toEqual([0]);
      });
      it('should return the number of triples matching subject ', () => {
        expect(joinSampler.countTriplesJoinRelation(sampleArrayIndex, 's1', null , null))
        .toEqual([2, 1]);
      });
      it('should return the number of triples matching subject without result', () => {
        expect(joinSampler.countTriplesJoinRelation(sampleArrayIndex, 's4', null, null))
        .toEqual([0]);
      });
      it('should return the number of triples matching predicate ', () => {
        expect(joinSampler.countTriplesJoinRelation(sampleArrayIndex, null, 'p1', null))
        .toEqual([1,2,1]);
      });
      it('should return the number of triples matching predicate ', () => {
        expect(joinSampler.countTriplesJoinRelation(sampleArrayIndex, null, 'p2', null))
        .toEqual([1]);
      });
      it('should return the number of triples matching predicate without result', () => {
        expect(joinSampler.countTriplesJoinRelation(sampleArrayIndex, null, 'p3', null))
        .toEqual([0]);
      });
      it('should return the number of triples matching object ', () => {
        expect(joinSampler.countTriplesJoinRelation(sampleArrayIndex, null, null, 'o1'))
        .toEqual([1]);
      });
      it('should return the number of triples matching object without result', () => {
        expect(joinSampler.countTriplesJoinRelation(sampleArrayIndex, null, null, 'o5'))
        .toEqual([0]);
      });
      it('throws error when s, p, and o are all null', () => {
        // Expect the function to throw an error when called with null arguments
        expect(() => joinSampler.countTriplesJoinRelation(sampleArrayIndex, null, null, null)).toThrow(
          'Cant determine index of triple pattern with only variables',
        );
      });
    });
    describe('sampleIndex', () => {
      it('should sample correctly', () => { 
        spyOnMathRandom = jest.spyOn(global.Math, 'random').mockReturnValue(0.4);
        expect(joinSampler.sampleIndex('s1', null, null, 1, sampleArrayIndex)).toEqual(
          [['s1', 'p1', 'o2']]
      )});
    });
    describe('sampleTriplePatternsQuery', () => {
      let operation1: any;
      let operation2: any;
      let store: any; 
      beforeEach(() => {
        store = {
          _ids: {
            'pred1': 'p1',
            'pred2': 'p2',
          },
        };
        operation1 = {
          subject: { termType: 'Variable', value: 'v0' },
          predicate: {
            termType: 'NamedNode',
            value: 'pred1',
          },
          object: { termType: 'Variable', value: 'v2' },
        };
        operation2 = {
          subject: { termType: 'Variable', value: 'v1' },
          predicate: {
            termType: 'NamedNode',
            value: 'pred2',
          },
          object: { termType: 'Variable', value: 'v2' },
        };
      });

      it('should sample all patterns in query', () => {
        expect(joinSampler.sampleTriplePatternsQuery(1, store, {'': sampleArrayIndex}, 
          [operation1, operation2]
         )).toEqual([[{'v0': 's3', 'v2': 'o2'}], [{'v1':'s1', 'v2':'o3'}]])
      });
      it('should get all triples for n > number of triples', () => {
        expect(joinSampler.sampleTriplePatternsQuery(10, store, {'': sampleArrayIndex}, 
          [operation1, operation2]
         )).toEqual([[{'v0':'s1', 'v2':'o1'}, {'v0':'s3', 'v2':'o2'}, {'v0':'s1', 'v2':'o2'}, 
          {'v0':'s2', 'v2':'o4'}], [{'v1':'s1', 'v2':'o3'}]])
      });
    });
    describe('countCandidates', () => {
      let operation1: any;
      let operation2: any; 

      beforeEach(() => {
        operation1 = {
          subject: { termType: 'Variable', value: 'v0' },
          predicate: {
            termType: 'NamedNode',
            value: 'p1',
          },
          object: { termType: 'Variable', value: 'v1' },
        };
        operation2 = {
          subject: { termType: 'Variable', value: 'v0' },
          predicate: {
            termType: 'NamedNode',
            value: 'p2',
          },
          object: { termType: 'Variable', value: 'v2' },
        };

        // Redefine arrayIndex to get more extensive result sets
        sampleArrayIndex = {
          subjects: {
            s1: {
              p1: [ 'o1', 'o2', 'o5' ],
              p2: [ 'o3' ],
            },
            s2: {
              p1: [ 'o4', 'o5' ],
              p2: [ 'o1', 'o4' ]
            },
            s3: {
              p1: [ 'o2', 'o4' ],
            },
          },
          predicates: {
            p1: {
              o1: [ 's1' ],
              o2: [ 's3', 's1' ],
              o4: [ 's2', 's3' ],
              o5: [ 's1', 's2']
            },
            p2: {
              o3: [ 's1' ],
              o1: [ 's2' ],
              o4: [ 's2' ]
            },
          },
          objects: {
            o1: {
              s1: [ 'p1' ],
              s2: [ 'p2' ]
            },
            o2: {
              s1: [ 'p1' ],
              s3: [ 'p1' ],
            },
            o3: {
              s1: [ 'p2' ],
            },
            o4: {
              s2: [ 'p1', 'p2' ],
              s3: [ 'p1' ]
            },
            o5: {
              s1: [ 'p1' ],
              s2: [ 'p2' ]
            }
          },
        };
      });

      it('should correctly count', () => {
        const samples: Record<string,string>[] = joinSampler.sampleIndex(
          null, 'p2', null, Number.POSITIVE_INFINITY, sampleArrayIndex
        ).map((x: string[]) => {
          return {'v0': x[0], 'v2': x[2]}
        });
        expect(samples).toEqual(
          [{'v0': 's1' , 'v2': 'o3'},{'v0': 's2' , 'v2': 'o1'}, {'v0': 's2' , 'v2': 'o4'}]
        );  

        expect(joinSampler.candidateCounts(samples, null, 'p1', null, 's', 'v0', {'': sampleArrayIndex})).toEqual(
          {
            counts: [3, 2, 2],
            sampleRelations: [['s1', 'p1', null], ['s2', 'p1', null], ['s2', 'p1', null]]
          }
        )
      });
    });
    describe('sampleJoin', () => {
      let operation1: any;
      let operation2: any; 
      let operation3: any;
      let operation4: any;
      let operation5: any;
      let operation6: any;
      beforeEach(() => {
        operation1 = {
          subject: { termType: 'Variable', value: 'v0' },
          predicate: {
            termType: 'NamedNode',
            value: 'p1',
          },
          object: { termType: 'Variable', value: 'v1' },
        };
        operation2 = {
          subject: { termType: 'Variable', value: 'v0' },
          predicate: {
            termType: 'NamedNode',
            value: 'p2',
          },
          object: { termType: 'Variable', value: 'v2' },
        };
        operation3 = {
          subject: { termType: 'Variable', value: 'v1' },
          predicate: {
            termType: 'NamedNode',
            value: 'p1',
          },
          object: { termType: 'Variable', value: 'v2' },
        };
        operation4 = {
          subject: {
            termType: 'NamedNode',
            value: 's1',
          },
          predicate: { termType: 'Variable', value: 'p0' },
          object: { termType: 'Variable', value: 'v1' },
        };
        operation5 = {
          subject: {
            termType: 'NamedNode',
            value: 's2',
          },
          predicate: { termType: 'Variable', value: 'p0' },
          object: {
            termType: 'NamedNode',
            value: 'o1',
          },
        };
        operation6 = {
          subject: { termType: 'Variable', value: 'v2' },
          predicate: {
            termType: 'NamedNode',
            value: 'p2',
          },
          object: { termType: 'Variable', value: 'v3' },
        };
  
        // Redefine arrayIndex to get more extensive result sets
        sampleArrayIndex = {
          subjects: {
            s1: {
              p1: [ 'o1', 'o2', 'o5' ],
              p2: [ 'o3' , 's2', 's3'],
            },
            s2: {
              p1: [ 'o4', 'o5', 's3' ],
              p2: [ 'o1', 'o4' ]
            },
            s3: {
              p1: [ 'o2', 'o4' ],
            },
          },
          predicates: {
            p1: {
              s3: [ 's2' ],
              o1: [ 's1' ],
              o2: [ 's3', 's1' ],
              o4: [ 's2', 's3' ],
              o5: [ 's1', 's2']
            },
            p2: {
              s2: [ 's1' ],
              s3: [ 's1' ],
              o3: [ 's1' ],
              o1: [ 's2' ],
              o4: [ 's2' ]
            },
          },
          objects: {
            s2: {
              s1: [ 'p2' ]
            },
            s3: {
              s1: [ 'p2' ],
              s2: [ 'p1' ]
            },
            o1: {
              s1: [ 'p1' ],
              s2: [ 'p2' ]
            },
            o2: {
              s1: [ 'p1' ],
              s3: [ 'p1' ],
            },
            o3: {
              s1: [ 'p2' ],
            },
            o4: {
              s2: [ 'p1', 'p2' ],
              s3: [ 'p1' ]
            },
            o5: {
              s1: [ 'p1' ],
              s2: [ 'p2' ]
            }
          },
        };
      });
      function arraysHaveSameElements(arr1: any[], arr2: any[]): boolean {
        return arr1.length === arr2.length && arr1.every(element => arr2.includes(element));
    }
      it('should sample star-subject joins', () => {
        // Get all triple in triple pattern
        const samples: Record<string,string>[] = joinSampler.sampleIndex(
          null, 'p2', null, Number.POSITIVE_INFINITY, sampleArrayIndex
        ).map((x: string[]) => {
          return {'v0': x[0], 'v2': x[2]}
        });

        expect(samples).toEqual(
          [ {'v0': 's1' , 'v2': 's2'}, {'v0': 's1' , 'v2': 's3'}, {'v0': 's1' , 'v2': 'o3'}, 
            {'v0': 's2' , 'v2': 'o1'}, {'v0': 's2' , 'v2': 'o4'},
          ]
        );  
        const spyOnSampleArray = jest.spyOn(joinSampler, 'sampleArray').mockReturnValue([3, 4]);
        const output = joinSampler.sampleJoin(2, samples, null, "p1", null, operation1, 'v0', 's', {'': sampleArrayIndex});
        expect(output.sample).toEqual([{'v0': 's1', 'v1': 'o5', 'v2':'s2'}, {'v0': 's1', 'v1': 'o1', 'v2': 's3'}]);
        expect(output.selectivity).toBeCloseTo(3);
      });

      it('should sample subject - object joins', () => {
        const samples: Record<string,string>[] = joinSampler.sampleIndex(
          null, 'p1', null, Number.POSITIVE_INFINITY, sampleArrayIndex
        ).map((x: string[]) => {
          return {'v0': x[0], 'v1': x[2]}
        });
        expect(samples).toEqual(
          [{'v0': 's2' , 'v1': 's3'} ,{'v0': 's1' , 'v1': 'o1'}, {'v0': 's3' , 'v1': 'o2'}, {'v0': 's1' , 'v1': 'o2'},
            {'v0': 's2' , 'v1': 'o4'}, {'v0': 's3' , 'v1': 'o4'}, {'v0': 's1' , 'v1': 'o5'}, {'v0': 's2' , 'v1': 'o5'}]
        );
        const spyOnSampleArray = jest.spyOn(joinSampler, 'sampleArray').mockReturnValue([1]);
        const output = joinSampler.sampleJoin(1, samples, null, 'p1', null, operation3, 'v1', 's', {'': sampleArrayIndex});
        // TODO MAKE BETTER TEST CASE WITH BETTER GRAPH
        expect(output.sample).toEqual([{'v0': 's2', 'v1': 's3', 'v2': 'o2'}])
      });
      it('should sample subject - predicate joins', () => {

      });
      it('should sample predicate joins', () => {

      });
      it('should sample subject - predicate joins', () => {

      });
      it('should sample predicate - object joins', () => {

      });
      it('should sample object star joins', () => {

      });
      it('should sample object-subject joins', () => {

      });
      it('should sample object-predicate joins', () => {

      });
      it('should sample with samples consisting of more than one triple pattern', () => {

      });
      it('should correctly sample empty', () => {

      });

    });

    describe('bottomUpEnumeration', () => {});
  });
});