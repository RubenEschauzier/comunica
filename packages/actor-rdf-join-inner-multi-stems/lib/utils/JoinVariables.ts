import type * as RDF from '@rdfjs/types';

/**
 * For each set of variables in `variableSets`, computes the distinct (sorted) intersections it
 * has with every other set. Two other sets producing the same intersection contribute only one
 * entry (exact array equality).
 *
 * This determines the hash keys a StemsOperatorStream must pre-index its tuples under: for it to
 * be probeable by another operator, it needs one key per distinct combination of variables it can
 * share with any other operator - the probe side computes the matching key from the arriving
 * intermediate result's own perspective (see BaseRouter#addOperatorIfOverlapping).
 *
 * `variableSets[i]` may represent a single join entry (as when called from
 * ActorRdfJoinMultiStems#getJoinVariables for base operators) or the union of several entries
 * merged into one composite resource (as when called from
 * StemsAdaptiveJoinComponent#computeJoinVariablesForSubset) - the intersection logic is identical
 * either way, since it only ever compares two variable sets against each other.
 */
export function computePairwiseJoinVariables(variableSets: RDF.Variable[][]): RDF.Variable[][][] {
  const result: RDF.Variable[][][] = [];

  for (let i = 0; i < variableSets.length; i++) {
    const outerVariablesByValue = new Map(variableSets[i].map(variable => [ variable.value, variable ]));
    const outerValues = [ ...outerVariablesByValue.keys() ];
    const overlapping: string[][] = [];

    for (let j = 0; j < variableSets.length; j++) {
      if (i === j) {
        continue;
      }
      const innerValues = new Set(variableSets[j].map(variable => variable.value));
      const intersection = outerValues.filter(value => innerValues.has(value)).sort();
      if (intersection.length === 0) {
        continue;
      }
      const alreadyPresent = overlapping.some(existing =>
        existing.length === intersection.length && existing.every((val, idx) => val === intersection[idx]));
      if (!alreadyPresent) {
        overlapping.push(intersection);
      }
    }

    result.push(overlapping.map(values => values.map(value => outerVariablesByValue.get(value)!)));
  }

  return result;
}
