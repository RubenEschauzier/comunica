import type * as RDF from '@rdfjs/types';
import type { Algebra } from '@comunica/utils-algebra';

/**
 * The canonical key of every operation compared so far. Operations are matched once per composite
 * source per component, and comparing whole algebra trees structurally every time is what made adding
 * a composite source cost hundreds of milliseconds.
 */
const operationKeys = new WeakMap<Algebra.Operation, string>();

function termKey(term: RDF.Term): string {
  if (term.termType === 'Quad') {
    return `<<${termKey(term.subject)} ${termKey(term.predicate)} ${termKey(term.object)} ${termKey(term.graph)}>>`;
  }
  const literal = <RDF.Literal> term;
  return `${term.termType}|${term.value}|${literal.language ?? ''}|${literal.datatype?.value ?? ''}`;
}

/**
 * The structure of an operation as a string, with RDF terms compared as RDF terms and the metadata of
 * an operation left out, as it differs between copies of the same operation.
 */
export function operationKey(operation: Algebra.Operation): string {
  let key = operationKeys.get(operation);
  if (key === undefined) {
    key = JSON.stringify(operation, (name, value) => {
      if (name === 'metadata') {
        return undefined;
      }
      if (value && typeof value === 'object' && typeof value.termType === 'string') {
        return termKey(value);
      }
      return value;
    });
    operationKeys.set(operation, key);
  }
  return key;
}

/**
 * Whether two operations are the same, compared structurally: the algebra objects handed around are
 * not always reference-identical to the ones the join entries were built from.
 */
export function operationsEqual(left: Algebra.Operation, right: Algebra.Operation): boolean {
  return left === right || operationKey(left) === operationKey(right);
}
