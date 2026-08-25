import { KeysRdfUpdateQuads } from '@comunica/context-entries';
import type { FragmentSelectorShape, IActionContext, IDataDestination, IQuerySourceWrapper } from '@comunica/types';
import { Algebra, algebraUtils, isKnownOperation, isKnownSubType } from '@comunica/utils-algebra';
import { getDataDestinationValue } from './Utils';
import type * as RDF from '@rdfjs/types';

/**
 * Check if the given shape accepts the given query operation.
 * @param shape A shape to test the query operation against.
 * @param operation A query operation to test.
 * @param options Additional options to consider.
 * @param options.joinBindings If additional bindings will be pushed down to the source for joining.
 * @param options.filterBindings If additional bindings will be pushed down to the source for filtering.
 */
export function doesShapeAcceptOperation(
  shape: FragmentSelectorShape,
  operation: Algebra.Operation,
  options?: FragmentSelectorShapeTestFlags,
): boolean {
  return doesShapeAcceptOperationRecurseShape(shape, shape, operation, options);
}

function doesShapeAcceptOperationRecurseShape(
  shapeTop: FragmentSelectorShape,
  shapeActive: FragmentSelectorShape,
  operation: Algebra.Operation,
  options?: FragmentSelectorShapeTestFlags,
): boolean {
  // Recurse into the shape
  if (shapeActive.type === 'conjunction') {
    return shapeActive.children
      .every(child => doesShapeAcceptOperationRecurseShape(shapeTop, child, operation, options));
  }
  if (shapeActive.type === 'disjunction') {
    return shapeActive.children
      .some(child => doesShapeAcceptOperationRecurseShape(shapeTop, child, operation, options));
  }
  if (shapeActive.type === 'negation') {
    return !doesShapeAcceptOperationRecurseShape(shapeActive.child, shapeActive.child, operation, options);
  }
  if (shapeActive.type === 'arity') {
    return doesShapeAcceptOperationRecurseShape(shapeTop, shapeActive.child, operation, options);
  }

  // Validate options
  if ((options?.joinBindings && !shapeActive.joinBindings) ??
    (options?.filterBindings && !shapeActive.filterBindings)) {
    return false;
  }

  // Check if the shape's operation matches with the given operation
  const shapeOperation = shapeActive.operation;
  switch (shapeOperation.operationType) {
    case 'type': {
      if (shapeOperation.type === Algebra.Types.EXPRESSION && isExtensionFunction(operation) &&
        !('extensionFunctions' in shapeOperation &&
        shapeOperation.extensionFunctions?.includes(operation.name.value))) {
        return false;
      }
      if (!doesShapeAcceptOperationRecurseOperationAndShape(shapeTop, shapeActive.children, operation, options) &&
        !doesShapeAcceptOperationRecurseOperation(shapeTop, operation, options)) {
        return false;
      }
      return shapeOperation.type === operation.type;
    }
    case 'pattern': {
      if (!doesShapeAcceptOperationRecurseOperationAndShape(shapeTop, shapeActive.children, operation, options) &&
        !doesShapeAcceptOperationRecurseOperation(shapeTop, operation, options)) {
        return false;
      }
      return shapeOperation.pattern.type === operation.type;
    }
    case 'wildcard': {
      // All possible operations are accepted by this shape.
      // As exception, extension functions are not accepted through wildcards.
      if (options?.wildcardAcceptAllExtensionFunctions) {
        return true;
      }
      if (isExtensionFunction(operation)) {
        return false;
      }
      // Also check for nested extension functions,
      // and only accept the wildcard if all nested extension functions are supported by the query shape.
      let hasUnsupportedExtensionFunction = false;
      algebraUtils.visitOperation(operation, {
        [Algebra.Types.EXPRESSION]: {
          visitor: (subOp) => {
            if (isExtensionFunction(subOp) && !doesShapeAcceptOperation(shapeTop, subOp, options)) {
              hasUnsupportedExtensionFunction = true;
              return false;
            }
            return true;
          },
        },
      });
      return !hasUnsupportedExtensionFunction;
    }
  }
}

function doesShapeAcceptOperationRecurseOperationAndShape(
  shapeTop: FragmentSelectorShape,
  shapeActiveChildren: FragmentSelectorShape[] | undefined,
  operation: Algebra.Operation,
  options?: FragmentSelectorShapeTestFlags,
): boolean {
  if (isExtensionFunction(operation) || isExtensionFunction((<any> operation).expression)) {
    return false;
  }
  if (shapeActiveChildren) {
    const operationCast = <Algebra.Operation & { input?: unknown; patterns?: any[] }> operation;
    const operationInputs: Algebra.Operation[] = operationCast.input ?
        (Array.isArray(operationCast.input) ? operationCast.input : [ operationCast.input ]) :
      operationCast.patterns ?? [];
    for (const [ i, shapeActiveChild ] of shapeActiveChildren.entries()) {
      if (!operationInputs[i] ||
        !doesShapeAcceptOperationRecurseShape(shapeTop, shapeActiveChild, operationInputs[i], options)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

function doesShapeAcceptOperationRecurseOperation(
  shapeTop: FragmentSelectorShape,
  operation: Algebra.Operation,
  options?: FragmentSelectorShapeTestFlags,
): boolean {
  // Recurse into the operation, and restart from the top-level shape
  const operationCast = <Algebra.Operation & { input?: unknown; patterns?: any[]; expression?: any }> operation;
  if (operationCast.input) {
    const inputs: Algebra.Operation[] = Array
      .isArray(operationCast.input) ?
      operationCast.input :
        [ operationCast.input ];
    if (!inputs.every(input => doesShapeAcceptOperationRecurseShape(shapeTop, shapeTop, input, options))) {
      return false;
    }
  }
  if (operationCast.expression && isExtensionFunction(operationCast.expression) &&
      !doesShapeAcceptOperationRecurseShape(shapeTop, shapeTop, operationCast.expression, options)) {
    return false;
  }
  return !(operationCast.patterns && !operationCast.patterns
    .every((input: Algebra.Pattern) => doesShapeAcceptOperationRecurseShape(shapeTop, shapeTop, input, options)));
}

function isStandardSparqlFunction(iri: string): boolean {
  return /^https?:\/\/www\.w3\.org\//u.test(iri);
}

function isExtensionFunction(operation: Algebra.Operation): operation is Algebra.NamedExpression {
  return operation && operation.type === Algebra.Types.EXPRESSION &&
    isKnownSubType(operation, Algebra.ExpressionTypes.NAMED) && !isStandardSparqlFunction(operation.name.value);
}

export type FragmentSelectorShapeTestFlags = {
  joinBindings?: boolean;
  filterBindings?: boolean;
  wildcardAcceptAllExtensionFunctions?: boolean;
};

export async function passFullOperationToSource(
  operation: Algebra.Operation,
  sources: IQuerySourceWrapper[],
  context: IActionContext,
): Promise<boolean> {
  if (sources.length === 1) {
    const sourceWrapper = sources[0];
    const destination: IDataDestination | undefined = context.get(KeysRdfUpdateQuads.destination);
    if (!destination || sourceWrapper.source.referenceValue === getDataDestinationValue(destination)) {
      try {
        const shape = await sourceWrapper.source.getSelectorShape(context);
        if (doesShapeAcceptOperation(shape, operation)) {
          return true;
        }
      } catch {
        // Fallback to the default case when the selector shape does not exist,
        // which can occur for a non-existent destination.
      }
    }
  }
  return false;
}

export function canAnswerBgp(
  shape: FragmentSelectorShape,
  operation: Algebra.Bgp,
  optionalVars: RDF.Variable[],
  requiredVars: RDF.Variable[],
): boolean {
  if (shape.type !== 'operation' || shape.operation.operationType !== 'pattern') {
    return false;
  }

  if (!isKnownOperation(shape.operation.pattern, Algebra.Types.BGP)) {
    return false;
  }

  const shapePatterns = shape.operation.pattern.patterns;
  const queryPatterns = operation.patterns;

  if (shapePatterns.length !== queryPatterns.length) {
    return false;
  }

  const shapePatternsOrdered = [...shapePatterns].sort(
    (a, b) => getConstraintScore(b, optionalVars, requiredVars) - getConstraintScore(a, optionalVars, requiredVars),
  );
  
  const queryPatternsOrdered = [...queryPatterns].sort(
    (a, b) => getConstraintScore(b, optionalVars, requiredVars) - getConstraintScore(a, optionalVars, requiredVars),
  );

  return matchPatternsRecursive(shapePatternsOrdered, queryPatternsOrdered, requiredVars, new Map());
}

function matchPatternsRecursive(
  patterns1: Algebra.Pattern[],
  patterns2: Algebra.Pattern[],
  requiredVars: RDF.Variable[],
  bindings: Map<string, RDF.Term>,
): boolean {
  if (patterns1.length !== patterns2.length) {
    return false;
  }

  if (patterns1.length === 0) {
    return true;
  }

  const currentPattern = patterns1[0];

  for (let i = 0; i < patterns2.length; i++) {
    const candidatePattern = patterns2[i];
    const localBindings = new Map(bindings);

    if (tryMatchTerms(currentPattern, candidatePattern, requiredVars, localBindings)) {
      const remainingPatterns1 = patterns1.slice(1);
      const remainingPatterns2 = patterns2.filter((_, idx) => idx !== i);

      if (matchPatternsRecursive(remainingPatterns1, remainingPatterns2, requiredVars, localBindings)) {
        return true;
      }
    }
  }

  return false;
}

function tryMatchTerms(
  shapePat: Algebra.Pattern,
  queryPat: Algebra.Pattern,
  requiredVars: RDF.Variable[],
  bindings: Map<string, RDF.Term>,
): boolean {
  const positions: (keyof Algebra.Pattern)[] = ['subject', 'predicate', 'object', 'graph'];

  for (const pos of positions) {
    const sTerm = shapePat[pos] as RDF.Term;
    const qTerm = queryPat[pos] as RDF.Term;

    if (sTerm.termType !== 'Variable') {
      if (!sTerm.equals(qTerm)) {
        return false;
      }
      continue;
    }

    const isRequired = requiredVars.some(v => v.equals(sTerm));
    if (isRequired && qTerm.termType === 'Variable') {
      return false;
    }

    // Ensure join consistency for shared variables across patterns
    const varKey = sTerm.value;
    if (bindings.has(varKey)) {
      if (!bindings.get(varKey)!.equals(qTerm)) {
        return false;
      }
    } else {
      bindings.set(varKey, qTerm);
    }
  }

  return true;
}

function getConstraintScore(
  pattern: Algebra.Pattern,
  optionalVars: RDF.Variable[],
  requiredVars: RDF.Variable[],
): number {
  let score = 0;
  for (const term of [pattern.subject, pattern.predicate, pattern.object, pattern.graph]) {
    if (term.termType !== 'Variable' || requiredVars.some(required => required.equals(term))) {
      score += 3;
    } else if (optionalVars.some(optional => optional.equals(term))) {
      score += 2;
    } else {
      score += 1;
    }
  }
  return score;
}