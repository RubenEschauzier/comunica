import type * as RDF from '@rdfjs/types';
import { uniqTerms } from 'rdf-terms';
import { Algebra, Util } from 'sparqlalgebrajs';

/**
 * Get all variables inside the given expression.
 * @param expression An expression.
 * @return An array of variables, which can be empty.
 */
export function getExpressionVariables(expression: Algebra.Expression): RDF.Variable[] {
  switch (expression.expressionType) {
    case Algebra.expressionTypes.AGGREGATE:
    case Algebra.expressionTypes.WILDCARD:
      throw new Error(`Getting expression variables is not supported for ${expression.expressionType}`);
    case Algebra.expressionTypes.EXISTENCE:
      return Util.inScopeVariables(expression.input);
    case Algebra.expressionTypes.NAMED:
      return [];
    case Algebra.expressionTypes.OPERATOR:
      return uniqTerms(expression.args.flatMap(arg => getExpressionVariables(arg)));
    case Algebra.expressionTypes.TERM:
      if (expression.term.termType === 'Variable') {
        return [ expression.term ];
      }
      return [];
  }
}

export function getOperatorTerms(operation: Algebra.Operation): Set<string> {
  function termToString(term: RDF.Term, position?: string): string {
    const prefix = position ? `${position}:` : '';
    
    if (term.termType === 'NamedNode') {
      return `${prefix}iri:${term.value}`;
    } else if (term.termType === 'Literal') {
      // Include datatype and language to distinguish different literals
      const lang = term.language ? `@${term.language}` : '';
      const datatype = term.datatype ? `^^${term.datatype.value}` : '';
      return `${prefix}literal:${term.value}${lang}${datatype}`;
    } else if (term.termType === 'BlankNode') {
      return `${prefix}bnode:${term.value}`;
    } else if (term.termType === 'Quad') {
      // Handle RDF-star quoted triples
      return `${prefix}quad:${term.value}`;
    }
    // Skip variables - caller handles those separately
    return '';
  }

  function addTerm(term: RDF.Term, position?: string) {
    const termStr = termToString(term, position);
    if (termStr) {
      terms.add(termStr);
    }
  }

  function extractFromPath(path: Algebra.PropertyPathSymbol) {
    if (path.type === 'link') {
      addTerm(path.iri, 'predicate');
    } else if ('path' in path) {
      extractFromPath(path.path);
    } else if ('left' in path && 'right' in path) {
      extractFromPath(path.left);
      extractFromPath(path.right);
    }
  }

  function extractFromExpression(expr: Algebra.Expression) {
    switch (expr.expressionType) {
      case Algebra.expressionTypes.TERM:
        if (expr.term.termType !== 'Variable') {
          addTerm(expr.term);
        }
        break;
      case Algebra.expressionTypes.OPERATOR:
        expr.args.forEach(arg => extractFromExpression(arg));
        break;
      case Algebra.expressionTypes.EXISTENCE:
        // Recursively extract from the subquery
        Util.recurseOperation(expr.input, {
          pattern: (op) => {
            addTerm(op.subject, 'subject');
            addTerm(op.predicate, 'predicate');
            addTerm(op.object, 'object');
            if (op.graph && op.graph.termType !== 'DefaultGraph') {
              addTerm(op.graph, 'graph');
            }
            return false; // Already handled
          },
        });
        break;
    }
  }

  const terms = new Set<string>();

  Util.recurseOperation(operation, {
    pattern: (op) => {
      addTerm(op.subject, 'subject');
      addTerm(op.predicate, 'predicate');
      addTerm(op.object, 'object');
      if (op.graph && op.graph.termType !== 'DefaultGraph') {
        addTerm(op.graph, 'graph');
      }
      return false;
    },

    expression: (op) => {
      extractFromExpression(op);
      return false;
    },

    path: (op) => {
      addTerm(op.subject, 'subject');
      addTerm(op.object, 'object');
      extractFromPath(op.predicate);
      return false;
    },
    
    graph: (op) => {
      addTerm(op.name, 'graph');
      return true;
    },
    
    service: (op) => {
      addTerm(op.name, 'service');
      return true;
    },
    
    values: (op) => {
      for (const binding of op.bindings) {
        for (const term of Object.values(binding)) {
          addTerm(term);
        }
      }
      return false;
    },
    
    filter: (op) => {
      extractFromExpression(op.expression);
      return true;
    },
    
    extend: (op) => {
      extractFromExpression(op.expression);
      return true;
    },
  
  });

  return terms;
}
