import { HINT_OBJECT, HINT_PREDICATE, HINT_SUBJECT } from
  '@comunica/actor-optimize-query-operation-query-hint-join-order-fixed';
import { TypesComunica } from '@comunica/utils-algebra';
import { toAlgebra12Builder } from '@traqula/algebra-sparql-1-2';
import { createAlgebraContext } from '@traqula/algebra-transformations-1-2';
import type { Algebra } from '@traqula/algebra-transformations-1-2';
import type { IndirDef } from '@traqula/core';
import { IndirBuilder } from '@traqula/core';
import type { TripleNesting } from '@traqula/rules-sparql-1-2';

// Get original rules from the 1.2 builder
const origTranslateGraphPattern = toAlgebra12Builder.getRule('translateGraphPattern');
const origTranslateExpression = toAlgebra12Builder.getRule('translateExpression');
const origTranslateBgp = toAlgebra12Builder.getRule('translateBgp');

/**
 * Checks if a TripleNesting in the AST represents the query hint triple.
 * The hint triple has the form: comunica:hint comunica:optimizer "None" .
 *
 * In the AST, prefixed names have `prefix` and `value` (local name) fields.
 * We check for both the prefix-resolved form and the raw prefix form.
 */
function isHintTriple(triple: TripleNesting, prefixes: Record<string, string>): boolean {
  const s: any = triple.subject;
  const p: any = triple.predicate;
  const o: any = triple.object;

  if (!(o?.subType === 'literal' && o.value === HINT_OBJECT)) {
    return false;
  }

  return matchesHintIri(s, HINT_SUBJECT, prefixes) &&
    matchesHintIri(p, HINT_PREDICATE, prefixes);
}

/**
 * Checks if an AST term matches a hint IRI.
 * Handles both full IRIs and prefixed names.
 */
function matchesHintIri(term: any, expectedFullIri: string, prefixes: Record<string, string>): boolean {
  if (term?.subType !== 'namedNode') {
    return false;
  }
  // Full IRI match
  if (term.value === expectedFullIri) {
    return true;
  }
  // Prefixed name: resolve prefix + local name
  if (term.prefix && prefixes[term.prefix]) {
    const resolved = prefixes[term.prefix] + term.value;
    return resolved === expectedFullIri;
  }
  return false;
}

/**
 * Checks if any BGP pattern in the given pattern list contains the hint triple.
 */
function containsHintTriple(patterns: any[], astFactory: any, prefixes: Record<string, string>): boolean {
  return patterns.some(
    p => astFactory.isPatternBgp(p) &&
      (p).triples?.some((t: TripleNesting) => isHintTriple(t, prefixes)),
  );
}

/**
 * Custom translateGraphPattern that detects the query hint triple in group graph patterns
 * and produces hinted-group algebra operations instead of flattened BGP/Join operations.
 *
 * When the hint triple is found in a group, hintedMode is enabled on the context,
 * so all nested groups also preserve their syntactic structure.
 */
const customTranslateGraphPattern: IndirDef<any, 'translateGraphPattern', any, [any]> = {
  name: 'translateGraphPattern',
  fun: ({ SUBRULE }) => (C: any, pattern: any) => {
    const { astFactory: F, algebraFactory: AF, currentPrefixes } = C;

    if (F.isPatternGroup(pattern)) {
      const hasHint = containsHintTriple((pattern).patterns, F, currentPrefixes || {});

      if (hasHint || C.hintedMode) {
        C.hintedMode = true;

        // Separate filters and non-filters
        const filters: any[] = [];
        const nonfilters: any[] = [];
        for (const subPattern of (pattern).patterns) {
          if (F.isPatternFilter(subPattern)) {
            filters.push(subPattern);
          } else {
            nonfilters.push(subPattern);
          }
        }

        // Translate each non-filter pattern, stripping hint triples from BGPs
        const translatedPatterns: any[] = [];
        for (const subPattern of nonfilters) {
          if (F.isPatternBgp(subPattern)) {
            const cleanedTriples = (subPattern).triples.filter(
              (t: TripleNesting) => !isHintTriple(t, currentPrefixes || {}),
            );
            if (cleanedTriples.length > 0) {
              translatedPatterns.push(SUBRULE(origTranslateBgp, { ...subPattern, triples: cleanedTriples }));
            }
          } else {
            // For nested groups and other patterns, translate recursively
            // This will use our override since SUBRULE resolves by name
            translatedPatterns.push(SUBRULE(customTranslateGraphPattern, subPattern));
          }
        }

        // Create the result
        let result: any;
        if (translatedPatterns.length === 0) {
          result = AF.createBgp([]);
        } else if (translatedPatterns.length === 1) {
          result = translatedPatterns[0];
        } else {
          result = { type: TypesComunica.HINTED_GROUP, input: translatedPatterns };
        }

        // Apply filters (same as standard SPARQL 1.1 Section 18.2.2.7)
        if (filters.length > 0) {
          const expressions = filters.map(
            (f: any) => SUBRULE(origTranslateExpression, f.expression),
          );
          let conjunction = expressions[0];
          for (const expression of expressions.slice(1)) {
            conjunction = AF.createOperatorExpression('&&', [ conjunction, expression ]);
          }
          result = AF.createFilter(result, conjunction);
        }

        return result;
      }
    }

    // Fall through to original implementation for non-hinted groups and other patterns
    return origTranslateGraphPattern.fun({ SUBRULE })(C, pattern);
  },
};

/**
 * Custom algebra builder that patches translateGraphPattern
 * to produce hinted-group operations when the query hint is present.
 */
export const hintedGroupAlgebraBuilder = IndirBuilder
  .create(toAlgebra12Builder)
  .patchRule(customTranslateGraphPattern);

/**
 * Translates a SPARQL query to algebra, with support for the query hint triple.
 * When the hint `comunica:hint comunica:optimizer "None" .` is found,
 * nested group graph patterns produce hinted-group operations instead of being flattened.
 */
export function toAlgebraWithHintedGroup(
  query: any,
  options: any = {},
): Algebra.Operation {
  const c: any = createAlgebraContext(options);
  const transformer = hintedGroupAlgebraBuilder.build();
  return (<any> transformer).translateQuery(c, query, options.quads, options.blankToVariable);
}
