import {
  HINT_OBJECT,
  HINT_PREDICATE,
  HINT_SUBJECT,
} from '@comunica/actor-optimize-query-operation-query-hint-join-order-fixed';
import { TypesComunica } from '@comunica/utils-algebra';
import { toAlgebra12Builder } from '@traqula/algebra-sparql-1-2';
import { createAlgebraContext } from '@traqula/algebra-transformations-1-2';
import type { Algebra, ContextConfigs } from '@traqula/algebra-transformations-1-2';
import type { IndirDef } from '@traqula/core';
import { IndirBuilder } from '@traqula/core';
import type {
  GraphNode,
  Path,
  Pattern,
  PatternBgp,
  PatternFilter,
  PatternGroup,
  SparqlQuery,
  TermIri,
  TermIriFull,
  TermIriPrefixed,
  TermVariable,
  TripleNesting,
} from '@traqula/rules-sparql-1-2';

// Get original rules from the 1.2 builder
const origTranslateGraphPattern = toAlgebra12Builder.getRule('translateGraphPattern');
const origTranslateExpression = toAlgebra12Builder.getRule('translateExpression');
const origTranslateBgp = toAlgebra12Builder.getRule('translateBgp');

type HintedAlgebraContext = ReturnType<typeof createAlgebraContext> & {
  hintedMode?: boolean;
  topLevelGroupProcessed?: boolean;
};

type BasicPatternEntry = PatternBgp['triples'][number];

/**
 * Checks if a TripleNesting in the AST represents the query hint triple.
 * The hint triple has the form: comunica:hint comunica:optimizer "None" .
 *
 * In the AST, prefixed names have `prefix` and `value` (local name) fields.
 * We check for both the prefix-resolved form and the raw prefix form.
 */
function isHintTriple(triple: TripleNesting, prefixes: Record<string, string>): boolean {
  const s = triple.subject;
  const p: TermIri | TermVariable | Path = triple.predicate;
  const o = triple.object;

  if (!(o?.subType === 'literal' && o.value === HINT_OBJECT)) {
    return false;
  }

  return matchesHintIri(s, HINT_SUBJECT, prefixes) &&
    isIriTerm(p) && matchesHintIri(p, HINT_PREDICATE, prefixes);
}

function isTripleNesting(entry: BasicPatternEntry): entry is TripleNesting {
  return entry.type === 'triple';
}

export function isIriTerm(node: TermIri | TermVariable | Path): node is TermIriFull | TermIriPrefixed {
  return node.type === 'term' && node.subType === 'namedNode';
}

/**
 * Checks if an AST term matches a hint IRI.
 * Handles both full IRIs and prefixed names.
 */
function matchesHintIri(term: GraphNode, expectedFullIri: string, prefixes: Record<string, string>): boolean {
  if (term?.subType !== 'namedNode') {
    return false;
  }
  // Full IRI match
  if (term.value === expectedFullIri) {
    return true;
  }
  // If prefix doesn't exist on a term and it doesn't match above
  // it won't match
  if (!('prefix' in term)) {
    return false;
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
function containsHintTriple(patterns: Pattern[], prefixes: Record<string, string>): boolean {
  return patterns.some(
    (pattern): boolean => pattern.subType === 'bgp' &&
      pattern.triples.some((entry: BasicPatternEntry) => isTripleNesting(entry) && isHintTriple(entry, prefixes)),
  );
}

/**
 * Custom translateGraphPattern that detects the query hint triple in the top-level group graph pattern
 * and produces hinted-group algebra operations instead of flattened BGP/Join operations.
 *
 * When the hint triple is found in a top-level group BGP, hintedMode is enabled on the context,
 * so all BGPs in nested groups also preserve their syntactic structure.
 */
const customTranslateGraphPattern: IndirDef<
  HintedAlgebraContext,
  'translateGraphPattern',
  Algebra.Operation,
  [Pattern]
> = {
  name: 'translateGraphPattern',
  fun: ({ SUBRULE }) => (context: HintedAlgebraContext, pattern: Pattern): Algebra.Operation => {
    const { astFactory: F, algebraFactory: AF, currentPrefixes } = context;

    const translateBgpAsHintedGroup = (bgpPattern: PatternBgp, stripHint: boolean): Algebra.Operation => {
      const entries = stripHint ?
        bgpPattern.triples.filter((entry: BasicPatternEntry): boolean => !isTripleNesting(entry) ||
          !isHintTriple(entry, currentPrefixes)) :
        bgpPattern.triples;

      if (entries.length === 0) {
        return AF.createBgp([]);
      }

      const input: Algebra.Operation[] = entries
        .map((entry: BasicPatternEntry): Algebra.Operation => SUBRULE(origTranslateBgp, {
          ...bgpPattern,
          triples: [ entry ],
        }));

      return <Algebra.Operation> <unknown> { type: TypesComunica.HINTED_GROUP, input };
    };

    if (F.isPatternGroup(pattern)) {
      const groupPattern: PatternGroup = pattern;
      const isTopLevelGroup = !context.topLevelGroupProcessed;
      if (isTopLevelGroup) {
        context.topLevelGroupProcessed = true;
        context.hintedMode = containsHintTriple(groupPattern.patterns, currentPrefixes);
      }

      if (context.hintedMode) {
        const filters: PatternFilter[] = [];
        const nonfilters: Pattern[] = [];
        for (const subPattern of groupPattern.patterns) {
          if (subPattern.subType === 'filter') {
            filters.push(subPattern);
          } else {
            nonfilters.push(subPattern);
          }
        }

        const translatedPatterns: Algebra.Operation[] = nonfilters
          .map((subPattern: Pattern): Algebra.Operation => {
            if (subPattern.subType === 'bgp') {
              // Hint triple removal only applies at top-level BGPs.
              // Nested BGPs preserve all triples.
              const stripHint = isTopLevelGroup;
              return translateBgpAsHintedGroup(subPattern, stripHint);
            }
            return SUBRULE(customTranslateGraphPattern, subPattern);
          })
          .filter((operation: Algebra.Operation): boolean => !(operation.type === 'bgp' &&
            operation.patterns.length === 0));

        let result: Algebra.Operation;
        if (translatedPatterns.length === 0) {
          result = AF.createBgp([]);
        } else if (translatedPatterns.length === 1) {
          result = translatedPatterns[0];
        } else {
          result = <Algebra.Operation> <unknown> { type: TypesComunica.HINTED_GROUP, input: translatedPatterns };
        }

        if (filters.length > 0) {
          const expressions = filters
            .map((filter: PatternFilter) => SUBRULE(origTranslateExpression, filter.expression));
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
    return origTranslateGraphPattern.fun({ SUBRULE })(context, pattern);
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
  query: SparqlQuery,
  options: ContextConfigs = {},
): Algebra.Operation {
  const c: HintedAlgebraContext = createAlgebraContext(options);
  const transformer = hintedGroupAlgebraBuilder.build();
  return transformer.translateQuery(c, query, options.quads, options.blankToVariable);
}
