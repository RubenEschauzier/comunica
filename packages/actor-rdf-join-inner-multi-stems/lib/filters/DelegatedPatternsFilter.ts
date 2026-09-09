import { Bindings } from "@comunica/utils-bindings-factory";
import { bitForIndex, isDisjointMask } from "../utils/BitUtils";

/**
 * Characters that may follow a subject domain without the term leaving it: a path
 * separator, a fragment, or a query. Checking one of these is what stops a domain from
 * capturing a sibling that merely shares a prefix of its name, such as `/posts`
 * capturing `/posts-archive`.
 */
const CHAR_SLASH = 47;
const CHAR_HASH = 35;
const CHAR_QUESTION = 63;

/**
 * Whether a domain already ends on a separator, in which case a value starting with it is
 * inside it and no boundary character has to be inspected. Worth hoisting out of a loop
 * when the same domain is tested against many values.
 */
export function isDomainDelimited(domain: string): boolean {
  const lastChar = domain.charCodeAt(domain.length - 1);
  return lastChar === CHAR_SLASH || lastChar === CHAR_HASH;
}

/**
 * Whether a term or document URI lies within a domain.
 * @param domainIsDelimited Pass the precomputed `isDomainDelimited(domain)` on paths that
 * test many values against one domain; it is derived per call otherwise.
 */
export function isWithinDomain(
  value: string,
  domain: string,
  domainIsDelimited: boolean = isDomainDelimited(domain),
): boolean {
  if (!value.startsWith(domain)) {
    return false;
  }
  if (domainIsDelimited || value.length === domain.length) {
    return true;
  }
  const boundary = value.charCodeAt(domain.length);
  return boundary === CHAR_SLASH || boundary === CHAR_HASH || boundary === CHAR_QUESTION;
}

export interface IDelegatedPatterns {
  // /**
  //  * Routing bits mask of all operators covered by the delegated
  //  */
  // blockMask: number;
  // /**
  //  * The index of the operator associated with the composite resource (CR)
  //  * answering the delegated patterns
  //  */
  // ownerOperatorIndex: number;
  // /**
  //  * The variables that must be within authoritative domain of the CR for the
  //  * associated bindings to be covered by the CR.
  //  */
  // anchorVars: string[];
  // /**
  //  * The authority scope of the CR: the space of subjects it answers for in full. A
  //  * mapping is claimed by the CR exactly when all of its anchors fall in this space, so
  //  * the space has to be one the CR is complete for. Where a CR aggregates only part of
  //  * the pod publishing it, this is that part and not the pod, since the pod hosts
  //  * subjects the CR never saw and whose mappings it therefore cannot supply.
  //  */
  // subjectDomain: string;
  // /**
  //  * The domains (selectors)
  //  */
  // domains: IParsedUri[];
  shouldFilter: (binding: Bindings, doneMask: number, crMask: number) => boolean;
}


export class DelegatedPatternsFilter implements IDelegatedPatterns {
  /**
   * Bit of the owning composite resource
   */
  private readonly ownerBit: number;

  /**
   * Whether the subject domain already ends on a separator, in which case a term
   * starting with it is inside it and no boundary character has to be inspected.
   */
  private readonly subjectDomainIsDelimited: boolean;

  public constructor(
    protected readonly blockMask: number,
    protected readonly ownerOperatorIndex: number,
    protected readonly anchorVars: string[],
    protected readonly sourceExtractor: (binding: Bindings) => string[],
    protected readonly subjectDomain: string,
  ){
    this.ownerBit = bitForIndex(ownerOperatorIndex);
    this.subjectDomainIsDelimited = isDomainDelimited(subjectDomain);
  }

  /**
   * Whether the mapping lies in the part of the block delegated to this composite
   * resource, and may therefore be discarded by the operator holding it.
  */
  public shouldFilter(binding: Bindings, doneMask: number, crMask: number): boolean {
    // Don't filter bindings touched by the CR that created this filter. 
    if (!isDisjointMask(crMask, this.ownerBit)){
      return false;
    }

    // If no overlap between block and the involved operators in the binding
    // we never filter.
    if (isDisjointMask(doneMask, this.blockMask)){
      return false;
    }

    // A blank node carries no domain to compare against, so its membership is settled by
    // the document it was read from instead of by its value.
    let hasBlankNodeAnchor = false;

    // An empty anchor set means every anchor was a constant discharged at registration,
    // in which case any mapping reaching this point is delegated.
    for (const anchor of this.anchorVars){
      const anchorBinding = binding.get(anchor);
      if (!anchorBinding){
        return false;
      }
      if (anchorBinding.termType === 'BlankNode'){
        hasBlankNodeAnchor = true;
        continue;
      }
      if (!this.withinSubjectDomain(anchorBinding.value)){
        return false;
      }
    }

    if (hasBlankNodeAnchor && !this.sourcesWithinSubjectDomain(binding)){
      return false;
    }

    return true;
  }

  /**
   * Whether a term or document URI lies within the subject domain.
  */
  public withinSubjectDomain(value: string): boolean {
    return isWithinDomain(value, this.subjectDomain, this.subjectDomainIsDelimited);
  }

  /**
   * Whether every document contributing to the mapping lies within the subject domain.
   * If the binding contains sources used to answer the triple pattern that are outside
   * this block, these sources may prevent pruning while the block is completely within
   * the subject domain. This can cause duplicates.
  */
  public sourcesWithinSubjectDomain(binding: Bindings): boolean {
    const sources = this.sourceExtractor(binding);
    if (sources.length === 0){
      return false;
    }
    for (const source of sources){
      if (!this.withinSubjectDomain(source)){
        return false;
      }
    }
    return true;
  }
}

