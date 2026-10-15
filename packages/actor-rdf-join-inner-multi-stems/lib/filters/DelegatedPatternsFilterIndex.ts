import type { Bindings } from '@comunica/utils-bindings-factory';
import { isDisjointMask } from '../utils/BitUtils';
import type { DelegatedPatternsFilter, IDelegatedPatterns } from './DelegatedPatternsFilter';

/**
 * The filters of one group: all registered for the same composite resource operator, over the same
 * block, with the same anchors. They only differ in the domain they are authoritative for, which in
 * practice means one filter per pod the composite resource was answered by.
 */
interface IFilterGroup {
  ownerBit: number;
  blockMask: number;
  anchorVars: string[];
  filters: DelegatedPatternsFilter[];
  /**
   * The first filter registered for each domain. A second filter for the same domain would decide
   * every mapping the same way, so it is never needed for a lookup.
   */
  byDomain: Map<string, DelegatedPatternsFilter>;
  /**
   * The distinct lengths of the domains.
   */
  lengths: number[];
}

/**
 * The delegated pattern filters of an operator, or of the controller, indexed by their domain.
 *
 * A mapping is dropped when any filter claims it. Every pod a traversal reaches adds a filter, and a
 * linear scan over them for every mapping makes each mapping cost more the further the traversal
 * gets. A filter can only claim a mapping whose anchor lies in its domain, though, so this looks up the
 * filters whose domain is a prefix of the anchor instead, and asks only those.
 *
 * A group with a single filter is still scanned, which is what a query over one pod has, as a lookup
 * costs more than checking that one filter. From the second filter on, the prefixes looked up are one
 * per distinct domain length: pod roots that share a naming scheme mostly share their length too, so
 * that is a single lookup. Where domains come in too many lengths for that, the prefixes ending on the
 * separators `isWithinDomain` accepts as a boundary are looked up instead.
 *
 * The filter found by a lookup still decides the mapping itself, so this claims exactly the mappings
 * the linear scan over the same filters would.
 */
export class DelegatedPatternsFilterIndex implements IDelegatedPatterns {
  /**
   * Up to this many filters, a group is scanned rather than looked up in.
   */
  public static readonly SCAN_THRESHOLD = 1;
  /**
   * Past this many distinct domain lengths, a group looks up by separator instead of by length.
   */
  public static readonly MAX_LENGTH_LOOKUPS = 4;

  private readonly groups = new Map<string, IFilterGroup>();
  private readonly groupList: IFilterGroup[] = [];
  private filterCount = 0;

  /**
   * The number of filters added.
   */
  public get size(): number {
    return this.filterCount;
  }

  public add(filter: DelegatedPatternsFilter): void {
    const key = `${filter.ownerBit}|${filter.blockMask}|${filter.anchorVars.join(' ')}`;
    let group = this.groups.get(key);
    if (!group) {
      group = {
        ownerBit: filter.ownerBit,
        blockMask: filter.blockMask,
        anchorVars: filter.anchorVars,
        filters: [],
        byDomain: new Map(),
        lengths: [],
      };
      this.groups.set(key, group);
      this.groupList.push(group);
    }
    group.filters.push(filter);
    if (!group.byDomain.has(filter.subjectDomain)) {
      group.byDomain.set(filter.subjectDomain, filter);
      if (!group.lengths.includes(filter.subjectDomain.length)) {
        group.lengths.push(filter.subjectDomain.length);
      }
    }
    this.filterCount++;
  }

  public shouldFilter(binding: Bindings, doneMask: number, crMask: number): boolean {
    for (const group of this.groupList) {
      // The masks are the same for every filter of the group, so they are tested once for all of them
      if (!isDisjointMask(crMask, group.ownerBit) || isDisjointMask(doneMask, group.blockMask)) {
        continue;
      }
      if (group.filters.length <= DelegatedPatternsFilterIndex.SCAN_THRESHOLD) {
        for (const filter of group.filters) {
          if (filter.shouldFilter(binding, doneMask, crMask)) {
            return true;
          }
        }
        continue;
      }
      if (this.groupShouldFilter(group, binding, doneMask, crMask)) {
        return true;
      }
    }
    return false;
  }

  protected groupShouldFilter(group: IFilterGroup, binding: Bindings, doneMask: number, crMask: number): boolean {
    // Without variable anchors, every filter of the group claims every mapping the masks let through
    if (group.anchorVars.length === 0) {
      return group.filters[0].shouldFilter(binding, doneMask, crMask);
    }

    // A filter only claims a mapping whose IRI anchors are all in its domain, and whose source documents
    // are when an anchor is a blank node. Its domain is therefore a prefix of the first IRI anchor, or of
    // the first source document if every anchor is a blank node
    let probe: string | undefined;
    for (const anchor of group.anchorVars) {
      const term = binding.get(anchor);
      if (!term) {
        // No filter claims a mapping with an unbound anchor
        return false;
      }
      if (term.termType !== 'BlankNode') {
        probe = term.value;
        break;
      }
    }
    if (probe === undefined) {
      const sources = group.filters[0].sourceExtractor(binding);
      if (sources.length === 0) {
        return false;
      }
      probe = sources[0];
    }

    if (group.lengths.length <= DelegatedPatternsFilterIndex.MAX_LENGTH_LOOKUPS) {
      for (const length of group.lengths) {
        if (length <= probe.length && this.claims(group, probe.slice(0, length), binding, doneMask, crMask)) {
          return true;
        }
      }
      return false;
    }

    if (this.claims(group, probe, binding, doneMask, crMask)) {
      return true;
    }
    for (let i = 0; i < probe.length; i++) {
      const char = probe.charCodeAt(i);
      // `/`, `#` and `?`: a domain ends right before one, or, ending on a separator itself, on it
      if ((char === 47 || char === 35 || char === 63) &&
        (this.claims(group, probe.slice(0, i), binding, doneMask, crMask) ||
          this.claims(group, probe.slice(0, i + 1), binding, doneMask, crMask))) {
        return true;
      }
    }
    return false;
  }

  /**
   * Whether the filter for this domain, if there is one, claims the mapping.
   */
  protected claims(group: IFilterGroup, domain: string, binding: Bindings, doneMask: number, crMask: number): boolean {
    const filter = group.byDomain.get(domain);
    return filter !== undefined && filter.shouldFilter(binding, doneMask, crMask);
  }
}
