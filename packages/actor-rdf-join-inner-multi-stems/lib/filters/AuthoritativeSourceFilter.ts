import { Bindings } from "@comunica/types";
import { ICompositeRule, SegmentedUriTrieFilter } from "./SegmentedUriTrieFilter";

export class AuthoritativeSourceFilter {
  // Caches applicable candidate rules per source URI
  public readonly sourceRulesCache: Map<string, ICompositeRule[]> = new Map();

  // Unified domain and rule trie
  protected readonly uriTrie: SegmentedUriTrieFilter<ICompositeRule> =
    new SegmentedUriTrieFilter<ICompositeRule>();

  protected hasRules: boolean = false;

  // Run-length temporal cache for quad bursts emitted from the same document
  private lastSource: string | null = null;
  private lastRules: ICompositeRule[] | null = null;

  public constructor(
    protected readonly sourceExtractor: (binding: Bindings) => string
  ) {}

  /**
   * Registers a composite resource rule on a domain prefix.
   * Flushes the source cache and resets the burst pointer in O(1).
   */
  public registerCompositeResource(
    domainPrefix: string,
    requiredAuthoritativeVars: string[]
  ): void {
    this.hasRules = true;
    this.lastSource = null;
    this.lastRules = null;
    this.sourceRulesCache.clear();

    this.uriTrie.addDomainRule(domainPrefix, {
      domainPrefix,
      requiredAuthoritativeVars,
    });
  }

  /**
   * Evaluates whether a binding should be suppressed based on active composite rules.
   */
  public shouldFilter(binding: Bindings): boolean {
    if (!this.hasRules) {
      return false;
    }

    const source = this.sourceExtractor(binding);

    // 1. Resolve candidate rules covering this source (Burst cache -> Map cache -> Trie)
    let rules: ICompositeRule[];

    if (source === this.lastSource && this.lastRules !== null) {
      rules = this.lastRules;
    } else {
      const cached = this.sourceRulesCache.get(source);
      if (cached !== undefined) {
        rules = cached;
      } else {
        rules = this.uriTrie.getAllMatchingRules(source);
        this.sourceRulesCache.set(source, rules);
      }
      this.lastSource = source;
      this.lastRules = rules;
    }

    // Fast-path: document source is not covered by any composite resource
    if (rules.length === 0) {
      return false;
    }

    // 2. Test candidate composite rules
    for (let i = 0; i < rules.length; i++) {
      const rule = rules[i];
      let allVarsMatch = true;

      for (let j = 0; j < rule.requiredAuthoritativeVars.length; j++) {
        const varName = rule.requiredAuthoritativeVars[j];
        const term = binding.get(varName);

        if (!term || !this.uriTrie.matchesPrefix(term.value, rule.domainPrefix)) {
          allVarsMatch = false;
          break;
        }
      }

      // If all required variables reside in the authoritative domain, filter out
      if (allVarsMatch) {
        return true;
      }
    }

    return false;
  }
}