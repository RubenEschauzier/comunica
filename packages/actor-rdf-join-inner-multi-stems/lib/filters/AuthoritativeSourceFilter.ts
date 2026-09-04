import { Bindings } from "@comunica/utils-bindings-factory";
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
    protected readonly sourceExtractor: (binding: Bindings) => string[]
  ) {}

  /**
   * Registers a composite resource rule on a domain prefix.
   * Flushes the source cache and resets the burst pointer in O(1).
   */
  public registerResourceFilter(
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
   * Checks whether a term's URI value resides within the given domain prefix.
   * Exposed so callers can resolve constant (non-Variable) terms once, at registration time,
   * instead of registering a per-binding obligation for something that never varies.
   */
  public isWithinDomain(uri: string, domainPrefix: string): boolean {
    return this.uriTrie.matchesPrefix(uri, domainPrefix);
  }

  /**
   * Evaluates whether a binding should be suppressed based on active composite rules.
   */
  public shouldFilter(binding: Bindings): boolean {
    if (!this.hasRules) {
      return false;
    }

    const source = this.sourceExtractor(binding);

    if (source.length !== 1){
      console.log(this)
      console.log([...(<any>binding).entries.entries()]);
      throw Error(`${this.constructor.name} expects one source per binding, got: 
        ${JSON.stringify(source, null, 2)}`);
    }
    const sourceString = source[0];
    // Resolve candidate rules covering this source
    let rules: ICompositeRule[];

    if (sourceString === this.lastSource && this.lastRules !== null) {
      rules = this.lastRules;
    } else {
      const cached = this.sourceRulesCache.get(sourceString);
      if (cached !== undefined) {
        rules = cached;
      } else {
        rules = this.uriTrie.getAllMatchingRules(sourceString);
        this.sourceRulesCache.set(sourceString, rules);
      }
      this.lastSource = sourceString;
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