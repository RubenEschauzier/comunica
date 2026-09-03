import { Bindings } from "@comunica/utils-bindings-factory";
import { SegmentedUriTrieFilter } from "./SegmentedUriTrieFilter";

export class AuthoritativeSourceFilter {
  // Backed by native Map to avoid V8 dictionary deoptimizations
  public readonly sourceDecisionCache: Map<string, boolean> = new Map();
  public readonly contentDecisionCache: Map<string, boolean> = new Map();

  protected readonly uriTrieSource: SegmentedUriTrieFilter = new SegmentedUriTrieFilter();
  protected readonly uriTrieContent: SegmentedUriTrieFilter = new SegmentedUriTrieFilter();

  protected hasFilter: boolean = false;

  // Run-length temporal cache for quad bursts emitted from the same document
  private lastSource: string | null = null;
  private lastSourceDecision: boolean = false;

  public constructor(
    protected readonly sourceExtractors: (binding: Bindings) => string[],
    protected readonly valueExtractors: (binding: Bindings) => string[],
  ) {}

  /**
   * Registers a URI pattern for authoritative content namespaces.
   * Flushes the content cache in O(1) time.
   */
  public addFilterValue(uriPattern: string): void {
    this.hasFilter = true;
    this.contentDecisionCache.clear();
    this.uriTrieContent.addStringFilter(uriPattern);
  }

  /**
   * Registers a URI pattern for authoritative source namespaces.
   * Flushes the source cache and resets the temporal pointer in O(1) time.
   */
  public addFilterSource(uriPattern: string): void {
    this.hasFilter = true;
    this.lastSource = null;
    this.sourceDecisionCache.clear();
    this.uriTrieSource.addStringFilter(uriPattern);
  }

  /**
   * Evaluates whether a binding should be filtered out (suppressed)
   * because it originates from and references data already covered by an
   * authoritative composite resource.
   */
  public shouldFilter(binding: Bindings): boolean {
    if (!this.hasFilter){
        return false;
    }

    // Extraction of source attribution
    const sources = this.sourceExtractors(binding);
    if (sources.length !== 1) {
      throw new Error(
        `AuthoritativeSourceFilter expects exactly one source attribution per binding. Got: [${sources.join(', ')}]`
      );
    }
    const source = sources[0];

    // Determine if source is covered by a composite resource
    let shouldFilterSource: boolean;

    if (source === this.lastSource) {
      // Check for string equality
      shouldFilterSource = this.lastSourceDecision;
    } else {
      const cached = this.sourceDecisionCache.get(source);
      if (cached !== undefined) {
        // If we can get decision from cache use that
        shouldFilterSource = cached;
      } else {
        // Otherwise ask the trie
        shouldFilterSource = this.uriTrieSource.hasFilterMatching(source);
        this.sourceDecisionCache.set(source, shouldFilterSource);
      }
      // Update pointer
      this.lastSource = source;
      this.lastSourceDecision = shouldFilterSource;
    }

    // If binding source is not covered by composite resource return
    if (!shouldFilterSource) {
      return false;
    }

    // Extract values we should check 
    // (subject for star, subject and object for triple pattern in 'middle' of linear shape)
    const values = this.valueExtractors(binding);
    const contentKey = this.getContentKey(values);

    // Find cached decision
    const cachedContent = this.contentDecisionCache.get(contentKey);
    if (cachedContent !== undefined) {
      return cachedContent;
    }

    // Evaluate content values against the trie
    // All extracted values must match the composite resource's authority
    const shouldFilterValue = values.every((val) => this.uriTrieContent.hasFilterMatching(val));
    this.contentDecisionCache.set(contentKey, shouldFilterValue);

    return shouldFilterValue;
  }

  /**
   * Fast key generator avoiding heap-allocated array joins for 1-ary and 2-ary values.
   * Uses ASCII Unit Separator (\x1F) to prevent delimiter collision issues.
   */
  private getContentKey(values: string[]): string {
    const len = values.length;
    if (len === 1) {
      return values[0]; // Zero allocations: reuses existing string pointer
    }
    if (len === 2) {
      return `${values[0]}\x1F${values[1]}`;
    }
    return values.join('\x1F');
  }
}