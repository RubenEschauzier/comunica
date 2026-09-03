import { Bindings } from '@comunica/types';
import { AuthoritativeSourceFilter } from '../lib/filters/AuthoritativeSourceFilter';

function createMockBinding(entries: Record<string, string>): Bindings {
  const termsMap = new Map<string, { value: string }>();
  for (const [key, value] of Object.entries(entries)) {
    termsMap.set(key, { value });
  }

  return {
    get: (variable: string) => termsMap.get(variable),
    has: (variable: string) => termsMap.has(variable),
  } as unknown as Bindings;
}

describe('AuthoritativeSourceFilter', () => {
  let filter: AuthoritativeSourceFilter;
  const sourceExtractor = (binding: Bindings) => binding.get('_source')!.value;

  beforeEach(() => {
    filter = new AuthoritativeSourceFilter(sourceExtractor);
  });

  describe('An shouldFilter invocation', () => {
    describe('for uninitialized or empty state', () => {
      it('should return false when no composite rules have been registered', () => {
        const binding = createMockBinding({
          _source: 'https://pod.example/alice/card.ttl',
          s: 'https://pod.example/alice/card.ttl#me',
        });

        expect(filter.shouldFilter(binding)).toBe(false);
      });

      it('should return false when the source URI does not match any registered domain', () => {
        filter.registerCompositeResource('https://pod.example/alice/', ['s']);

        const binding = createMockBinding({
          _source: 'https://pod.example/bob/profile.ttl',
          s: 'https://pod.example/alice/item',
        });

        expect(filter.shouldFilter(binding)).toBe(false);
      });
    });

    describe('for star patterns (single authoritative variable)', () => {
      beforeEach(() => {
        filter.registerCompositeResource('https://pod.example/alice/', ['s']);
      });

      it('should return true when source and subject are both within the domain', () => {
        const binding = createMockBinding({
          _source: 'https://pod.example/alice/data/items.ttl',
          s: 'https://pod.example/alice/data/items.ttl#1',
          p: 'http://schema.org/name',
          o: 'Alice Item',
        });

        expect(filter.shouldFilter(binding)).toBe(true);
      });

      it('should return false when subject is external to the domain', () => {
        const binding = createMockBinding({
          _source: 'https://pod.example/alice/data/items.ttl',
          s: 'https://external-vocab.org/concepts/Item',
          p: 'http://schema.org/name',
          o: 'Alice Item',
        });

        expect(filter.shouldFilter(binding)).toBe(false);
      });

      it('should return false when required variable is absent from the binding', () => {
        const binding = createMockBinding({
          _source: 'https://pod.example/alice/data/items.ttl',
          p: 'http://schema.org/name',
          o: 'Alice Item',
        });

        expect(filter.shouldFilter(binding)).toBe(false);
      });
    });

    describe('for linear path patterns (multiple authoritative variables)', () => {
      beforeEach(() => {
        // Linear path (?s -> ?o1 -> ?o2) requiring ?s and intermediate hop ?o1 to be local
        filter.registerCompositeResource('https://pod.example/alice/', ['s', 'o1']);
      });

      it('should return true when all required variables are in the domain, ignoring leaf ?o2', () => {
        const binding = createMockBinding({
          _source: 'https://pod.example/alice/dataset.ttl',
          s: 'https://pod.example/alice/node1',
          o1: 'https://pod.example/alice/node2',
          o2: 'https://external.org/values/42', // Terminal leaf may be external
        });

        expect(filter.shouldFilter(binding)).toBe(true);
      });

      it('should return false if intermediate hop variable leaks outside the domain', () => {
        const binding = createMockBinding({
          _source: 'https://pod.example/alice/dataset.ttl',
          s: 'https://pod.example/alice/node1',
          o1: 'https://external-pod.example/bob/node2', // Intermediate hop external
          o2: 'https://pod.example/alice/node3',
        });

        expect(filter.shouldFilter(binding)).toBe(false);
      });
    });

    describe('for heterogeneous shape coexistence across domains', () => {
      beforeEach(() => {
        // Alice has a Star pattern
        filter.registerCompositeResource('https://pod.example/alice/', ['s']);
        // Bob has a Linear pattern
        filter.registerCompositeResource('https://pod.example/bob/', ['s', 'o1']);
      });

      it('should evaluate Alice bindings according to the Star rule', () => {
        const aliceBinding = createMockBinding({
          _source: 'https://pod.example/alice/profile.ttl',
          s: 'https://pod.example/alice/me',
          o1: 'https://external.org/external-node',
        });

        expect(filter.shouldFilter(aliceBinding)).toBe(true);
      });

      it('should evaluate Bob bindings according to the Linear rule', () => {
        const bobIncomplete = createMockBinding({
          _source: 'https://pod.example/bob/profile.ttl',
          s: 'https://pod.example/bob/me',
          o1: 'https://external.org/external-node',
        });
        expect(filter.shouldFilter(bobIncomplete)).toBe(false);

        const bobComplete = createMockBinding({
          _source: 'https://pod.example/bob/profile.ttl',
          s: 'https://pod.example/bob/me',
          o1: 'https://pod.example/bob/sub',
        });
        expect(filter.shouldFilter(bobComplete)).toBe(true);
      });
    });

    describe('for temporal burst and Map cache optimization', () => {
      it('should reuse rules via temporal pointer comparison for identical consecutive sources', () => {
        filter.registerCompositeResource('https://pod.example/alice/', ['s']);

        const spy = jest.spyOn(filter['uriTrie'], 'getAllMatchingRules');

        const binding1 = createMockBinding({
          _source: 'https://pod.example/alice/items.ttl',
          s: 'https://pod.example/alice/item/1',
        });
        const binding2 = createMockBinding({
          _source: 'https://pod.example/alice/items.ttl',
          s: 'https://pod.example/alice/item/2',
        });

        expect(filter.shouldFilter(binding1)).toBe(true);
        expect(filter.shouldFilter(binding2)).toBe(true);

        // First quad queries the trie; second quad hits the temporal pointer check
        expect(spy).toHaveBeenCalledTimes(1);
      });

      it('should serve rules from Map cache when source switches back after an interleave', () => {
        filter.registerCompositeResource('https://pod.example/alice/', ['s']);
        filter.registerCompositeResource('https://pod.example/bob/', ['s']);

        const spy = jest.spyOn(filter['uriTrie'], 'getAllMatchingRules');

        const aliceBinding = createMockBinding({
          _source: 'https://pod.example/alice/items.ttl',
          s: 'https://pod.example/alice/1',
        });
        const bobBinding = createMockBinding({
          _source: 'https://pod.example/bob/items.ttl',
          s: 'https://pod.example/bob/1',
        });

        filter.shouldFilter(aliceBinding); // Trie lookup 1 (Alice)
        filter.shouldFilter(bobBinding);   // Trie lookup 2 (Bob)
        filter.shouldFilter(aliceBinding); // Map cache hit (Alice)

        expect(spy).toHaveBeenCalledTimes(2);
      });
    });

    describe('for cache invalidation during rule registration', () => {
      it('should clear the source rule cache and reset burst pointer when a new rule is added', () => {
        filter.registerCompositeResource('https://pod.example/alice/', ['s']);

        const binding = createMockBinding({
          _source: 'https://pod.example/alice/items.ttl',
          s: 'https://pod.example/alice/item/1',
        });

        filter.shouldFilter(binding);
        expect(filter.sourceRulesCache.size).toBe(1);

        filter.registerCompositeResource('https://pod.example/bob/', ['s']);

        expect(filter.sourceRulesCache.size).toBe(0);
        expect(filter['lastSource']).toBeNull();
        expect(filter['lastRules']).toBeNull();
      });
    });
  });
});