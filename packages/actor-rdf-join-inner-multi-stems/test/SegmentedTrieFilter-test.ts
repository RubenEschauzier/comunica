import { SegmentedUriTrieFilter, type IParsedUri } from '../lib/filters/SegmentedUriTrieFilter';

class TestableSegmentedUriTrieFilter extends SegmentedUriTrieFilter {
  public override parseUri(uri: string): IParsedUri {
    return super.parseUri(uri) as IParsedUri;
  }
}

describe('SegmentedUriTrieFilter', () => {
  let filter: TestableSegmentedUriTrieFilter;

  beforeEach(() => {
    filter = new TestableSegmentedUriTrieFilter();
  });

  describe('An parseUri invocation', () => {
    describe('for standard extraction', () => {
      it('should split a standard HTTP/HTTPS URI into scheme, authority, and path', () => {
        const result = filter.parseUri('https://example.com/api/v1/users');
        expect(result).toEqual({
          scheme: 'https',
          authority: 'example.com',
          path: '/api/v1/users',
        });
      });

      it('should default path to "/" when no path is provided', () => {
        const result = filter.parseUri('https://example.com');
        expect(result).toEqual({
          scheme: 'https',
          authority: 'example.com',
          path: '/',
        });
      });

      it('should handle custom protocol schemes', () => {
        const result = filter.parseUri('grpc://internal.service/rpc.Endpoint');
        expect(result).toEqual({
          scheme: 'grpc',
          authority: 'internal.service',
          path: '/rpc.Endpoint',
        });
      });
    });

    describe('for RFC 3986 authority handling and segment stripping', () => {
      it('should strip query parameters from the path', () => {
        const result = filter.parseUri('https://example.com/search?query=jest&page=1');
        expect(result).toEqual({
          scheme: 'https',
          authority: 'example.com',
          path: '/search',
        });
      });

      it('should strip URL fragment hashes from the path', () => {
        const result = filter.parseUri('https://example.com/docs#overview');
        expect(result).toEqual({
          scheme: 'https',
          authority: 'example.com',
          path: '/docs',
        });
      });

      it('should preserve non-default ports in authority', () => {
        const result = filter.parseUri('http://localhost:8080/dashboard');
        expect(result).toEqual({
          scheme: 'http',
          authority: 'localhost:8080',
          path: '/dashboard',
        });
      });

      it('should preserve default ports verbatim under RFC 3986 syntax rules', () => {
        const result = filter.parseUri('https://cdn.example.org:443/assets/img/');
        expect(result).toEqual({
          scheme: 'https',
          authority: 'cdn.example.org:443',
          path: '/assets/img/',
        });
      });

      it('should preserve userinfo in the authority', () => {
        const result = filter.parseUri('https://admin:secret123@secure.internal/admin');
        expect(result).toEqual({
          scheme: 'https',
          authority: 'admin:secret123@secure.internal',
          path: '/admin',
        });
      });

      it('should preserve full authority with userinfo and port while stripping query and fragment', () => {
        const result = filter.parseUri(
          'https://user:pass@api.test.io:8443/v2/items?filter=active&limit=10#section',
        );
        expect(result).toEqual({
          scheme: 'https',
          authority: 'user:pass@api.test.io:8443',
          path: '/v2/items',
        });
      });

      it('should preserve authority consisting of an IPv6 host and port', () => {
        const result = filter.parseUri('http://[::1]:9000/metrics');
        expect(result).toEqual({
          scheme: 'http',
          authority: '[::1]:9000',
          path: '/metrics',
        });
      });
    });

    describe('for casing and edge cases', () => {
      it('should normalize scheme and host to lowercase while preserving userinfo and path case', () => {
        const result = filter.parseUri('HTTPS://User:Secret@MyDomain.ORG:8080/UserData/Profile');
        expect(result).toEqual({
          scheme: 'https',
          authority: 'User:Secret@mydomain.org:8080',
          path: '/UserData/Profile',
        });
      });

      it('should preserve trailing slashes in the path', () => {
        const result = filter.parseUri('https://example.com/static/');
        expect(result.path).toBe('/static/');
      });
    });
  });

  describe('An hasFilter invocation', () => {
    it('should return false for any URI when the trie is empty', () => {
      expect(filter.hasFilterMatching('https://example.com/api')).toBe(false);
      expect(filter.hasFilterMatching('http://localhost:8080/')).toBe(false);
    });

    describe('for exact and disjoint URI matches', () => {
      it('should return true for an exact URI registered in the trie', () => {
        filter.addStringFilter('https://example.com/api/v1/users');

        expect(filter.hasFilterMatching('https://example.com/api/v1/users')).toBe(true);
      });

      it('should return false for URIs matching on path but belonging to a different authority', () => {
        filter.addStringFilter('https://example.com/api/v1/users');

        expect(filter.hasFilterMatching('https://other.org/api/v1/users')).toBe(false);
      });

      it('should return false for parent paths of an inserted specific filter', () => {
        filter.addStringFilter('https://example.com/api/v1/users');

        expect(filter.hasFilterMatching('https://example.com/api/v1')).toBe(false);
        expect(filter.hasFilterMatching('https://example.com/api')).toBe(false);
        expect(filter.hasFilterMatching('https://example.com/')).toBe(false);
      });

      it('should match root-level path filters', () => {
        filter.addStringFilter('https://example.com/');

        expect(filter.hasFilterMatching('https://example.com/')).toBe(true);
        expect(filter.hasFilterMatching('https://example.com')).toBe(true);
      });
    });

    describe('for hierarchical and prefix paths', () => {
      it('should return true for deeper sub-paths when a prefix filter is registered', () => {
        filter.addStringFilter('https://example.com/api');

        expect(filter.hasFilterMatching('https://example.com/api')).toBe(true);
        expect(filter.hasFilterMatching('https://example.com/api/v1')).toBe(true);
        expect(filter.hasFilterMatching('https://example.com/api/v1/users/123')).toBe(true);
      });

      it('should not match partial string prefixes across path boundaries', () => {
        filter.addStringFilter('https://example.com/api');

        expect(filter.hasFilterMatching('https://example.com/apigateway')).toBe(false);
      });

      it('should distinguish multiple branching paths on the same authority', () => {
        filter.addStringFilter('https://example.com/api/v1');
        filter.addStringFilter('https://example.com/api/v1/specialized');
        filter.addStringFilter('https://example.com/static/images');

        expect(filter.hasFilterMatching('https://example.com/api/v1')).toBe(true);
        expect(filter.hasFilterMatching('https://example.com/api/v1/data')).toBe(true);
        expect(filter.hasFilterMatching('https://example.com/static/images')).toBe(true);
        expect(filter.hasFilterMatching('https://example.com/static/images/logo.png')).toBe(true);

        expect(filter.hasFilterMatching('https://example.com/api/v2')).toBe(false);
        expect(filter.hasFilterMatching('https://example.com/static/css')).toBe(false);
      });
    });

    describe('for authority and port differentiation', () => {
      it('should treat authorities with different ports as distinct entries', () => {
        filter.addStringFilter('http://localhost:8080/metrics');

        expect(filter.hasFilterMatching('http://localhost:8080/metrics')).toBe(true);
        expect(filter.hasFilterMatching('http://localhost:3000/metrics')).toBe(false);
        expect(filter.hasFilterMatching('http://localhost/metrics')).toBe(false);
      });

      it('should normalize authority casing when matching filters', () => {
        filter.addStringFilter('https://API.Example.COM/v1/resource');

        expect(filter.hasFilterMatching('https://api.example.com/v1/resource')).toBe(true);
        expect(filter.hasFilterMatching('https://Api.Example.Com/v1/resource')).toBe(true);
      });
    });

    describe('for query parameter and fragment handling', () => {
      it('should ignore query strings and fragments when checking hasFilter', () => {
        filter.addStringFilter('https://example.com/search');

        expect(filter.hasFilterMatching('https://example.com/search?query=jest&page=1')).toBe(true);
        expect(filter.hasFilterMatching('https://example.com/search#results')).toBe(true);
        expect(filter.hasFilterMatching('https://example.com/search?query=jest#results')).toBe(true);
      });

      it('should strip query strings and fragments during filter registration', () => {
        filter.addStringFilter('https://example.com/catalog?category=all#top');

        expect(filter.hasFilterMatching('https://example.com/catalog')).toBe(true);
        expect(filter.hasFilterMatching('https://example.com/catalog/item-42')).toBe(true);
      });
    });

    describe('for trailing slashes', () => {
      it('should match paths with or without trailing slash consistently', () => {
        filter.addStringFilter('https://example.com/data');

        expect(filter.hasFilterMatching('https://example.com/data/')).toBe(true);
        expect(filter.hasFilterMatching('https://example.com/data')).toBe(true);
      });

      it('should match lookups without a trailing slash when added with one', () => {
        filter.addStringFilter('https://example.com/data/');

        expect(filter.hasFilterMatching('https://example.com/data')).toBe(true);
        expect(filter.hasFilterMatching('https://example.com/data/')).toBe(true);
      });
    });
  });
});