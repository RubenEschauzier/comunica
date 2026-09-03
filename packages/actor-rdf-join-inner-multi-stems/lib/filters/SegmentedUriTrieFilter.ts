import { Bindings } from "@comunica/types";

export class SegmentedUriTrieFilter {
  protected readonly authorities: Map<string, TrieNode> = new Map();

  public constructor() {}

  public hasFilterMatching(uri: string): boolean {
    const { authority, path } = this.parseUri(uri);

    let currentNode = this.authorities.get(authority);
    if (!currentNode) {
      return false;
    }

    const len = path.length;
    // Checks for "/"
    let startIdx = path.charCodeAt(0) === 47 ? 1 : 0;

    // Handle root authority matching (e.g., filter registered at root "/")
    if (startIdx >= len) {
      return Boolean(currentNode.isTerminal);
    }

    while (startIdx < len) {
      if (currentNode.isTerminal) {
        return true;
      }

      let endIdx = path.indexOf('/', startIdx);
      if (endIdx === -1) {
        endIdx = len;
      }

      if (endIdx > startIdx) {
        const segment = path.substring(startIdx, endIdx);
        const child = currentNode.getChild(segment);
        if (!child) {
          return false;
        }
        currentNode = child;
      }

      startIdx = endIdx + 1;
    }

    return Boolean(currentNode.isTerminal);
  }

  public addStringFilter(filterUri: string): void {
    const { authority, path } = this.parseUri(filterUri);

    let currentNode = this.authorities.get(authority);
    if (!currentNode) {
      currentNode = new TrieNode(authority, (binding: Bindings) => true, false);
      this.authorities.set(authority, currentNode);
    }

    const len = path.length;
    let startIdx = path.charCodeAt(0) === 47 /* '/' */ ? 1 : 0;

    // Filter applies to the entire authority root (e.g. "http://example.org/")
    if (startIdx >= len) {
      currentNode.isTerminal = true;
      return;
    }

    while (startIdx < len) {
      let endIdx = path.indexOf('/', startIdx);
      const isLastSegment = endIdx === -1 || endIdx === len - 1;
      
      if (endIdx === -1) {
        endIdx = len;
      }

      if (endIdx > startIdx) {
        const segment = path.substring(startIdx, endIdx);
        let child: TrieNode | undefined = currentNode!.getChild(segment);

        if (!child) {
          child = new TrieNode(segment, (binding: Bindings) => true, isLastSegment);
          currentNode!.addChild(child);
        } else if (isLastSegment) {
          child.isTerminal = true;
        }

        currentNode = child;
      }

      startIdx = endIdx + 1;
    }
  }

  protected parseUri(uri: string): IParsedUri {
    const splitIndex = uri.indexOf("://");
    const scheme = splitIndex === -1 ? "" : uri.slice(0, splitIndex).toLowerCase();
    const rest = splitIndex === -1 ? uri : uri.slice(splitIndex + 3);

    const authorityPathSplit = rest.indexOf('/');
    if (authorityPathSplit === -1) {
      return {
        scheme,
        authority: this.normalizeAuthority(rest),
        path: "/",
      };
    }

    const authority = rest.slice(0, authorityPathSplit);
    let path = rest.slice(authorityPathSplit);

    const indexQuery = path.indexOf('?');
    if (indexQuery > -1) {
      path = path.slice(0, indexQuery);
    }
    const indexFragment = path.indexOf('#');
    if (indexFragment > -1) {
      path = path.slice(0, indexFragment);
    }

    return {
      scheme,
      authority: this.normalizeAuthority(authority),
      path,
    };
  }

  private normalizeAuthority(authority: string): string {
    const atIndex = authority.indexOf('@');
    return atIndex === -1
      ? authority.toLowerCase()
      : authority.slice(0, atIndex + 1) + authority.slice(atIndex + 1).toLowerCase();
  }
}

export class TrieNode {
  public readonly value: string;
  public readonly filterFn: any;
  public isTerminal: boolean;
  public readonly children: Map<string, TrieNode> = new Map();

  public constructor(value: string, filterFn: any, isTerminal = false) {
    this.value = value;
    this.filterFn = filterFn;
    this.isTerminal = isTerminal;
  }

  public addChild(child: TrieNode): void {
    this.children.set(child.value, child);
  }

  public getChild(value: string): TrieNode | undefined {
    return this.children.get(value);
  }
}

export interface IParsedUri {
  scheme: string;
  authority: string;
  path: string;
}