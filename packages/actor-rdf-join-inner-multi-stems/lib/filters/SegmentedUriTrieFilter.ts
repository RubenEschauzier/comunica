import { Bindings } from "@comunica/types";

export class SegmentedUriTrieFilter {
  protected readonly authorities: Map<string, TrieNode> = new Map();

  public constructor() {

  }

  public hasFilterMatching(uri: string){
    const {scheme, authority, path} = this.parseUri(uri);
    // TODO: Remove split and fix authority / and authoriy only URIs
    const splitPath = path.split("/");

    const rootNode = this.authorities.get(authority);
    if (!rootNode){
      return false;
    }

    let currentNode: TrieNode | undefined = rootNode;
    for (let i = 0; i < splitPath.length; i++){
      if (currentNode.isTerminal){
        return true;
      }
      currentNode = currentNode?.getChild(splitPath[i]);
      if (!currentNode){
        return false;
      }
    }
    if (!currentNode.isTerminal){
      return false;
    }
    
    return true;
  }

  public addStringFilter(filterUri: string){
    const {scheme, authority, path} = this.parseUri(filterUri);
    // TODO: Remove split
    const splitPath = path.split("/");

    let rootNode = this.authorities.get(authority)!;
    if (!rootNode){
      // Root of Trie is the authority
      const newTrieNode = new TrieNode(
        authority,
        (binding: Bindings) => true,
      )
      // For each path entry add a childNode to parentNode and
      // then switch parentNode so the next path is attached to child.
      this.addNodeSequence(newTrieNode, splitPath);

      this.authorities.set(authority, newTrieNode);
      return;
    }

    // Authority exists and we should update the trie
    for (let i = 0; i < splitPath.length; i++){
      const child = rootNode.getChild(splitPath[i]);
      if (child){
        rootNode = child;
        continue;
      }
      // This part of path doesn't exist so we add the rest of path as sequence
      // of nodes
      this.addNodeSequence(rootNode, splitPath.slice(i));
      return;
    }
    return;
  }

  protected addNodeSequence(parentNode: TrieNode, splitPathSequence: string[]){
    for (let i = 0; i < splitPathSequence.length; i++){
      const childNode = new TrieNode(
        splitPathSequence[i],
        (binding: Bindings) => true,
        i === (splitPathSequence.length - 1),
      );
      parentNode.addChild(childNode);
      parentNode = childNode;
    }
  }

  /**
   * Faster parser of URIs. Assumes source attribution and quad terms have already
   * been parsed before, allowing us to avoid doing the full WHATWG spec-compliant 
   * new URL()
   * @param uri 
   * @returns Split uri into scheme, authority, and path
   */
  protected parseUri(uri: string): IParsedUri{
    const splitIndex = uri.indexOf("://");

    // Scheme is case-insenstive
    const scheme: string = uri.slice(0,splitIndex).toLowerCase()
    const rest = uri.slice(splitIndex + 3);

    const authorityPathSplit = rest.indexOf('/');
    if (authorityPathSplit === -1){
      return {
        scheme,
        authority: rest,
        path: "/"
      }
    }

    const authority = rest.slice(0,authorityPathSplit);
    const atIndex = authority.indexOf('@');

    // Domain should be case insensitive
    const normalizedAuthority = atIndex === -1
      ? authority.toLowerCase()
      : authority.slice(0, atIndex + 1) + authority.slice(atIndex + 1).toLowerCase();

    let path: string = rest.slice(authorityPathSplit);

    const indexQuery = path.indexOf("?");
    if (indexQuery > -1){
      path = path.slice(0, indexQuery);
    }
    const indexFragment = path.indexOf("#");
    if (indexFragment > -1){
      path = path.slice(0, indexFragment);
    }

    return {
      scheme,
      authority: normalizedAuthority,
      path,
    }
  }
}

export class TrieNode {
  protected readonly value: string;
  protected readonly filterFn: any;
  public readonly isTerminal?: boolean;

  protected readonly children: Map<string, TrieNode> = new Map();

  // TODO add filterFn type
  public constructor(value: string, filterFn: any, isTerminal?: boolean){
    this.value = value;
    this.filterFn = filterFn;
    this.isTerminal = isTerminal;
  }

  public addChild(child: TrieNode){
    this.children.set(child.value, child);
  }

  public getChild(value: string): TrieNode | undefined{
    return this.children.get(value);
  }
}

export interface IParsedUri {
  scheme: string;
  authority: string;
  path: string;
}
