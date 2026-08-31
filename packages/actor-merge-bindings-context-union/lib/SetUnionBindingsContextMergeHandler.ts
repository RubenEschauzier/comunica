import type { IBindingsContextMergeHandler } from '@comunica/bus-merge-bindings-context';

export class SetUnionBindingsContextMergeHandler implements IBindingsContextMergeHandler<any> {
  public run(...inputSets: any[][]): any[] {
    if (inputSets.length === 1) {
      return inputSets[0];
    }
    if (inputSets.length === 2 && inputSets[0] === inputSets[1]) {
      return inputSets[0];
    }
    return [ ...new Set<string>(inputSets.flat()) ];
  }
}
