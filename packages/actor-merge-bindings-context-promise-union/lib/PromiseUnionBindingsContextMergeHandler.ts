import type { IBindingsContextMergeHandler } from '@comunica/bus-merge-bindings-context';

export class PromiseUnionBindingsContextMergeHandler implements IBindingsContextMergeHandler<any> {
  public run(...inputSets: Promise<any>[][]): Promise<any>[] {
    return [ ...inputSets.flat() ];
  }
}
