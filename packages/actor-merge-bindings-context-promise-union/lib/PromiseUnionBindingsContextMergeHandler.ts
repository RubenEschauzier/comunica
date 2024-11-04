import type { IBindingsContextMergeHandler } from '@comunica/bus-merge-bindings-context';
import { UnionIterator, AsyncIterator } from 'asynciterator';

export class PromiseUnionBindingsContextMergeHandler implements IBindingsContextMergeHandler<any> {
  public run(...inputStreams: AsyncIterator<any>[]): UnionIterator<any> {
    return new UnionIterator([...inputStreams])
  }
}
