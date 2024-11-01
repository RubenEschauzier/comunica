import type {
  IActionMergeBindingsContext,
  IActorMergeBindingsContextOutput,
  IActorMergeBindingsContextArgs,
  IBindingsContextMergeHandler,
} from '@comunica/bus-merge-bindings-context';
import { ActorMergeBindingsContext } from '@comunica/bus-merge-bindings-context';
import type { IActorTest, TestResult } from '@comunica/core';
import { passTestVoid } from '@comunica/core';
import { PromiseUnionBindingsContextMergeHandler } from './PromiseUnionBindingsContextMergeHandler';

/**
 * A comunica Union Merge Bindings Context Actor.
 */
export class ActorMergeBindingsContextPromiseUnion extends ActorMergeBindingsContext {
  private readonly contextKey: string;
  public constructor(args: IActorMergeBindingsContextPromiseUnionArgs) {
    super(args);
    this.contextKey = args.contextKey;
  }

  public async test(_action: IActionMergeBindingsContext): Promise<TestResult<IActorTest>> {
    return passTestVoid();
  }

  public async run(_action: IActionMergeBindingsContext): Promise<IActorMergeBindingsContextOutput> {
    // Merge function: Union with set semantics
    const mergeHandlers: Record<string, IBindingsContextMergeHandler<any>> = {};
    mergeHandlers[this.contextKey] = new PromiseUnionBindingsContextMergeHandler();

    return { mergeHandlers };
  }
}

export interface IActorMergeBindingsContextPromiseUnionArgs extends IActorMergeBindingsContextArgs {
  /**
   * The context key name to merge over.
   */
  contextKey: string;
}
