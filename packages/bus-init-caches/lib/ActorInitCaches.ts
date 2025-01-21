import type { IAction, IActorArgs, IActorOutput, IActorTest, Mediate } from '@comunica/core';
import { Actor } from '@comunica/core';
import type { IActionContext } from '@comunica/types';

/**
 * A comunica actor for init-caches events.
 *
 * Actor types:
 * * Input:  IActionInitCaches: The cache context object to be filled.
 * * Test:   <none>
 * * Output: IActorInitCachesOutput: The cache context filled by init-caches actors
 *
 * @see IActionInitCaches
 * @see IActorInitCachesOutput
 */
export abstract class ActorInitCaches<TS = undefined> 
  extends Actor<IActionInitCaches, IActorTest, IActorInitCachesOutput, TS> {
  /**
  * @param args - @defaultNested {<default_bus> a <cc:components/Bus.jsonld#Bus>} bus
  */
  /**
   * @param args -
   *   \ @defaultNested {<default_bus> a <cc:components/Bus.jsonld#Bus>} bus
   *   \ @defaultNested {Init caches ailed} busFailMessage
   */
  public constructor(args: IActorInitCachesArgs<TS>) {
    super(args);
  }
}

export interface IActionInitCaches extends IAction {
  /**
   * The context object holding the caches. This context will not be changed after
   * engine initialization.
   */
  cacheContext: IActionContext;
}

export interface IActorInitCachesOutput extends IActorOutput {
  /**
   * The context object holding the caches. This context will not be changed after
   * engine initialization.
   */
  cacheContext: IActionContext;
}

export type IActorInitCachesArgs<TS = undefined>  = IActorArgs<
IActionInitCaches, IActorTest, IActorInitCachesOutput, TS>;

export type MediatorInitCaches = Mediate<
IActionInitCaches, IActorInitCachesOutput>;