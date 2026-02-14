import { PersistentCacheManager } from '@comunica/actor-context-preprocess-set-persistent-cache-manager';
import { IActionContextPreprocess, IActorContextPreprocessOutput } from '@comunica/bus-context-preprocess';
import type { ActorHttpInvalidateListenable } from '@comunica/bus-http-invalidate';
import { KeysCaching } from '@comunica/context-entries';
import type { Actor, IActionObserverArgs, IActorTest } from '@comunica/core';
import { ActionObserver } from '@comunica/core';

export class ActionObserverContextPreprocess extends ActionObserver<IActionContextPreprocess, IActorContextPreprocessOutput> {
  public readonly httpInvalidator: ActorHttpInvalidateListenable;
  public readonly observedActors: string[];
  
  private cacheManagerInstance: PersistentCacheManager | undefined;

  public constructor(args: IActionObserverContextPreprocessArgs) {
    super(args);
    this.httpInvalidator = args.httpInvalidator;
    this.observedActors = args.observedActors;
    this.bus.subscribeObserver(this);
    
    this.httpInvalidator.addInvalidateListener(() => {
      this.cacheManagerInstance = undefined;
    });
  }

  public onRun(
    actor: Actor<IActionContextPreprocess, IActorTest, IActorContextPreprocessOutput, undefined>,
    _action: IActionContextPreprocess,
    output: Promise<IActorContextPreprocessOutput>,
  ): void {
    if (this.observedActors.includes(actor.name)) {
      output.then((processOutput) => {
        const cacheManager = processOutput.context.get(KeysCaching.cacheManager);
        if (cacheManager) {
          this.cacheManagerInstance = cacheManager;
        }
      }).catch(() => {
        // Prevent unhandled promise rejections if the context preprocess fails upstream
      });
    }
  }

  public getManager(): PersistentCacheManager | undefined {
    return this.cacheManagerInstance;
  }

  public getCacheMetrics(cacheManager: PersistentCacheManager) {
    return cacheManager.endAllSessions();
  }
}

export interface IActionObserverContextPreprocessArgs extends IActionObserverArgs<IActionContextPreprocess, IActorContextPreprocessOutput> {
  httpInvalidator: ActorHttpInvalidateListenable;
  observedActors: string[];
}