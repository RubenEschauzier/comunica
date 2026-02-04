import { PersistentCacheManager } from '@comunica/actor-context-preprocess-set-persistent-cache-manager';
import { IActionContextPreprocess, IActorContextPreprocessOutput } from '@comunica/bus-context-preprocess';
import type { IActionHttp, IActorHttpOutput } from '@comunica/bus-http';
import type { ActorHttpInvalidateListenable } from '@comunica/bus-http-invalidate';
import { KeysCaching } from '@comunica/context-entries';
import type { Actor, IActionObserverArgs, IActorTest } from '@comunica/core';
import { ActionObserver } from '@comunica/core';

/**
 * Observes HTTP actions, and maintains a counter of the number of requests.
 */
export class ActionObserverContextPreprocess extends ActionObserver<IActionContextPreprocess, IActorContextPreprocessOutput> {
  public readonly httpInvalidator: ActorHttpInvalidateListenable;
  public readonly observedActors: string[];
  // public cacheManager: Promise<PersistentCacheManager> | undefined;
  private resolveCacheManager!: (value: PersistentCacheManager) => void; 
  public cacheManager: Promise<PersistentCacheManager> | undefined = new Promise((resolve) => {
    this.resolveCacheManager = resolve;
  });
  /* eslint-disable max-len */
  /**
   * @param args - @defaultNested {<npmd:@comunica/bus-context-preprocess/^5.0.0/components/ActorContextPreprocess.jsonld#ActorContextPreprocess_default_bus>} bus
   */
  public constructor(args: IActionObserverContextPreprocessArgs) {
    super(args);
    this.httpInvalidator = args.httpInvalidator;
    this.observedActors = args.observedActors;
    this.bus.subscribeObserver(this);
    this.httpInvalidator.addInvalidateListener(() => {
      this.cacheManager = new Promise((resolve) => {
        this.resolveCacheManager = resolve;
      });
    });
  }
  /* eslint-enable max-len */

  public onRun(
    actor: Actor<IActionContextPreprocess, IActorTest, IActorContextPreprocessOutput, undefined>,
    _action: IActionContextPreprocess,
    output: Promise<IActorContextPreprocessOutput>,
  ): void {
    if (this.observedActors.includes(actor.name)) {
      output.then((processOutput)=> {
        const cacheManager = processOutput.context.get(KeysCaching.cacheManager);
        if (cacheManager){
          this.resolveCacheManager(cacheManager)
        } 
      }
    )
    }
  }

  public async getCacheMetrics(){
    const cacheManager = await this.cacheManager;
    if (!cacheManager){
      console.log("Cachemanager doesn't exist!!");
      return;
    }
    const cacheMetrics = cacheManager.endAllSessions();
    return cacheMetrics;
  }
}

export interface IActionObserverContextPreprocessArgs extends IActionObserverArgs<IActionContextPreprocess, IActorContextPreprocessOutput> {
  /* eslint-enable max-len */
  /**
   * An actor that listens to HTTP invalidation events
   * @default {<default_invalidator> a <npmd:@comunica/bus-http-invalidate/^5.0.0/components/ActorHttpInvalidateListenable.jsonld#ActorHttpInvalidateListenable>}
   */
  httpInvalidator: ActorHttpInvalidateListenable;
  /**
   * The URIs of the observed actors.
   * @default {urn:comunica:default:http/actors#fetch}
   */
  observedActors: string[];
}
