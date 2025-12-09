// import type { IAdaptivePlanStatistics } from '@comunica/actor-rdf-join-inner-multi-stems';
import type { ActorHttpInvalidateListenable } from '@comunica/bus-http-invalidate';
import type { IActionRdfJoin, IActorRdfJoinOutputInner } from '@comunica/bus-rdf-join';
import { KeysStatistics } from '@comunica/context-entries';
import type { Actor, IActionObserverArgs, IActorTest } from '@comunica/core';
import { ActionObserver } from '@comunica/core';
import { IAdaptivePlanStatistics } from '@comunica/types';

/**
 * Observes HTTP actions, and maintains a counter of the number of requests.
 */
export class ActionObserverAdaptiveJoin extends ActionObserver<IActionRdfJoin, IActorRdfJoinOutputInner> {
  public readonly httpInvalidator: ActorHttpInvalidateListenable;
  public readonly observedActors: string[];
  public joinStatsContainers: Record<number, IAdaptivePlanStatistics> | undefined;

  /* eslint-disable max-len */
  /**
   * @param args - @defaultNested {<npmd:@comunica/bus-rdf-join/^4.0.0/components/ActorRdfJoin.jsonld#ActorRdfJoin_default_bus>} bus
   */
  public constructor(args: IActionObserverAdaptiveJoinArgs) {
    super(args);
    this.bus.subscribeObserver(this);
    this.httpInvalidator.addInvalidateListener(() => {
      this.joinStatsContainers = undefined;
    });
  }
  /* eslint-enable max-len */

  public onRun(
    actor: Actor<IActionRdfJoin, IActorTest, IActorRdfJoinOutputInner, undefined>,
    action: IActionRdfJoin,
    _output: Promise<IActorRdfJoinOutputInner>,
  ): void {
    if (this.observedActors.includes(actor.name)) {
      this.joinStatsContainers = action.context.get(KeysStatistics.adaptiveJoinStatistics);
    }
  }
}

export interface IActionObserverAdaptiveJoinArgs extends IActionObserverArgs<IActionRdfJoin, IActorRdfJoinOutputInner> {
  /* eslint-disable max-len */
  /**
   * An actor that listens to HTTP invalidation events
   * @default {<default_invalidator> a <npmd:@comunica/bus-http-invalidate/^4.0.0/components/ActorHttpInvalidateListenable.jsonld#ActorHttpInvalidateListenable>}
   */
  httpInvalidator: ActorHttpInvalidateListenable;
  /* eslint-enable max-len */
  /**
   * The URIs of the observed actors.
   * @default {urn:comunica:default:rdf-join/actors#inner-multi-stems}
   */
  observedActors: string[];
}
