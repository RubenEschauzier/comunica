import type { AsyncIterator } from 'asynciterator';
import type * as RDF from '@rdfjs/types';
import type { Bindings, IJoinEntryWithMetadata } from '@comunica/types';
import type { Algebra } from '@comunica/utils-algebra';

/**
 * Represents an active connected component in ANY adaptive query execution engine
 * (SteMs, classic Eddies, STAIRS, plan migration, etc.).
 */
export interface IAdaptiveJoinComponent {
  readonly id: number | string;

  // The operations (e.g. triple patterns, BGPs) executed by this component
  readonly operations: Algebra.Operation[];

  // Flag indicating whether this component has finished executing or closed
  readonly ended: boolean;

  /**
   * Checks if this component contains all the operations answered by the dynamic source.
   */
  canCoverOperations: (operations: Algebra.Operation[]) => boolean;

  /**
   * Injects a dynamic composite source (e.g. derived resource, index, or materialized view)
   * into this component's active execution plan.
   *
   * @param operations The operations satisfied by this data source.
   * @param dataStream Stream of either Bindings or raw RDF.Quads.
   * @param metadata Optional metadata, cost coefficients, or routing ticket hints.
   * @returns True if the source was accepted and attached, false otherwise.
   */
  addCompositeSource: <T extends Bindings | RDF.Quad>(
    operations: Algebra.Operation[],
    dataStream: AsyncIterator<T>,
    metadata?: Record<string, any>,
  ) => boolean;
}

/**
 * Top-level manager for dynamic source usage across all active join components and subqueries.
 */
export interface IAdaptiveJoinController {
  /**
   * High-level entry point used that finds all active components 
   * containing the given operations and uses the source.
   *
   * @returns True if the source was attached to at least one component, false otherwise.
   */
  addCompositeSource: <T extends Bindings | RDF.Quad>(
    operations: Algebra.Operation[],
    dataStream: AsyncIterator<T>,
    metadata?: Record<string, any>,
  ) => boolean;

  /**
   * Registers an active connected join component with the controller.
   */

  registerComponent: (component: IAdaptiveJoinComponent) => void;

  /**
   * Unregisters a connected join component (e.g. when its stream ends).
   */
  unregisterComponent: (component: IAdaptiveJoinComponent) => void;

  /**
   * Returns all active (non-ended) components that contain all of the specified operations.
   * If a pattern appears in multiple components (e.g. in UNION branches or subqueries),
   * all matching components are returned.
   */

  getComponentsForOperations: (operations: Algebra.Operation[]) => IAdaptiveJoinComponent[];
  
  /**
   * Returns all currently registered active components.
   */
  getComponents: () => IAdaptiveJoinComponent[];
}


/**
 * TODO: 
 * Gotcha 1: "Zombie" (Ended) Components
Subqueries often finish before the outer query finishes. If a subquery finishes and closes its 

StemsControllerStream
, you must not inject new derived sources into it:

Fix: Either auto-unregister the component when its stream emits 'end', or filter out ended streams during lookup:
typescript
public getComponentsForOperations(operations: Algebra.Operation[]): A[] {
  return this.components.filter(component => 
    !component.stemsControllerStream.ended && 
    component.canCoverOperations(operations)
  );
}
 */

/**
 * TODO: 
 * Gotcha 2: Repeated Subquery Execution (Correlated / Bind Joins)
If a subquery is evaluated inside an outer loop (e.g. repeated execution for each incoming binding):

A new 

StemsControllerStream
 is instantiated on each iteration.
If components don't clean themselves up on 'end', they will accumulate in memory (memory leak) and receive duplicate events.
Fix: Hook into the stream lifecycle inside addAdaptiveJoinConnectedComponent:
typescript
stemsControllerStream.on('end', () => {
  this.unregisterComponent(component);
});
 */