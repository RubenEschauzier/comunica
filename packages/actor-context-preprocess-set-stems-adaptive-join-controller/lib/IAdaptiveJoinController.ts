import type { AsyncIterator } from 'asynciterator';
import type * as RDF from '@rdfjs/types';
import type { Bindings } from '@comunica/types';
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
  addCompositeSource: (
    operations: Algebra.Operation[],
    dataStream: AsyncIterator<Bindings>,
    metadata?: Record<string, any>,
  ) => boolean;

  /**
   * Signals that no further composite sources will be added to this component.
   * Closes any open dynamic source arrays (e.g. pushes null to AsyncReiterableArray).
   */
  finalize: () => void;
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

  /**
   * Signals that query source discovery is complete and no more composite sources
   * will be pushed into active components.
   */
  finalize: () => void;
}
