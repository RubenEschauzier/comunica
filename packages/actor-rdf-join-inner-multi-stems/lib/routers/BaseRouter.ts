import { Algebra } from '@comunica/utils-algebra';
import type { Bindings } from '@comunica/utils-bindings-factory';

import type * as RDF from '@rdfjs/types';
import { stemsContextKeys } from '../StemsControllerStream';
import type { StemsOperatorStream } from '../StemsOperatorStream';
import equal = require('deep-equal');

export interface IRouteTableOperation {
  // Operations satisfied by this entry
  operations: Algebra.Operation[];
  // Bit mask representing the operations satisfied by this entry
  doneBitMask: number;
  // Variables in operation
  variables: RDF.Variable[];
  // Named nodes in operation
  namedNodes: RDF.NamedNode[];
}

export abstract class RouterBase implements IStemsRouter {
  protected routeOperations: IRouteTableOperation[] = [];

  public createRouteTable(
    operations: IRouteTableOperation[],
  ): Record<number, IStemsRoutingEntry[][]> {
    if (operations.length > 30) {
      throw new Error(`RouterBase supports up to 30 operations (received ${operations.length}).`);
    }
    // Guard against router reuse with different route operations
    if (this.routeOperations.length > 0) {
      if (
        this.routeOperations.length !== operations.length ||
        this.routeOperations.some((op, i) => !equal(op.operations, operations[i].operations))
      ) {
        throw new Error(
          'Router state error: createRouteTable was called with routeOperations that differ from already set routeOperations.',
        );
      }
    }

    this.routeOperations = operations;

    // Calculate total composite query completion mask across all operations
    const allBitsMask = operations.reduce((acc, op) => acc | op.doneBitMask, 0);
    // Find the number of bits needed to represent all states
    const nBits = allBitsMask === 0 ? 0 : 32 - Math.clz32(allBitsMask);
    const totalStates = 1 << nBits;

    const routeTable: Record<number, IStemsRoutingEntry[][]> = {};

    for (let state = 1; state < totalStates; state++) {
      // If all operations in the query are completed, no next entry
      if ((state & allBitsMask) === allBitsMask) {
        continue;
      }

      // Collect variables and namedNodes from all operations satisfied in this state
      const doneOperations = this.routeOperations.filter(
        op => (state & op.doneBitMask) === op.doneBitMask,
      );

      const doneVars = new Set(
        doneOperations.flatMap(op => op.variables.map(v => v.value)),
      );
      
      const doneNamedNodes = new Set(
        doneOperations.flatMap(op => op.namedNodes.map(n => n.value)),
      );

      const possibleNext: IStemsRoutingEntry[] = [];

      for (let nextIdx = 0; nextIdx < this.routeOperations.length; nextIdx++) {
        const nextEntry = this.routeOperations[nextIdx];

        // An operator can only be routed to if none of its operations have already been completed
        if ((state & nextEntry.doneBitMask) === 0) {
          this.addOperatorIfOverlapping(
            possibleNext,
            nextIdx,
            nextEntry,
            doneVars,
            doneNamedNodes,
          );
        }
      }

      routeTable[state] = [ possibleNext ];
    }
    return routeTable;
  }

  protected getSetBitIndexes(mask: number): number[] {
    const indexes: number[] = [];
    let position = 0;
    while (mask !== 0) {
      if ((mask & 1) === 1) {
        indexes.push(position);
      }
      mask >>>= 1;
      position++;
    }
    return indexes;
  }

  protected getUnSetBitIndexes(mask: number, n: number): number[] {
    const indexes: number[] = [];
    for (let i = 0; i < n; i++) {
      if ((mask & (1 << i)) === 0) {
        indexes.push(i);
      }
    }
    return indexes;
  }

  protected doneIndexesToMask(indexes: number[]){
    // Sum the indexes by their respective bit representation
    return indexes.reduce((acc: number, curr: number) => acc + (1 << curr), 0);
  }

  public routeBinding(binding: Bindings, n: number): number | undefined {
    const done = binding.getContextEntry(stemsContextKeys.eddiesMetadata)!.done;
    if (done === (1 << n) - 1) {
      return undefined;
    }
    const indexesReadyEddies = this.getUnSetBitIndexes(done, n);

    return Math.min(...indexesReadyEddies);
  }

  public addOperator(
    routeTable: Record<number, IStemsRoutingEntry[][]>,
    stemsOperatorStream: StemsOperatorStream,
    metadata?: Record<string, any>,
  ): Record<number, IStemsRoutingEntry[][]> {
    // TODO: This does not work currently, exclusivity routing is not enforced properly
    // we must construct an alternative routing strategy.
    // Possibly add one routing, but whenever a CR gets picked also route to an alternative non-cr
    // TODO, do we need to update existing routing?
    /**
     * 
     */
    const setBitsMask = stemsOperatorStream.doneBitMask; 

    for (const [ doneKey, routing ] of Object.entries(routeTable)) {
      const key = Number.parseInt(doneKey, 10);
      // If the done signature has no overlap with the current entry we can route to this derived resource
      if ((key & setBitsMask) === 0){
        // Find all done operations and their variables and namedNodes
        const doneOperations = this.routeOperations.filter(
          op => (key & op.doneBitMask) === op.doneBitMask,
        );
        const doneVars = new Set(
          doneOperations.flatMap(op => op.variables.map(v => v.value)),
        );
        const doneNamedNodes = new Set(
          doneOperations.flatMap(op => op.namedNodes.map(n => n.value)),
        );

        const newRouting: IStemsRoutingEntry[] = [];
        // Always add the composite resource to the routing entry
        const added = this.addOperatorIfOverlapping(
          newRouting,
          stemsOperatorStream.operatorIndex,
          stemsOperatorStream,
          doneVars,
          doneNamedNodes,
        );
        if (!added){
          continue;
        }
        
        // Iterate over all other operator streams to see which ones are
        // also options
        for (let nextIdx = 0; nextIdx < this.routeOperations.length; nextIdx++) {
          const nextEntry = this.routeOperations[nextIdx];

          // The possible iterators in a composite resource path are those that
          // are not yet completed and have no overlap with the composite resource
          if ((key & nextEntry.doneBitMask) === 0 && (setBitsMask & nextEntry.doneBitMask) === 0) {
            this.addOperatorIfOverlapping(
              newRouting,
              nextIdx,
              nextEntry,
              doneVars,
              doneNamedNodes,
            );
          }
        }
        routing.push(newRouting);
      }
    }
    this.routeOperations.push({
      operations: stemsOperatorStream.operations,
      doneBitMask: stemsOperatorStream.doneBitMask,
      variables: stemsOperatorStream.variables,
      namedNodes: stemsOperatorStream.namedNodes,
    });


    return routeTable;
  }

  protected getDerivedResourceBits(derivedOperations: Algebra.Operation[]){
    const indexes = this.routeOperations.flatMap((routeOperation, idx) =>
      derivedOperations.some(derivedOperation =>
        routeOperation.operations.some(op => equal(derivedOperation, op))
      ) ? [idx] : []
    );
    if (indexes.length === 0){
      throw new Error("Tried to add derived resource with no overlap with current routingTable");
    }
    return this.doneIndexesToMask(indexes);
  }

  protected addOperatorIfOverlapping(
    possibleNext: IStemsRoutingEntry[],
    nextIdx: number,
    nextEntry: IRouteTableOperation,
    doneVars: Set<string>,
    doneNamedNodes: Set<string>,
  ): boolean {
    // Check for overlapping variables between triple patterns
    const joinVars = nextEntry.variables.filter(v => doneVars.has(v.value));
    // When variables overlap we add it to possible next routing decision
    if (joinVars.length > 0) {
      possibleNext.push({
        next: nextIdx,
        operations: nextEntry.operations,
        joinVars,
      });
      return true;
    }

    // Same approach for IRIs, if an IRI matches between triple patterns it is a valid routing decision
    const joinNamedNodes = nextEntry.namedNodes.filter(n => doneNamedNodes.has(n.value));
    if (joinNamedNodes.length > 0) {
      // JoinVars is empty for joins with no overlapping variables
      possibleNext.push({
        next: nextIdx,
        operations: nextEntry.operations,
        joinVars: [],
      });
      return true;
    }

    return false;
  }

  public abstract updateRouteTable(
    operators: StemsOperatorStream[],
    routeTable: Record<string, IStemsRoutingEntry[][]>,
  ): Record<number, IStemsRoutingEntry[][]>;
}

/**
 * Stem routing interface using routing table. Routing tables allow
 * multiple exclusive next routes. These exclusive routes are in the outer array of 
 * IStemsRoutingEntry[][]. These exclusive routes are concurrently followed and should
 * naturally be exclusive through the set 'done' bits. 
 */
export interface IStemsRouter {
  addOperator(
    routeTable: Record<number, IStemsRoutingEntry[][]>,
    stemsOperatorStream: StemsOperatorStream,
    metadata?: Record<string, any>,
  ): Record<number, IStemsRoutingEntry[][]>;
  routeBinding: (binding: Bindings, n: number) => number | undefined;
  createRouteTable: (
    operations: IRouteTableOperation[],
  ) => Record<number, IStemsRoutingEntry[][]>;
  updateRouteTable: (
    operators: StemsOperatorStream[], 
    routeTable: Record<string, IStemsRoutingEntry[][]>,
  ) => Record<number, IStemsRoutingEntry[][]>;
}

export interface IStemsRoutingEntry {
  next: number;
  operations: Algebra.Operation[];
  joinVars: RDF.Variable[];
}

export interface IStemsRouterFactory {
  /**
   * Creates a new router instance for a specific query execution.
   */
  createRouter: () => IStemsRouter;
}
