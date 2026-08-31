import { IAdaptiveJoinComponent, IAdaptiveJoinController } from "./IAdaptiveJoinController";
import type { Algebra } from '@comunica/utils-algebra';
import type * as RDF from '@rdfjs/types';
import { AsyncIterator } from 'asynciterator';

export class AdaptiveJoinController implements IAdaptiveJoinController {
  protected readonly components: Set<IAdaptiveJoinComponent> = new Set();
  protected finalized = false;

  public finalize(): void {
    this.finalized = true;
    for (const component of this.components) {
      component.finalize();
    }
  }

  public registerComponent(component: IAdaptiveJoinComponent): void {
    this.components.add(component);
    if (this.finalized) {
      component.finalize();
    }
  }

  public unregisterComponent(component: IAdaptiveJoinComponent): void {
    this.components.delete(component);
  }

  public getComponents(): IAdaptiveJoinComponent[] {
    return Array.from(this.components).filter(c => !c.ended);
  }

  public getComponentsForOperations(operations: Algebra.Operation[]): IAdaptiveJoinComponent[] {
    return this.getComponents().filter(component => component.canCoverOperations(operations));
  }

  public addCompositeSource<T extends RDF.Bindings | RDF.Quad>(
    operations: Algebra.Operation[],
    dataStream: AsyncIterator<T>,
    metadata?: Record<string, any>,
  ): boolean {
    const matchingComponents = this.getComponentsForOperations(operations);
    if (matchingComponents.length === 0) {
      return false;
    }

    // Fast path: single matching component
    if (matchingComponents.length === 1) {
      return matchingComponents[0].addCompositeSource(operations, <AsyncIterator<RDF.Bindings>> <any> dataStream, metadata);
    }

    // If multiple components contain these patterns (e.g. across UNION branches),
    // clone the stream so each component gets its own readable copy.
    let anySucceeded = false;
    for (const component of matchingComponents) {
      if (component.addCompositeSource(operations, <AsyncIterator<RDF.Bindings>> <any> dataStream.clone(), metadata)) {
        anySucceeded = true;
      }
    }
    return anySucceeded;
  }
}
