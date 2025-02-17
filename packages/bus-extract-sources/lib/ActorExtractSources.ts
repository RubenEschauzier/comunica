import { Actor, IAction, IActorArgs, IActorOutput, IActorTest, Mediate } from '@comunica/core';
import { IQuerySource } from '@comunica/types';
import { Operation } from 'sparqlalgebrajs/lib/algebra';

/**
 * A comunica actor for extract-sources events.
 *
 * Actor types:
 * * Input:  IActionExtractSources: Action with all operations to extract sources from
 * * Test:   <none>
 * * Output: IActorExtractSourcesOutput: Extracted sources
 *
 * @see IActionExtractSources
 * @see IActorExtractSourcesOutput
 */
export abstract class ActorExtractSources<TS = undefined> 
  extends Actor<IActionExtractSources, IActorTest, IActorExtractSourcesOutput, TS> {
  /**
  * @param args - @defaultNested {<default_bus> a <cc:components/Bus.jsonld#Bus>} bus
  */
  public constructor(args: IActorExtractSourcesArgs<TS>) {
    super(args);
  }
}

export interface IActionExtractSources extends IAction {
  operations: Operation[];
}

export interface IActorExtractSourcesOutput extends IActorOutput {
  sources: IQuerySource[];
}

export type IActorExtractSourcesArgs<TS = undefined> = IActorArgs<
IActionExtractSources, IActorTest, IActorExtractSourcesOutput, TS>;

export type MediatorExtractSources = Mediate<
IActionExtractSources, IActorExtractSourcesOutput>;
