import { 
  ActorExtractSources, 
  IActionExtractSources,
   IActorExtractSourcesOutput, 
   IActorExtractSourcesArgs 
  } from '@comunica/bus-extract-sources';
import { IAction, IActorTest, passTestVoid, TestResult } from '@comunica/core';
import { IQuerySource } from '@comunica/types';
import { KeysCaches } from '@comunica/context-entries';

/**
 * A comunica Operation Extract Sources Actor.
 */
export class ActorExtractSourcesOperation extends ActorExtractSources {
  public constructor(args: IActorExtractSourcesArgs) {
    super(args);
  }

  public async test(_action: IAction): Promise<TestResult<IActorTest>> {
    return passTestVoid();
  }


  public async run(action: IActionExtractSources): Promise<IActorExtractSourcesOutput> {
    const sources: IQuerySource[] = []
    const withinQueryCache = action.context.getSafe(KeysCaches.withinQueryStoreCache);
    for (const key of withinQueryCache.keys()){
      sources.push( (await withinQueryCache.get(key)!).source);
    }
    return { sources }
  }
}
