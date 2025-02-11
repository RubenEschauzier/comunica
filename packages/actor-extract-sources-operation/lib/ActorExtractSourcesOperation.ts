import { QuerySourceSkolemized } from '@comunica/actor-context-preprocess-query-source-skolemize';
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
 * A comunica Operation Extract Sources Actor. Takes all ISourceState's from the
 * withQuerySourceCache and extracts their inner sources.
 */
export class ActorExtractSourcesOperation extends ActorExtractSources {
  public constructor(args: IActorExtractSourcesArgs) {
    super(args);
  }

  public async test(_action: IAction): Promise<TestResult<IActorTest>> {
    return passTestVoid();
  }


  public async run(action: IActionExtractSources): Promise<IActorExtractSourcesOutput> {
    const sources: IQuerySource[] = [];
    const sourceCache = action.context.getSafe(KeysCaches.withinQueryStoreCache);
    const resolvedSources = await Promise.all([...sourceCache.values()]);
    sources.push(...resolvedSources.map(source => source.source));
    return { sources };
  }
}
