import { QuerySourceSkolemized } from '@comunica/actor-context-preprocess-query-source-skolemize';
import { 
  ActorExtractSources, 
  IActionExtractSources,
   IActorExtractSourcesOutput, 
   IActorExtractSourcesArgs 
  } from '@comunica/bus-extract-sources';
import { IAction, IActorTest, passTestVoid, TestResult } from '@comunica/core';
import { IQuerySource, IQuerySourceWrapper } from '@comunica/types';
import { QuerySourceHypermedia } from '@comunica/actor-query-source-identify-hypermedia';
import { getSources } from '@comunica/utils-query-operation';

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
    // const sourcesOperations = action.operations.map(x => getSources(x));
    // const sources: Record<string, IQuerySource> = {};
    // sourcesOperations.map(x => x.map((sourceWrapper: IQuerySourceWrapper) => {
    //   const sourceString = sourceWrapper.source.toString();
    //   if (!sources[sourceString]) {
    //     sources[sourceString] = sourceWrapper.source;
    //   }
    // }));
    
    // // Currently this only takes one source (a QuerySourceHypermedia) with possibly more source states
    // // (Will have to be more general / better in future)
    // const sourceToSample = <QuerySourceHypermedia><unknown>
    // (<QuerySourceSkolemized><unknown>sources[Object.keys(sources)[0]]).innerSource;
    // const sourcesInner = await Promise.all([...sourceToSample.sourcesState.keys()].map(async key => {
    //   return (await sourceToSample.sourcesState.get(key)!).source
    // }));

    return {sources: []};
  }
}
