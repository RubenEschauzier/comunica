import { ActionContext, Bus } from '@comunica/core';
import type { IActionContext } from '@comunica/types';
import { ActorMergeBindingsContextUnion  SetUnionBindingsContextMergeHandler } from '@comunica/actor-merge-bindings-context-union';
import '@comunica/utils-jest';

// This test does not function properly as it does not test the promise actor
describe('ActorMergeBindingFactoryContextUnion', () => {
  let bus: any;

  beforeEach(() => {
    bus = new Bus({ name: 'bus' });
  });

  describe('An ActorMergeBindingFactoryContextUnion instance', () => {
    let setUnionMergeHandler: SetUnionBindingsContextMergeHandler;
    let actor: ActorMergeBindingsContextUnion;
    let context: IActionContext;

    beforeEach(() => {
      setUnionMergeHandler = new SetUnionBindingsContextMergeHandler();
      actor = new ActorMergeBindingsContextUnion({ name: 'actor', bus, contextKey: 'sources' });
      context = new ActionContext();
    });

    it('should test', async() => {
      await expect(actor.test({ context })).resolves.toPassTestVoid();
    });

    it('should run', async() => {
      await expect(actor.run({ context })).resolves.toMatchObject(
        { mergeHandlers: { sources: new SetUnionBindingsContextMergeHandler() }},
      );
    });
    it('should be SetUnionConstructor', () => {
      expect(new (<any> SetUnionBindingsContextMergeHandler)())
        .toBeInstanceOf(SetUnionBindingsContextMergeHandler);
    });
    it('merge handler should return set union', () => {
      const inputSets = [[ '1', '2', '3' ], [ '1', '3', '5' ], [ '2', '1' ]];
      expect(setUnionMergeHandler.run(...inputSets)).toStrictEqual([ '1', '2', '3', '5' ]);
    });
  });
});
