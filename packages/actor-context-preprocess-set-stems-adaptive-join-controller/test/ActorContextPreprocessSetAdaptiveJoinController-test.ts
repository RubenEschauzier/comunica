import { ActionContext, Bus } from '@comunica/core';
import { KeysRdfJoin } from '@comunica/context-entries';
import { ActorContextPreprocessSetStemsAdaptiveJoinController } from '../lib/ActorContextPreprocessSetStemsAdaptiveJoinController';
import '@comunica/utils-jest';

describe('ActorContextPreprocessSetAdaptiveJoinController', () => {
  let bus: any;

  beforeEach(() => {
    bus = new Bus({ name: 'bus' });
  });

  describe('An ActorContextPreprocessSetAdaptiveJoinController instance', () => {
    let actor: ActorContextPreprocessSetStemsAdaptiveJoinController;

    beforeEach(() => {
      actor = new ActorContextPreprocessSetStemsAdaptiveJoinController({ name: 'actor', bus });
    });

    it('should test', () => {
      return expect(actor.test({ context: new ActionContext() })).resolves.toPassTestVoid();
    });

    it('should run', async () => {
      const output = await actor.run({ context: new ActionContext() });
      expect(output.context.has(KeysRdfJoin.adaptiveJoinController)).toBe(true);
    });
  });
});
