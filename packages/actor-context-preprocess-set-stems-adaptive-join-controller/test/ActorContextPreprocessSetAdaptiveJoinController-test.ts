import { Bus } from '@comunica/core';
import { ActorContextPreprocessSetAdaptiveJoinController } from '../lib/ActorContextPreprocessSetAdaptiveJoinController';
import '@comunica/utils-jest';

describe('ActorContextPreprocessSetAdaptiveJoinController', () => {
  let bus: any;

  beforeEach(() => {
    bus = new Bus({ name: 'bus' });
  });

  describe('An ActorContextPreprocessSetAdaptiveJoinController instance', () => {
    let actor: ActorContextPreprocessSetAdaptiveJoinController;

    beforeEach(() => {
      actor = new ActorContextPreprocessSetAdaptiveJoinController({ name: 'actor', bus });
    });

    it('should test', () => {
      return expect(actor.test({ todo: true })).resolves.toPassTestVoid(); // TODO
    });

    it('should run', () => {
      return expect(actor.run({ todo: true })).resolves.toMatchObject({ todo: true }); // TODO
    });
  });
});
