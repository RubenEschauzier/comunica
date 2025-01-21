import { Bus } from '@comunica/core';
import { ActorContextPreprocessSetSourceCache } from '../lib/ActorContextPreprocessSetSourceCache';

describe('ActorContextPreprocessSetSourceCache', () => {
  let bus: any;

  beforeEach(() => {
    bus = new Bus({ name: 'bus' });
  });

  describe('An ActorContextPreprocessSetSourceCache instance', () => {
    let actor: ActorContextPreprocessSetSourceCache;

    beforeEach(() => {
      actor = new ActorContextPreprocessSetSourceCache({ name: 'actor', bus });
    });

    it('should test', () => {
      return expect(actor.test({ todo: true })).resolves.toEqual({ todo: true }); // TODO
    });

    it('should run', () => {
      return expect(actor.run({ todo: true })).resolves.toMatchObject({ todo: true }); // TODO
    });
  });
});
