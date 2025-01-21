import { Bus } from '@comunica/core';
import { ActorInitCachesPolicyCacheLru } from '../lib/ActorInitCachesPolicyCacheLru';

describe('ActorInitCachesPolicyCacheLru', () => {
  let bus: any;

  beforeEach(() => {
    bus = new Bus({ name: 'bus' });
  });

  describe('An ActorInitCachesPolicyCacheLru instance', () => {
    let actor: ActorInitCachesPolicyCacheLru;

    beforeEach(() => {
      actor = new ActorInitCachesPolicyCacheLru({ name: 'actor', bus });
    });

    it('should test', () => {
      return expect(actor.test({ todo: true })).resolves.toEqual({ todo: true }); // TODO
    });

    it('should run', () => {
      return expect(actor.run({ todo: true })).resolves.toMatchObject({ todo: true }); // TODO
    });
  });
});
