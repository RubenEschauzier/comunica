import { Bus } from '@comunica/core';
import { ActorInitCachesStoreCacheLru } from '../lib/ActorInitCachesStoreCacheLru';

describe('ActorInitCachesStoreCacheLru', () => {
  let bus: any;

  beforeEach(() => {
    bus = new Bus({ name: 'bus' });
  });

  describe('An ActorInitCachesStoreCacheLru instance', () => {
    let actor: ActorInitCachesStoreCacheLru;

    beforeEach(() => {
      actor = new ActorInitCachesStoreCacheLru({ name: 'actor', bus });
    });

    it('should test', () => {
      return expect(actor.test({ todo: true })).resolves.toEqual({ todo: true }); // TODO
    });

    it('should run', () => {
      return expect(actor.run({ todo: true })).resolves.toMatchObject({ todo: true }); // TODO
    });
  });
});
