import { Bus } from '@comunica/core';
import { ActorRdfJoinInnerMultiStems } from '../lib/ActorRdfJoinInnerMultiStems';

describe('ActorRdfJoinInnerMultiStems', () => {
  let bus: any;

  beforeEach(() => {
    bus = new Bus({ name: 'bus' });
  });

  describe('An ActorRdfJoinInnerMultiStems instance', () => {
    let actor: ActorRdfJoinInnerMultiStems;

    beforeEach(() => {
      actor = new ActorRdfJoinInnerMultiStems({ name: 'actor', bus });
    });

    it('should test', () => {
      return expect(actor.test({ todo: true })).resolves.toEqual({ todo: true }); // TODO
    });

    it('should run', () => {
      return expect(actor.run({ todo: true })).resolves.toMatchObject({ todo: true }); // TODO
    });
  });
});
