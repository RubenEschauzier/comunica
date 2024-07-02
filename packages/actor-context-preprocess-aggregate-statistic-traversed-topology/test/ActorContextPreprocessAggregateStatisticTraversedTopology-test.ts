import { Bus } from '@comunica/core';
import { ActorContextPreprocessAggregateStatisticTraversedTopology } from '../lib/ActorContextPreprocessAggregateStatisticTraversedTopology';

describe('ActorContextPreprocessAggregateStatisticTraversedTopology', () => {
  let bus: any;

  beforeEach(() => {
    bus = new Bus({ name: 'bus' });
  });

  describe('An ActorContextPreprocessAggregateStatisticTraversedTopology instance', () => {
    let actor: ActorContextPreprocessAggregateStatisticTraversedTopology;

    beforeEach(() => {
      actor = new ActorContextPreprocessAggregateStatisticTraversedTopology({ name: 'actor', bus });
    });

    it('should test', () => {
      return expect(actor.test({ todo: true })).resolves.toEqual({ todo: true }); // TODO
    });

    it('should run', () => {
      return expect(actor.run({ todo: true })).resolves.toMatchObject({ todo: true }); // TODO
    });
  });
});
