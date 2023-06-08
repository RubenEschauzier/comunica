import { Bus } from '@comunica/core';
import { ActorModelInferenceTreeLstmQValue } from '../lib/ActorModelInferenceTreeLstmQValue';

describe('ActorModelInferenceTreeLstmQValue', () => {
  let bus: any;

  beforeEach(() => {
    bus = new Bus({ name: 'bus' });
  });

  describe('An ActorModelInferenceTreeLstmQValue instance', () => {
    let actor: ActorModelInferenceTreeLstmQValue;

    beforeEach(() => {
      actor = new ActorModelInferenceTreeLstmQValue({ name: 'actor', bus });
    });

    it('should test', () => {
      return expect(actor.test({ todo: true })).resolves.toEqual({ todo: true }); // TODO
    });

    it('should run', () => {
      return expect(actor.run({ todo: true })).resolves.toMatchObject({ todo: true }); // TODO
    });
  });
});
