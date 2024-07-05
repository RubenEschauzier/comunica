import { StatisticsHolder } from '@comunica/actor-context-preprocess-set-defaults';
import { KeysStatisticsTracker, KeysTrackableStatistics } from '@comunica/context-entries';
import { ActionContext, Bus } from '@comunica/core';
import { ActorContextPreprocessStatisticLinkDereference } from '../lib/ActorContextPreprocessStatisticLinkDereference';
import { StatisticLinkDereference } from '../lib/StatisticLinkDereference';

describe('ActorContextPreprocessStatisticLinkDereference', () => {
  let bus: any;

  beforeEach(() => {
    bus = new Bus({ name: 'bus' });
  });

  describe('An ActorContextPreprocessStatisticLinkDereference instance', () => {
    let actor: ActorContextPreprocessStatisticLinkDereference;

    beforeEach(() => {
      actor = new ActorContextPreprocessStatisticLinkDereference({ name: 'actor', bus });
    });

    it('should test', async() => {
      await expect(actor.test({ context: new ActionContext() })).resolves.toBeTruthy();
    });

    describe('run', () => {
      it('with only a statisticsHolder', async() => {
        const contextIn = new ActionContext({ [KeysStatisticsTracker.statistics.name]: new StatisticsHolder() });
        const { context: contextOut } = await actor.run({ context: contextIn });
        expect(contextOut).toEqual(new ActionContext({
          [KeysStatisticsTracker.statistics.name]:
           new StatisticsHolder([[ KeysTrackableStatistics.dereferencedLinks.name, new StatisticLinkDereference() ]]),
        }));
      });
      it('should error with empty context', async() => {
        await expect(actor.run({ context: new ActionContext() })).rejects.toThrow(new Error(
          'Tried to track link dereference statistic without statisticsHolder object in context',
        ));
      });
    });
  });
});
