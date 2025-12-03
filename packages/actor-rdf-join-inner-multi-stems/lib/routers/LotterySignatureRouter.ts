import type { EddieOperatorStream, ISelectivityData } from '../EddieOperatorStream';
import type { IEddieRouterFactory, IEddieRoutingEntry } from './BaseRouter';
import { RouterLotteryScheduling } from './LotteryRouter';

export class RouterLotterySchedulingSignature extends RouterLotteryScheduling {
  public override updateRouteTable(
    operators: EddieOperatorStream[],
    routeTable: Record<string, IEddieRoutingEntry[]>,
  ): Record<number, IEddieRoutingEntry[]> {
    const selectivityMetadata: Record<string, Record<number, ISelectivityData>>[] = this.collectMetadata(operators);

    const updatedRoutingTable: Record<string, IEddieRoutingEntry[]> = {};
    for (const [ doneKey, routing ] of Object.entries(routeTable)) {
      const key = Number.parseInt(doneKey, 10);
      // If size 0 or 1 no choice to be made
      if (routing.length < 2) {
        updatedRoutingTable[key] = routing;
        continue;
      }
      const ticketWeights: number[] = [];
      for (const idx of routing.map(x => x.next)) {
        const selectivityData = selectivityMetadata[idx].selectivity[key];
        if (selectivityData) {
          ticketWeights.push(selectivityData.in - selectivityData.out);
        } else {
          ticketWeights.push(0);
        }
      }

      const minScore = Math.min(...ticketWeights);
      const offset = minScore < 0 ? -minScore : 0;
      const ticketWeightsNonNegative = ticketWeights.map(w => w + offset + 1);
      updatedRoutingTable[key] = this.reorderWeightedChoice(routing, ticketWeightsNonNegative);
    }
    return updatedRoutingTable;
  }

  protected collectMetadata(operators: EddieOperatorStream[]): { selectivity: Record<number, ISelectivityData> }[] {
    return operators.map(op => ({ selectivity: op.selectivities }));
  }

  protected shuffle<T>(array: T[]): T[] {
    const result = [ ...array ];
    for (let i = result.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [ result[i], result[j] ] = [ result[j], result[i] ];
    }
    return result;
  }
}

export class LotterySignatureRouterFactory implements IEddieRouterFactory {
  public createRouter(): RouterLotterySchedulingSignature {
    return new RouterLotterySchedulingSignature();
  }
}
