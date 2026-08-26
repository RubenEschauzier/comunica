import type { StemsOperatorStream, ISelectivityData } from '../StemsOperatorStream';
import type { IStemsRouterFactory, IStemsRoutingEntry } from './BaseRouter';
import { RouterLotteryScheduling } from './LotteryRouter';

export class RouterLotterySchedulingSignature extends RouterLotteryScheduling {
  public override updateRouteTable(
    operators: StemsOperatorStream[],
    routeTable: Record<string, IStemsRoutingEntry[][]>,
  ): Record<number, IStemsRoutingEntry[][]> {
    const selectivityMetadata: Record<string, Record<number, ISelectivityData>>[] = this.collectMetadata(operators);

    const updatedRoutingTable: Record<string, IStemsRoutingEntry[][]> = {};
    for (const [ doneKey, routing ] of Object.entries(routeTable)) {
      const key = Number.parseInt(doneKey, 10);

      const updatedRoutings: IStemsRoutingEntry[][] = [];
      for (const route of routing){
        // If size 0 or 1 no choice to be made
        if (route.length < 2) {
          updatedRoutings.push(route);
          continue;
        }
        const ticketWeights: number[] = [];
        for (const idx of route.map(x => x.next)) {
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

        updatedRoutings.push(this.reorderWeightedChoice(route, ticketWeightsNonNegative));
      }
      updatedRoutingTable[key] = updatedRoutings;
    }
    
    return updatedRoutingTable;
  }

  protected collectMetadata(operators: StemsOperatorStream[])
  : { selectivity: Record<number, ISelectivityData> }[] {
    return operators.map(op => ({ selectivity: op.selectivitiesSignatures }));
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

export class LotterySignatureRouterFactory implements IStemsRouterFactory {
  public createRouter(): RouterLotterySchedulingSignature {
    return new RouterLotterySchedulingSignature();
  }
}
