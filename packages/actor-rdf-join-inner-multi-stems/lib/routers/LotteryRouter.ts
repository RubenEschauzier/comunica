import { Algebra } from '@comunica/utils-algebra';
import type { StemsOperatorStream } from '../StemsOperatorStream';
import { type IStemsRouterFactory, type IStemsRoutingEntry, RouterBase } from './BaseRouter';

export class RouterLotteryScheduling extends RouterBase {
  public override updateRouteTable(
    operators: StemsOperatorStream[],
    routeTable: Record<string, IStemsRoutingEntry[][]>,
  ): Record<number, IStemsRoutingEntry[][]> {
    const ticketMetadata: Record<string, number>[] = operators.map(op => ({ tickets: op.tickets }));

    const updatedRoutingTable: Record<string, IStemsRoutingEntry[][]> = {};
    for (const [ doneKey, routing ] of Object.entries(routeTable)) {
      const key = Number.parseInt(doneKey, 10);

      const updatedRouting = routing.map(exRoute => { 
        if (exRoute.length < 2) {
          return exRoute;
        }
        const ticketWeights: number[] = exRoute.map(x => x.next).map(idx => ticketMetadata[idx].tickets);
        const minScore = Math.min(...ticketWeights);
        const offset = minScore < 0 ? -minScore : 0;
        const ticketWeightsNonNegative = ticketWeights.map(w => w + offset + 1);
        return this.reorderWeightedChoice(exRoute, ticketWeightsNonNegative);
      });

      updatedRoutingTable[key] = updatedRouting;
    }
    return updatedRoutingTable;
  }

  protected reorderWeightedChoice<T>(items: T[], weights: number[]): T[] {
    const reordered = [ ...items ];
    const total = weights.reduce((a, b) => a + b, 0);
    const r = Math.random() * total;
    let acc = 0;
    for (let i = 0; i < items.length; i++) {
      acc += weights[i];
      if (r < acc) {
        const temp = reordered[0];
        reordered[0] = reordered[i];
        reordered[i] = temp;
        return reordered;
      };
    }
    const temp = reordered[0];
    reordered[0] = reordered[items.length - 1];
    reordered[items.length - 1] = temp;
    return reordered;
  }
}

export class LotteryRouterFactory implements IStemsRouterFactory {
  public createRouter(patterns: Algebra.Pattern[]): RouterLotteryScheduling {
    return new RouterLotteryScheduling(patterns);
  }
}
