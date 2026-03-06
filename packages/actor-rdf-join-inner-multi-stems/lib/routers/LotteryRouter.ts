import type { EddieOperatorStream } from '../EddieOperatorStream';
import { type IEddieRouterFactory, type IEddieRoutingEntry, RouterBase } from './BaseRouter';

export class RouterLotteryScheduling extends RouterBase {
  public override updateRouteTable(
    operators: EddieOperatorStream[],
    routeTable: Record<string, IEddieRoutingEntry[]>,
  ): Record<number, IEddieRoutingEntry[]> {
    const ticketMetadata: Record<string, number>[] = operators.map(op => ({ tickets: op.tickets }));

    const updatedRoutingTable: Record<string, IEddieRoutingEntry[]> = {};
    for (const [ doneKey, routing ] of Object.entries(routeTable)) {
      const key = Number.parseInt(doneKey, 10);
      // If size 0 or 1 no choice to be made
      if (routing.length < 2) {
        updatedRoutingTable[key] = routing;
        continue;
      }
      const ticketWeights: number[] = routing.map(x => x.next).map(idx => ticketMetadata[idx].tickets);

      const minScore = Math.min(...ticketWeights);
      const offset = minScore < 0 ? -minScore : 0;
      const ticketWeightsNonNegative = ticketWeights.map(w => w + offset + 1);
      updatedRoutingTable[key] = this.reorderWeightedChoice(routing, ticketWeightsNonNegative);
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

export class LotteryRouterFactory implements IEddieRouterFactory {
  public createRouter(): RouterLotteryScheduling {
    return new RouterLotteryScheduling();
  }
}
