import { Algebra } from '@comunica/utils-algebra';
import type { StemsOperatorStream } from '../StemsOperatorStream';
import { type IStemsRouterFactory, type IStemsRoutingEntry, RouterBase } from './BaseRouter';

export class RouterFixedMinimalIndex extends RouterBase {
  public updateRouteTable(
    _operators: StemsOperatorStream[],
    routeTable: Record<string, IStemsRoutingEntry[][]>,
  ): Record<number, IStemsRoutingEntry[][]> {
    return routeTable;
  };
}

export class FixedRouterFactory implements IStemsRouterFactory {
  public createRouter(patterns: Algebra.Pattern[]): RouterFixedMinimalIndex {
    return new RouterFixedMinimalIndex(patterns);
  }
}
