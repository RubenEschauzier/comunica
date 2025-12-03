import type { EddieOperatorStream } from '../EddieOperatorStream';
import { type IEddieRouterFactory, type IEddieRoutingEntry, RouterBase } from './BaseRouter';

export class RouterFixedMinimalIndex extends RouterBase {
  public updateRouteTable(
    _operators: EddieOperatorStream[],
    routeTable: Record<string, IEddieRoutingEntry[]>,
  ): Record<number, IEddieRoutingEntry[]> {
    return routeTable;
  };
}

export class FixedRouterFactory implements IEddieRouterFactory {
  public createRouter(): RouterFixedMinimalIndex {
    return new RouterFixedMinimalIndex();
  }
}
