import type { IActionHttp, IActorHttpArgs, IActorHttpOutput, MediatorHttp } from '@comunica/bus-http';
import { ActorHttp } from '@comunica/bus-http';
import { KeysHttpWayback, KeysHttpProxy } from '@comunica/context-entries';
import type { IActorTest, TestResult } from '@comunica/core';
import { passTestVoid } from '@comunica/core';
import type { IActionContext, IProxyHandler, IRequest } from '@comunica/types';
import { stringify as stringifyStream } from '@jeswr/stream-to-string';

const WAYBACK_URL = 'http://wayback.archive-it.org/';

function addWayback(action: IRequest): IRequest {
  const request = new Request(action.input, action.init);
  return {
    input: new Request(new URL(`/${request.url}`, WAYBACK_URL), request),
  };
}

function getProxyHandler(context: IActionContext): (action: IRequest) => Promise<IRequest> {
  const handler = context.get<IProxyHandler>(KeysHttpProxy.httpProxyHandler);
  if (handler) {
    return (action: IRequest) => handler.getProxy(addWayback(action));
  }
  return (action: IRequest) => Promise.resolve(addWayback(action));
}

/**
 * A Comunica actor to intercept HTTP requests to recover broken links using the WayBack Machine
 */
export class ActorHttpWayback extends ActorHttp {
  public readonly mediatorHttp: MediatorHttp;

  public constructor(args: IActorHttpWaybackArgs) {
    super(args);
  }

  public async test(_action: IActionHttp): Promise<TestResult<IActorTest>> {
    return passTestVoid();
  }

  public async run(action: IActionHttp): Promise<IActorHttpOutput> {
    let result = await this.mediatorHttp.mediate(action);

    const res = result.response;

    // If we get no response, it is a validated request from cache
    if (!res) {
      return result;
    }

    if (res.status === 404 && action.context.get(KeysHttpWayback.recoverBrokenLinks)) {
      let fallbackResult = await this.mediatorHttp.mediate({
        ...action,
        context: action.context
          .set(KeysHttpWayback.recoverBrokenLinks, false)
          .set<IProxyHandler>(KeysHttpProxy.httpProxyHandler, { getProxy: getProxyHandler(action.context) }),
      });
      const resFallback = fallbackResult.response;
      if (!resFallback) {
        return fallbackResult;
      }

      // If the wayback machine returns a 200 status then use that result
      if (resFallback.status === 200) {
        [ result, fallbackResult ] = [ fallbackResult, result ];
      }

      // Consume stream to avoid process
      const { body } = resFallback;
      if (body) {
        if ('cancel' in body && typeof body.cancel === 'function') {
          await body.cancel();
        } else if ('destroy' in body && typeof (<any>body).destroy === 'function') {
          (<any>body).destroy();
        } else {
          await stringifyStream(ActorHttp.toNodeReadable(body));
        }
      }
    }

    return result;
  }
}

export interface IActorHttpWaybackArgs extends IActorHttpArgs {
  mediatorHttp: MediatorHttp;
}
