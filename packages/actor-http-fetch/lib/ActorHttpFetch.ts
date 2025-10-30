import type { IActionHttp, IActorHttpOutput, IActorHttpArgs } from '@comunica/bus-http';
import { ActorHttp } from '@comunica/bus-http';
import { KeysCaches, KeysHttp } from '@comunica/context-entries';
import type { TestResult } from '@comunica/core';
import { passTest } from '@comunica/core';
import type { IMediatorTypeTime } from '@comunica/mediatortype-time';

import type {
  Request as CacheRequest,
  Response as CacheResponse,
} from 'http-cache-semantics';

// eslint-disable-next-line ts/no-require-imports
import CachePolicy = require('http-cache-semantics');

import { version as actorVersion } from '../package.json';

import { FetchInitPreprocessor } from './FetchInitPreprocessor';
import type { IFetchInitPreprocessor } from './IFetchInitPreprocessor';

export class ActorHttpFetch extends ActorHttp {
  private readonly fetchInitPreprocessor: IFetchInitPreprocessor;

  private static readonly userAgent = ActorHttp.createUserAgent('ActorHttpFetch', actorVersion);

  public constructor(args: IActorHttpFetchArgs) {
    super(args);
    this.fetchInitPreprocessor = new FetchInitPreprocessor(args.agentOptions);
  }

  public async test(_action: IActionHttp): Promise<TestResult<IMediatorTypeTime>> {
    return passTest({ time: Number.POSITIVE_INFINITY });
  }

  public async run(action: IActionHttp): Promise<IActorHttpOutput> {
    const headers = this.prepareRequestHeaders(action);

    const init: RequestInit = { method: 'GET', ...action.init, headers };

    this.logInfo(action.context, `Requesting ${ActorHttp.getInputUrl(action.input).href}`, () => ({
      headers: ActorHttp.headersToHash(headers),
      method: init.method,
    }));

    // TODO: remove this workaround once this has a fix: https://github.com/inrupt/solid-client-authn-js/issues/1708
    if (action.context.has(KeysHttp.fetch)) {
      init.headers = ActorHttp.headersToHash(headers);
    }

    if (action.context.get(KeysHttp.includeCredentials)) {
      init.credentials = 'include';
    }

    const httpTimeout = action.context.get(KeysHttp.httpTimeout);
    const httpBodyTimeout = action.context.get(KeysHttp.httpBodyTimeout);
    const fetchFunction = action.context.get(KeysHttp.fetch) ?? fetch;
    const requestInit = await this.fetchInitPreprocessor.handle(init);

    let timeoutCallback: () => void;
    let timeoutHandle: NodeJS.Timeout | undefined;

    if (httpTimeout) {
      const abortController = new AbortController();
      requestInit.signal = abortController.signal;
      timeoutCallback = () => abortController.abort(new Error(`Fetch timed out for ${ActorHttp.getInputUrl(action.input).href} after ${httpTimeout} ms`));
      timeoutHandle = setTimeout(() => timeoutCallback(), httpTimeout);
    }

    const policyCache = action.context.get(KeysCaches.policyCache);
    const storeCache = action.context.get(KeysCaches.storeCache);
    if (action.validate &&
        policyCache &&
        storeCache &&
        storeCache.get(ActorHttpFetch.getUrl(action.input))
    ) {
      const oldPolicy = action.validate;
      const requestToValidateCache = ActorHttpFetch.requestToCacheRequest(
        new Request(action.input, requestInit),
      );

      // Empty response will be propegated by any wrappers without processing.
      // Policy does not need to be updated
      if (oldPolicy.satisfiesWithoutRevalidation(requestToValidateCache)) {
        return { validationOutput: { isValidated: true, requestMade: false }};
      }

      // We have to validate the response and then return it
      const revalidationHeaders = oldPolicy.revalidationHeaders(requestToValidateCache);
      const revalidationRequestInit = { ...requestInit, headers: {
        ...(<Record<string, string>> (requestInit?.headers)),
        ...(<Record<string, string>> revalidationHeaders),
      }};

      const response = await fetchFunction(action.input, revalidationRequestInit);

      if (httpTimeout && (!httpBodyTimeout || !response.body)) {
        clearTimeout(timeoutHandle);
      }

      const { policy, modified } = oldPolicy.revalidatedPolicy(
        ActorHttpFetch.requestToCacheRequest(new Request(action.input, revalidationRequestInit)),
        ActorHttpFetch.responseToCacheResponse(response),
      );
        // Update the policy after revalidation
      policyCache.set(ActorHttpFetch.getUrl(action.input), policy);

      // If the server returned 304 we can reuse cache and don't have to parse the result
      if (!modified) {
        return { validationOutput: { isValidated: true, requestMade: true }};
      }
      // If it is modified we return the response and process like normal
      return { response, validationOutput: { isValidated: false, requestMade: true }};
    }
    const response = await fetchFunction(action.input, requestInit);

    if (httpTimeout && (!httpBodyTimeout || !response.body)) {
      clearTimeout(timeoutHandle);
    }

    if (policyCache && storeCache) {
      const cacheRequest = ActorHttpFetch.requestToCacheRequest(
        new Request(action.input, requestInit),
      );
      const newPolicyResponse = new CachePolicy(
        cacheRequest,
        ActorHttpFetch.responseToCacheResponse(response),
      );
      policyCache.set(ActorHttpFetch.getUrl(action.input), newPolicyResponse);
    }

    return { response };
  }

  /**
   * Prepares the request headers, taking into account the environment.
   * @param {IActionHttp} action The HTTP action
   * @returns {Headers} Headers
   */
  public prepareRequestHeaders(action: IActionHttp): Headers {
    const headers = new Headers(action.init?.headers);

    if (ActorHttp.isBrowser()) {
      // When running in a browser, the User-Agent header should never be set
      headers.delete('user-agent');
    } else if (!headers.has('user-agent')) {
      // Otherwise, if no header value is provided, use the actor one
      headers.set('user-agent', ActorHttpFetch.userAgent!);
    }

    const authString = action.context.get(KeysHttp.auth);
    if (authString) {
      headers.set('Authorization', `Basic ${ActorHttpFetch.stringToBase64(authString)}`);
    }

    return headers;
  }

  /**
   * Converts a string, including ones with Unicode symbols, to Base64 encoding.
   * This function was adapted from the MDN example function here:
   * https://developer.mozilla.org/en-US/docs/Glossary/Base64#the_unicode_problem
   * @param {string} value The string value to encode
   * @returns {string} The Base64-encoded value
   */
  public static stringToBase64(value: string): string {
    const bytes = new TextEncoder().encode(value);
    const binString = Array.from(bytes, byte => String.fromCodePoint(byte)).join('');
    return btoa(binString);
  }

  public static requestToCacheRequest(request: Request): CacheRequest {
    return { ...request, headers: this.headersToHash(request.headers) };
  }

  public static responseToCacheResponse(response: Response): CacheResponse {
    return { ...response, headers: this.headersToHash(response.headers) };
  }

  public static getUrl(input: RequestInfo): string {
    if (input instanceof Request) {
      return input.url;
    }
    return input;
  }
}

export interface IActorHttpFetchArgs extends IActorHttpArgs {
  /**
   * The agent options for the HTTP agent
   * @range {json}
   * @default {{ "keepAlive": true, "maxSockets": 5 }}
   */
  agentOptions?: Record<string, any>;
}
