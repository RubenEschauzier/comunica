import { ActorHttp, IActionHttp, IActorHttpOutput, IActorHttpArgs, MediatorHttp } from '@comunica/bus-http';
import { MediatorHttpInvalidate } from '@comunica/bus-http-invalidate';
import { KeysHttp } from '@comunica/context-entries';
import { ActionContext, ActionContextKey, failTest, IActorTest, passTest, passTestVoid, TestResult } from '@comunica/core';
import { request } from 'http';
import type {
  Request as CacheRequest,
  Response as CacheResponse,
  RevalidationPolicy,
} from 'http-cache-semantics';
import CachePolicy = require('http-cache-semantics');

/**
 * A comunica Cache Http Actor. This actor implements the http-cache-semantics over a web-cache
 */
export class ActorHttpCache extends ActorHttp {
  // public cache: Cache;
  private readonly mediatorHttp: MediatorHttp;
  private readonly mediatorHttpInvalidate: MediatorHttpInvalidate;
  private httpCache: IHttpCache;

  public constructor(args: IActorHttpCacheArgs) {
    super(args);
  }

  public async test(action: IActionHttp): Promise<TestResult<IActorTest, undefined>>{
    if (action.context.get(KEY_ATTEMPTED_CACHE)){
      return failTest('Can not use cached response for this request.')
    }
    return passTest({ time: Number.NEGATIVE_INFINITY });
  }

  public async run(action: IActionHttp): Promise<IActorHttpOutput> {
    // Initialize the cache if it doesn't exist yet. This can be initialized with a full
    // web-cache and no policies, however this means that any requests can only be done by the
    // cache after a revalidation request. I could also always make it get the cache from context
    // so the cache can be switched during query execution
    if (this.httpCache === undefined){
      const webCache = action.context.getSafe(KeysHttp.cache);
      this.httpCache = {
        cache: webCache,
        policyCache: {}
      };  
    }

    // Poll cache
    const cacheRequest = new Request(action.input, action.init);
    const cacheResponse = await this.requestCached(cacheRequest);
    console.log(cacheRequest)
    console.log(cacheResponse);
    // Not in cache
    if (!cacheResponse || !cacheResponse.policy){
      console.log("Not cached")
      const response = await this.fetchDocument(action);
      if (!response.response){
        return response;
      }
      const newPolicy = this.CachePolicyFactory(cacheRequest, response.response);
      if (newPolicy.storable()){
        this.updateCaches(cacheRequest, response.response.clone(), newPolicy);
      }
      return response;
    }

    const cacheRequestTransformed = ActorHttpCache.requestToCacheRequest(cacheRequest);
    // In cache, but content requires revalidation
    if (cacheResponse.policy && 
      !cacheResponse.policy.satisfiesWithoutRevalidation(cacheRequestTransformed)
    ){
      console.log("Revalidating")
      const revalHeaders = cacheResponse.policy.revalidationHeaders(cacheRequestTransformed);
      let initHeaders: HeadersInit | undefined = undefined;
      if (action.init && action.init.headers){
        initHeaders = action.init.headers
      }
      const newRequestHeaders = this.getRevalidationHeadersForRequest(revalHeaders, initHeaders);
      action.init = {...action.init, headers: newRequestHeaders};

      // Send request to the origin server. The server may respond with status 304
      const validationResponse = await this.fetchDocument(action);

      // Create updated policy and combined response from the old and new data
      const { policy, modified } = cacheResponse.policy.revalidatedPolicy(
          ActorHttpCache.requestToCacheRequest(new Request(action.input, action.init)),
          ActorHttpCache.responseToCacheResponse(validationResponse.response!)
      );

      const response = modified ? validationResponse.response : cacheResponse.response;

      if (modified){
        // Invalidate all entries associated with the modified url.
        this.mediatorHttpInvalidate.mediate(
          {url: cacheRequest.url, context: ActionContext.ensureActionContext()}
        );
      }
      // Update the cache with the newer/fresher response
      // TODO: Should we use the revalidation request for cache or the original request?
      this.updateCaches(cacheRequest, response!.clone(), policy);

      // TODO: We create a new response with new headers. However we discard some parts of the response
      // If problems occur look here first!
      if (policy.responseHeaders()){
        const updatedResponse: Response = new Response(
          response!.body, 
          {headers: <Record<string, string>> policy.responseHeaders()}
        );   
        return {response: updatedResponse };
      }
      return {response: response};
    }
    // Can use cache without revalidation
    console.log("Using without revalidation")
    return {response: cacheResponse.response.clone()};
  }

  public getRevalidationHeadersForRequest(
    revalidationHeaders: CachePolicy.Headers, 
    existingHeaders?: HeadersInit
  ){
    if (!existingHeaders){
      return new Headers(<Record<string, string>> revalidationHeaders);
    }

    const combinedHeaders = new Headers(existingHeaders);
    // Add new headers from revalidation headers. The order is important to 
    // ensure revalidation headers take priority
    const revalidationHeadersNewVersion = new Headers(
      <Record<string, string>> revalidationHeaders
    );
    revalidationHeadersNewVersion.forEach((value, key) => {
      combinedHeaders.set(key, value); // Overwrites if key exists
    });
    return combinedHeaders;
  }

  public async requestCached(request: Request): Promise<ICachePolicyResponse | undefined>{
    const result = await this.httpCache.cache.match(request);
    if (!result){
      return undefined
    }
    const policy: CachePolicy | undefined = this.httpCache.policyCache[JSON.stringify(request)];
    return {
      response: result,
      policy: policy
    }
  }

  public async fetchDocument(action: IActionHttp){
    const response = await this.mediatorHttp.mediate({
      ...action,
      context: action.context.set(KEY_ATTEMPTED_CACHE, true),
    });
    return response;
  }

  /**
   * @author @jaxoncreed (https://github.com/jaxoncreed)
   * @param request 
   * @param response 
   * @returns 
   */
  private CachePolicyFactory(request: Request, response: Response): CachePolicy {
    return new CachePolicy(
      ActorHttpCache.requestToCacheRequest(request), 
      ActorHttpCache.responseToCacheResponse(response),
      cacheOptions
    );
  }

  /**
   * @author @jaxoncreed (https://github.com/jaxoncreed)
   * @param request 
   * @returns 
   */
  public static requestToCacheRequest(
    request: Request,
  ): CacheRequest {
    return { ...request, headers: this.headersToHash(request.headers) };
  }
  
  /**
   * @author @jaxoncreed (https://github.com/jaxoncreed)
   * @param response 
   * @returns 
   */
  public static responseToCacheResponse(
    response: Response,
  ): CacheResponse {
    return { ...response, headers: this.headersToHash(response.headers) };
  }

  public updateCaches(request: Request, response: Response, policy: CachePolicy){
    this.httpCache.cache.put( request, response );
    this.httpCache.policyCache[JSON.stringify(request)] = policy;
  }

}

export interface IActorHttpCacheArgs extends IActorHttpArgs{
  mediatorHttp: MediatorHttp;
  mediatorHttpInvalidate: MediatorHttpInvalidate;
}

export interface ICachePolicyResponse{
  response: Response;
  policy?: CachePolicy;
}

export const KEY_ATTEMPTED_CACHE = new ActionContextKey<boolean>(
  '@comunica/actor-http-cache:attemptedCache',
);

/**
 * Cache options for the http-cache-policy. This should be a parameter for the actor
 */
const cacheOptions = {
  shared: true,
  cacheHeuristic: 0.1,
  immutableMinTimeToLive: 24 * 3600 * 1000, // 24h
  ignoreCargoCult: false,
};

/**
 * Interface for the HTTP cache implemented in this actor. It uses a web-cache compliant cache
 * as base and tracks a seperate cache storing the policy object of the http-cache-semantics package
 * to ensure HTTP caching compliance.
 */
export interface IHttpCache {
  /**
   * Arbitrary web cache supplied by the context
   */
  cache: Cache;

  /**
   * Record storing HTTP cache policies
   */
  policyCache: Record<string, CachePolicy>
}