import { LRUCache } from 'lru-cache';

export class LRUWebCache implements Cache {
    private readonly cache: LRUCache<RequestInfo | URL, Response>;
    private readonly verbose: boolean;

    public constructor(options: LRUCache.Options<RequestInfo | URL, Response, unknown>, verbose=false){
        this.cache = new LRUCache(options);
        this.verbose = verbose
    }
    public async add(request: RequestInfo | URL): Promise<void> {
        throw new Error("Method not implemented.");
    }
    public async addAll(requests: RequestInfo[]): Promise<void> {
        throw new Error("Method not implemented.");
    }
    public async delete(request: RequestInfo | URL, options?: CacheQueryOptions): Promise<boolean> {
        throw new Error("Method not implemented.");
    }
    public async keys(request?: RequestInfo | URL, options?: CacheQueryOptions): Promise<ReadonlyArray<Request>> {

        throw new Error("Method not implemented.");
    }
    public async match(request: RequestInfo | URL, options?: CacheQueryOptions): Promise<Response | undefined> {
        console.log(request);
        return this.cache.get(request);
    }
    public async matchAll(request?: RequestInfo | URL, options?: CacheQueryOptions): Promise<ReadonlyArray<Response>> {
        throw new Error("Method not implemented.");
    }
    public async put(request: RequestInfo | URL, response: Response): Promise<void> {
        let responseSize = response.headers.get('Content-Length');
        // TODO: Should I include it in a logger or is console.warn ok? If I use console.warn you maybe can't silence it
        // unless you use the verbose thing. Also what should the fallback value be?
        if (responseSize === null && this.verbose){
            console.warn("Response without Content-Length header received");
            responseSize = '1024';
        }
        this.cache.set(request, response, {size: Number(responseSize)});
    }
}