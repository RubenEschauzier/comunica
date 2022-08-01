import { StateSpaceTree } from '@comunica/mediator-join-reinforcement-learning';
import type { IActionContext, IActionContextKey } from '@comunica/types';
import { Map as IMap } from 'immutable';
import { EpisodeLogger, MCTSJoinInformation, MCTSJoinPredictionOutput } from '@comunica/model-trainer';

/**
 * Implementation of {@link IActionContext} using Immutable.js.
 */
export class ActionContext implements IActionContext {
  private readonly map: IMap<string, any>;
  public episode: EpisodeLogger;

  public constructor(data: Record<string, any> = {}, episodeLogged: EpisodeLogger = new EpisodeLogger()) {
    this.map = IMap<string, any>(data);
    this.episode = episodeLogged;
  }

  /**
   * Will only set the value if the key is not already set.
   */
  public setDefault<V>(key: IActionContextKey<V>, value: V): IActionContext {
    return this.has(key) ? this : this.set(key, value);
  }

  public set<V>(key: IActionContextKey<V>, value: V): IActionContext {
    return this.setRaw(key.name, value);
  }

  public setRaw(key: string, value: any): IActionContext {
    return new ActionContext(this.map.set(key, value), this.episode);
  }

  public setEpisodeTime(time: number): void{
    this.episode.setTime(time);
  }

  public setEpisodeState(joinState: StateSpaceTree): void{
    this.episode.setState(joinState);
  }

  public setJoinStateMasterTree(joinState: MCTSJoinPredictionOutput){
    this.episode.setJoinState(joinState);
  }

  public setNodeIdMapping(key: number, value: number){
    this.episode.setNodeIdMapping(key, value);
  }

  public addEpisodeStateJoinIndexes(joinIndexes: number[]): void{
    this.episode.addJoinIndexes(joinIndexes);
  }

  public getEpisodeState(): StateSpaceTree{
    return this.episode.getState();
  }

  public getEpisodeTime(): number {
    return this.episode.executionTime
  }

  public getJoinStateMasterTree(joinIndexes: number[][]){
    try{
      return this.episode.getJoinStateMasterTree(joinIndexes);
    }
    catch{
      const blankJoinState: MCTSJoinInformation = {N: 0, meanValueJoin: 0, totalValueJoin: 0, 
                                                    priorProbability: 0};
      return blankJoinState
    }
  }

  public getEpisode(){
    return this.episode
  }

  public getNodeIdMappingKey(key: number){
    return this.episode.getNodeIdMappingKey(key);
  }

  public getNodeIdMapping(): Map<number,number>{
    return this.episode.getNodeMapping();
  }
  // public setInPlace<V>(key: IActionContextKey<V>, value: V): void{
  //   this.setInPlaceRaw(key.name, value);
  // }

  // public setInPlaceRaw(key:string, value:any): void {
  //   this.map.set(key,value);
  // }

  public delete<V>(key: IActionContextKey<V>): IActionContext {
    return new ActionContext(this.map.delete(key.name), this.episode);
  }

  public get<V>(key: IActionContextKey<V>): V | undefined {
    return this.getRaw(key.name);
  }

  public getRaw(key: string): any | undefined {
    return this.map.get(key);
  }

  public getSafe<V>(key: IActionContextKey<V>): V {
    if (!this.has(key)) {
      throw new Error(`Context entry ${key.name} is required but not available`);
    }
    return <V> this.get(key);
  }

  public has<V>(key: IActionContextKey<V>): boolean {
    return this.hasRaw(key.name);
  }

  public hasRaw(key: string): boolean {
    return this.map.has(key);
  }

  public merge(...contexts: IActionContext[]): IActionContext {
    // eslint-disable-next-line @typescript-eslint/no-this-alias,consistent-this
    let context: IActionContext = this;
    for (const source of contexts) {
      for (const key of source.keys()) {
        context = context.set(key, source.get(key));
      }
    }
    return context;
  }

  public keys(): IActionContextKey<any>[] {
    return [ ...<any> this.map.keys() ]
      .map(keyName => new ActionContextKey(keyName));
  }

  public toJS(): any {
    return this.map.toJS();
  }

  public toString(): string {
    return `ActionContext(${JSON.stringify(this.map.toJS())})`;
  }

  public [Symbol.for('nodejs.util.inspect.custom')](): string {
    return `ActionContext(${JSON.stringify(this.map.toJS(), null, '  ')})`;
  }

  /**
   * Convert the given object to an action context object if it is not an action context object yet.
   * If it already is an action context object, return the object as-is.
   * @param maybeActionContext An action context or record.
   * @return {ActionContext} An action context object.
   */
  public static ensureActionContext(maybeActionContext?: IActionContext | Record<string, any>): IActionContext {
    // I create a new EpisodeLogger for this function not sure if ok
    return maybeActionContext instanceof ActionContext ?
      maybeActionContext :
      new ActionContext(IMap(maybeActionContext || {}), new EpisodeLogger());
  }
}

/**
 * Simple implementation of {@link IActionContextKey}.
 */
export class ActionContextKey<V> implements IActionContextKey<V> {
  /**
   * A unique context key name.
   */
  public readonly name: string;

  public constructor(name: string) {
    this.name = name;
  }
}
