import { inspect } from 'util';
import { Logger } from '@comunica/core';

/**
 * A logger that voids everything.
 */
export class LoggerTimer extends Logger {
  startTime: number;
  endTime: number;
  elapsedTime: number;

  private readonly level: string;
  private readonly levelOrdinal: number;
  private readonly actors?: Record<string, boolean>;

  constructor(args: ILoggerTimerArgs){
    super();
    this.level = args.level;
    this.levelOrdinal = Logger.getLevelOrdinal(this.level);
    this.actors = args.actors;

    const hrTime = process.hrtime();
    this.startTime = hrTime[0] * 1000000 + hrTime[1] / 1000;

  }
  public time(){
    const hrTime = process.hrtime();
    this.endTime = hrTime[0] * 1000000 + hrTime[1] / 1000;
    this.elapsedTime = this.endTime - this.startTime;
  }

  public debug(message: string, data?: any): void {
    this.log('debug', message, data);
  }

  public error(message: string, data?: any): void {
    this.log('error', message, data);
  }

  public fatal(message: string, data?: any): void {
    this.log('fatal', message, data);
  }

  public info(message: string, data?: any): void {
    this.log('info', message, data);
  }

  public trace(message: string, data?: any): void {
    this.log('trace', message, data);
  }

  public warn(message: string, data?: any): void {
    this.log('warn', message, data);
  }

  protected log(level: string, message: string, data?: any): void {
    if (Logger.getLevelOrdinal(level) >= this.levelOrdinal &&
      (!data || !('actor' in data) || !this.actors || this.actors[data.actor])) {
      process.stderr.write(`[${new Date().toISOString()}]  ${level.toUpperCase()}: ${message} ${inspect(data)}\n`);
    }
  }
}
export interface ILoggerTimerArgs {
  /**
   * The minimum logging level.
   */
  level: string;
  /**
   * A whitelist of actor IRIs to log for.
   */
  actors?: Record<string, boolean>;
}
