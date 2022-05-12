
/**
 * Hardcoded timer for query timing, should be an observer but this is the quick and dirty way.
 */
export class Timer {
  public startTime: number;
  public elapsedTime: number;

  constructor() {
    // Get start time at creation of this observer (start of engine)
    const hrTime = process.hrtime();
    this.startTime = hrTime[0] + hrTime[1] /1000000000 ;
  }

  public updateEndTime(): number {
    const hrTimeEnd = process.hrtime();
    const endTime = hrTimeEnd[0] + hrTimeEnd[1] / 1000000000 ;
    this.elapsedTime = endTime - this.startTime;
    console.log(this.elapsedTime)
    return this.elapsedTime
  }
}
