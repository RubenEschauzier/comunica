
/**
 * Hardcoded timer for query timing, should be an observer but this is the quick and dirty way.
 */
export class Timer {
  public startTime: number;
  public elapsedTime: number;
  public startTimeTest: number;

  constructor() {
    // Get start time at creation of this observer (start of engine)
    const hrTime = process.hrtime();
    this.startTime = hrTime[0] + hrTime[1] / 1000000000 ;
    this.startTimeTest = new Date().getTime();

  }

  public updateEndTime(): number {
    const hrTimeEnd = process.hrtime();
    const endTime = hrTimeEnd[0] + hrTimeEnd[1] / 1000000000 ;
    this.elapsedTime = endTime - this.startTime;

    const testElapsedTime: number = new Date().getTime() - this.startTimeTest;
    return this.elapsedTime
  }
}
