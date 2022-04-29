import {ActionObserver, Actor, IActionObserverArgs, IActorTest} from "@comunica/core";
import { IActionInit,IActorOutputInit } from "@comunica/bus-init";

/**
 * The Timer observer is subscribed to the bus-init and is used to time the execution of a query.
 * This observer is mainly intended to log training episodes of the reinforcement learning agent used for join 
 * optimization.
 */
export class ObserverTimer extends ActionObserver<IActionInit, IActorOutputInit> {
  public startTime: number;
  public elapsedTime: number;
  // public path;
  public modelDirectory: string;

    /**
   * @param args - @defaultNested {<npmd:@comunica/bus-init/^2.0.0/components/ActorInit.jsonld#ActorInit_default_bus>} bus
   */

  constructor(args: IActionObserverArgs<IActionInit, IActorOutputInit>) {
    super(args);
    // Get start time at creation of this observer (start of engine)
    console.log("Creation!!");
    const hrTime = process.hrtime();
    this.startTime = hrTime[0] * 1000000 + hrTime[1] / 1000;

    // this.path = require('path');
    // this.modelDirectory = this.getModelDirectory();
    // Subscribe this observer to the init bus
    this.bus.subscribeObserver(this);
  }

  public onRun(actor: Actor<IActionInit, IActorTest, IActorOutputInit>, // The actor that executes the dereference action
               action: IActionInit, // The dereference action
               output: Promise<IActorOutputInit>, // The promise to a dereference action output, as executed by the actor
  ): void {
    // Wait for the query to finish and get elapsed time
    output.then( () => {
        const hrTimeEnd = process.hrtime();
        const endTime = hrTimeEnd[0] * 1000000 + hrTimeEnd[1] / 1000;
        this.elapsedTime = endTime - this.startTime;
    });
    // this.writeJoinOrderToFile();
  }

    // private getModelDirectory(){          
    // const modelDir: string = this.path.join(__dirname, '../../actor-rdf-join-inner-multi-reinforcement-learning/model');
    // return modelDir;
    // }

    // public writeJoinOrderToFile(){
    //     const fs = require('fs');
    //     const fileLocation = this.path.join(this.modelDirectory, 'joinEpisode.json')
        
    //     const data = {'elapsedTime': this.elapsedTime};
    //     const strData = JSON.stringify(data);
    //     fs.writeFile(fileLocation, strData, function(err: any) {
    //         if(err) {
    //             return console.log(err);
    //         }
    //     }); 
    // }

}
