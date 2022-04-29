export class episodeLogger{
    joinOrders: number[][];
    modelDirectory: string;
    startTime: number
    elapsedTime: number;
    path;

    constructor(){
        this.path = require('path');
        this.joinOrders = []
        this.modelDirectory = this.getModelDirectory();

        const hrTime = process.hrtime();
        this.startTime = hrTime[0] * 1000000 + hrTime[1] / 1000;
    }
    public logSelectedJoin(joinCombination: number[]){
        this.joinOrders.push(joinCombination);
    }

    public writeJoinOrderToFile(){
        const fs = require('fs');
        const fileLocation = this.path.join(this.modelDirectory, 'joinEpisode.json')
        
        const data = {'joinOrder': this.joinOrders, 'elapsedTime': this.elapsedTime};
        const strData = JSON.stringify(data);
        fs.writeFile(fileLocation, strData, function(err: any) {
            if(err) {
                return console.log(err);
            }
        }); 
    }

    public updateFinalTime(){
        const finalTime: number[] = process.hrtime();
        const finalTimeSeconds: number = finalTime[0] * 1000000 + finalTime[1] / 1000;
        this.elapsedTime = finalTimeSeconds - this.startTime;
    }
    private getModelDirectory(){          
        const modelDir: string = this.path.join(__dirname, '../../actor-rdf-join-inner-multi-reinforcement-learning/model');
        return modelDir;
    }
}

let test = new episodeLogger();
test.logSelectedJoin([0,5]);
test.logSelectedJoin([1,3]);
test.updateFinalTime();
test.writeJoinOrderToFile();
