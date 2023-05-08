import { IAggregateValues, IResultSetRepresentation, MediatorJoinReinforcementLearning } from "@comunica/mediator-join-reinforcement-learning";
import * as fs from 'graceful-fs';
import * as tf from '@tensorflow/tfjs-node';

export class ExperienceBuffer{
    // Data structures to make fifo buffer
    experienceBufferMap: Map<string,Map<string,IExperience>>;
    queryLeafFeatures: Map<string, IResultSetRepresentation>;
    experienceAgeTracker: IExperienceKey[];
    maxSize: number;
    // Data structures to keep track of queries and their keys
    queryToKey: Map<string, string>;
    numUniqueKeys: number;

    /**
     * FIFO query execution buffer. We use this for periodical training. This allows for better data efficiency since the model is incredibly light weight
     * while query execution is the main bottleneck.
     * @param maxSize Maximal number of experiences in buffer
     * @param numQueries The total number of queries in the training data
    */
    constructor(maxSize: number){
        this.experienceBufferMap = new Map<string, Map<string, IExperience>>();
        this.queryLeafFeatures = new Map<string, IResultSetRepresentation>();
        this.experienceAgeTracker = [];
        this.maxSize = maxSize;

        this.queryToKey = new Map<string,string>();
        this.numUniqueKeys = 0;
    }

    public getExperience(queryKey: string, joinPlanKey: string){
        return this.experienceBufferMap.get(queryKey)!.get(joinPlanKey);
    }

    public getRandomExperience(): [IExperience, IExperienceKey] {
        const index = (Math.random() * (this.getSize()) ) << 0;
        const key = this.experienceAgeTracker[index];
        if (!this.getExperience(key.query, key.joinPlanKey)){
            this.printExperiences();
            console.error("Got invalid index or key");
            console.log(key);
        }

        return [this.getExperience(key.query, key.joinPlanKey)!, key];
    }

    public getFeatures(query: string){
        const features = this.queryLeafFeatures.get(query);
        if (!features){
            console.error("Query requested with no leaf features");
        }
        return features;
    }
    /**
     * Function to set a new experience. 
     * For incomplete join plans we keep track of the best recorded execution time from the partial join plan
     * For complete join plans we track the average execution time during training
     * When the maximum number of experiences is reached, the buffer acts as a queue and removes the oldest experience
     * When an already existing experience is revisted we DON'T refresh the age of the experience 
     * 
     * @param queryKey The query number of the executed query
     * @param joinPlanKey The key representing the experienced joinplan
     * @param experience The information obtained during query execution
     * @returns 
     */
    public setExperience(queryKey: string, joinPlanKey: string, experience: IExperience, runningMomentsY: IAggregateValues){
        console.log(`Before adding experience ${this.getSize()}`)

        const fullJoinPlanKeyLength = this.getNumJoinsQuery(queryKey);
        if (!fullJoinPlanKeyLength){
            console.log(queryKey)
            console.log(this.queryLeafFeatures)
            console.trace();
            throw new Error("Uninitialised query key");
        }
        
        const existingExperience = this.getExperience(queryKey, joinPlanKey);
        const joinPlanKeyArray = MediatorJoinReinforcementLearning.keyToIdx(joinPlanKey);

        // True if we have complete join plan
        const fullJoinPlan: boolean = joinPlanKeyArray.length == fullJoinPlanKeyLength
        
        if (existingExperience){
            console.log("Existing experience");
            console.log(queryKey);
            console.log(joinPlanKey);
            /**
             * Note that this function is not entirely correct, if the average execution time goes up due to chance it is not reflected in the execution time
             *of partial join plans that use the execution time of a complete join plan. This difference should be small though.
             */
            if (fullJoinPlan){
                // Update unnormalized execution time average
                existingExperience.actualExecutionTimeRaw += ((experience.actualExecutionTimeRaw - existingExperience.actualExecutionTimeRaw)/(existingExperience.N+1));
                existingExperience.N += 1;
                // Update the normalized execution time using new average raw value
                existingExperience.actualExecutionTimeNorm = (existingExperience.actualExecutionTimeRaw - runningMomentsY.mean) / runningMomentsY.std;
                return;    
            }
            // If partial we update to see if the recorded raw execution time is better than previous max, we use raw because the normalized can change
            // due to underlying statistics changing
            const existingExecutionTime = existingExperience.actualExecutionTimeRaw;
            if (existingExecutionTime > experience.actualExecutionTimeRaw){
                existingExperience.actualExecutionTimeRaw = experience.actualExecutionTimeRaw;
            }
            existingExperience.N+=1
            // Update the normalized execution times (At the start this can give big changes, but should be stable when num executions -> infinity)
            // We update even if the recorded experience is worse than current, to reflect the changes in distribution of Y
            existingExperience.actualExecutionTimeNorm = (existingExperience.actualExecutionTimeRaw - runningMomentsY.mean)/runningMomentsY.std;
            return;
        }
        console.log("New experience!")

        // If it doesn't exist we set new experience
        this.experienceBufferMap.get(queryKey)!.set(joinPlanKey, experience);
        // Add it to the 'queue' to keep track of age
        this.experienceAgeTracker.push({query: queryKey, joinPlanKey: joinPlanKey});

        // If size exceeds max from push we remove first pushed element from the age tracker and the map
        if (this.getSize()>this.maxSize){
            const removedElement: IExperienceKey = this.experienceAgeTracker.shift()!;
            this.experienceBufferMap.get(removedElement.query)!.delete(removedElement.joinPlanKey);

        }
        console.log(`After adding experience ${this.getSize()}`)
        return;
    }
    public getQueryKey(queryString: string){
        const queryKey = this.queryToKey.get(queryString)
        if (!queryKey){
            console.trace()
            throw new Error("Requesting query that is not registered");
        }
        return queryKey;
    }
    
    /**
     * Function to check initialisation of query string in experienceBuffer
     * @param queryString The query string we want to check
     * @returns Boolean indicating if query is initialised in experienceBuffer
     */
    public queryExists(queryString: string){
        return this.queryToKey.get(queryString)? true: false;
    }

    public initQueryInformation(queryString: string, leafFeatures: IResultSetRepresentation){
        // First we convert to string key
        const queryStringKey = this.numUniqueKeys.toString();

        // We record this queryString in our query to key map
        this.queryToKey.set(queryString, queryStringKey);

        // Update our leaf features
        this.setLeafFeaturesQuery(queryStringKey, leafFeatures);

        // Increment our string keys
        this.numUniqueKeys += 1;
    }

    public setLeafFeaturesQuery(queryKey: string, leafFeatures: IResultSetRepresentation){
        this.queryLeafFeatures.set(queryKey, leafFeatures);
        this.experienceBufferMap.set(queryKey, new Map<string, IExperience>());
    }

    public getNumJoinsQuery(queryKey: string){
        return this.queryLeafFeatures.get(queryKey)?.hiddenStates.length;
    }

    public getSize(){
        return this.experienceAgeTracker.length;
    }

    public saveBuffer(fileLocation: string){
        this.saveLeafFeatures(fileLocation+'/leafFeatures.txt');
        this.saveExperienceTracking(fileLocation+'/ageTracker.txt');
        this.saveExperienceMap(fileLocation + '/experienceMap.txt');
        this.saveInitTracking(fileLocation + '/queryToKey.txt', fileLocation + '/numUnique.txt');
    }
    
    public loadBuffer(fileLocation:string){
        this.loadLeafFeatures(fileLocation+'/leafFeatures.txt');
        this.loadExperienceTracking(fileLocation+'/ageTracker.txt');
        this.loadExperienceMap(fileLocation + '/experienceMap.txt');
        this.loadInitTracking(fileLocation + '/queryToKey.txt', fileLocation + '/numUnique.txt');

    }

    public saveLeafFeatures(fileLocation: string){
        const objToSave = new Map<string, number[][][][]>();
        for (const [key,value] of this.queryLeafFeatures.entries()){
            const featureArrayHS = value.hiddenStates.map(x=>x.arraySync() as number[][]);
            const featureArrayMem = value.memoryCell.map(x=>x.arraySync() as number[][]);
            objToSave.set(key,[featureArrayHS, featureArrayMem]); 
        }
        fs.writeFileSync(fileLocation, JSON.stringify([...objToSave]));
    }

    public loadLeafFeatures(fileLocation: string){
        const data = JSON.parse(fs.readFileSync(fileLocation, 'utf-8'));
        const newLeafFeatures: Map<string, IResultSetRepresentation> = new Map<string, IResultSetRepresentation>();
        for (const entry of data){
            const key = entry[0];
            const features = entry[1];
            const featureTensorHS = features[0].map((x: number[][]) =>tf.tensor(x));
            const featureTensorMem = features[1].map((x: number[][]) =>tf.tensor(x));
            newLeafFeatures.set(key, {hiddenStates: featureTensorHS, memoryCell: featureTensorMem});
        }
        this.queryLeafFeatures = newLeafFeatures;
    }

    public saveExperienceTracking(fileLocation: string){
        fs.writeFileSync(fileLocation, JSON.stringify(this.experienceAgeTracker));
    }

    public loadExperienceTracking(fileLocation: string){
        const data: IExperienceKey[] = JSON.parse(fs.readFileSync(fileLocation, 'utf-8'));
        this.experienceAgeTracker = data;
        // const newExpTracker: IExperienceKey[] = data.map((x: string[])=>{query: x[0], joinPlanKey: x[1]})
    }

    public saveExperienceMap(fileLocation: string){
        const dataToSave = [];
        for (const [key,value] of this.experienceBufferMap){
            dataToSave.push([key, [...value]])
        }
        fs.writeFileSync(fileLocation, JSON.stringify(dataToSave));
    }

    public loadExperienceMap(fileLocation: string){
        const data = JSON.parse(fs.readFileSync(fileLocation, 'utf-8'));
        const newExperienceMap = new Map<string, Map<string, IExperience>>();
        for (const expMap of data){
            const queryTemplateMap = new Map<string, IExperience>();
            const queryKey = expMap[0];
            const joinPlans = expMap[1]
            for (const joinPlan of joinPlans){
                queryTemplateMap.set(joinPlan[0], joinPlan[1]);
            }
            newExperienceMap.set(queryKey, queryTemplateMap);
        }
        this.experienceBufferMap = newExperienceMap;
    }

    public saveInitTracking(fileLocationMap: string, fileLocationUnique: string){
        fs.writeFileSync(fileLocationMap, JSON.stringify([...this.queryToKey]));
        fs.writeFileSync(fileLocationUnique, JSON.stringify(this.numUniqueKeys));
    }

    public loadInitTracking(fileLocationMap: string, fileLocationUnique: string){
        const loadedQueryToKeyMap: Map<string, string> = new Map<string, string>();
        const queryToKeyMapArray: string[] = JSON.parse(fs.readFileSync(fileLocationMap, 'utf-8'));
        for (let i=0;i<queryToKeyMapArray.length;i++){
            loadedQueryToKeyMap.set(queryToKeyMapArray[i][0], queryToKeyMapArray[i][1]);
        }
        this.queryToKey = loadedQueryToKeyMap;
        const uniqueKeys: number = JSON.parse(fs.readFileSync(fileLocationUnique, 'utf-8'));
        this.numUniqueKeys = uniqueKeys;

    }

    public printExperiences(){
        for (const [key, value] of this.experienceBufferMap.entries()){
            if (value.size>0){
                console.log(key);
                console.log(value);
            }
        }
    }

}

export interface IExperience{
    actualExecutionTimeNorm: number;
    actualExecutionTimeRaw: number;
    joinIndexes: number[][]
    N: number;
}
export interface IExperienceKey{
    query: string;
    joinPlanKey: string;
}