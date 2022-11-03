import { denseOwnImplementation, graphConvolutionLayer, graphConvolutionModel } from "@comunica/actor-rdf-join-inner-multi-reinforcement-learning/lib/GraphNeuralNetwork";

export class modelHolder{
    model: graphConvolutionModel;
    instantiated: boolean;
    public constructor(){
        this.model = new graphConvolutionModel();
        this.instantiated = false;


    }
    public async instantiateModel(){
        await this.model.instantiateModel()
        // Should do better thing for this
        // await this.model.loadModel('testDir');
        const test = this.model.layersHidden[0][0] as graphConvolutionLayer
        this.instantiated=true;
    }

    public setModel(){
        this.model = new graphConvolutionModel();
    }

    public getModel(): graphConvolutionModel{
        return this.model
    }
}