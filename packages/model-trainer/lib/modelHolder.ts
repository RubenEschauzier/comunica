import { graphConvolutionModel } from "@comunica/actor-rdf-join-inner-multi-reinforcement-learning/lib/GraphNeuralNetwork";

export class modelHolder{
    model: graphConvolutionModel;
    public constructor(){
        this.model = new graphConvolutionModel();
    }

    public setModel(){
        this.model = new graphConvolutionModel();
    }

    public getModel(): graphConvolutionModel{
        return this.model
    }
}