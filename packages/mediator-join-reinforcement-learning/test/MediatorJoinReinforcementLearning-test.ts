import { Actor, Bus, IAction, IActorOutput, IActorTest, Mediator } from '@comunica/core';
import { MediatorJoinReinforcementLearning } from '../lib/MediatorJoinReinforcementLearning';
import * as rdfjs from '@rdfjs/types';

describe('MediatorJoinReinforcementLearning', () => {
  let bus: Bus<DummyActor, IAction, IDummyTest, IDummyTest>;

  beforeEach(() => {
    bus = new Bus({ name: 'bus' });
  });

  describe('An MediatorJoinReinforcementLearning instance', () => {
  });
  describe('Creating graph views', () =>{
    let subjectObjects: rdfjs.Term[][]
    beforeEach(()=>{
      // Query graph of WatDiv Complex Query 1
      subjectObjects = 
      [
        [ { termType: 'Variable', value: 'v0', equals: () =>{return true;} }, { termType: 'Variable', value: 'v1', equals: () =>{return true;} } ],
        [ { termType: 'Variable', value: 'v0', equals: () =>{return true;} }, { termType: 'Variable', value: 'v2', equals: () =>{return true;} } ],
        [ { termType: 'Variable', value: 'v0', equals: () =>{return true;} }, { termType: 'Variable', value: 'v3', equals: () =>{return true;} } ],
        [ { termType: 'Variable', value: 'v0', equals: () =>{return true;} }, { termType: 'Variable', value: 'v4', equals: () =>{return true;} } ],
        [ { termType: 'Variable', value: 'v4', equals: () =>{return true;} }, { termType: 'Variable', value: 'v5', equals: () =>{return true;} } ],
        [ { termType: 'Variable', value: 'v4', equals: () =>{return true;} }, { termType: 'Variable', value: 'v6', equals: () =>{return true;} } ],
        [ { termType: 'Variable', value: 'v7', equals: () =>{return true;} }, { termType: 'Variable', value: 'v6', equals: () =>{return true;} } ],
        [ { termType: 'Variable', value: 'v7', equals: () =>{return true;} }, { termType: 'Variable', value: 'v8', equals: () =>{return true;} } ]
      ]
      MediatorJoinReinforcementLearning.createObjectObjectView(subjectObjects)
    });
    it('should create subject-subject view', () =>{
      const expectedOutput: number[][] = [
      [1,1,1,1,0,0,0,0],
      [1,1,1,1,0,0,0,0],
      [1,1,1,1,0,0,0,0],
      [1,1,1,1,0,0,0,0],
      [0,0,0,0,1,1,0,0],
      [0,0,0,0,1,1,0,0],
      [0,0,0,0,0,0,1,1],
      [0,0,0,0,0,0,1,1]]
      const subjectSubjectGraph = MediatorJoinReinforcementLearning.createSubjectSubjectView(subjectObjects);
      return expect(subjectSubjectGraph).toEqual(expectedOutput);
    });
    it('should create object-object view', () =>{
      const expectedOutput: number[][] = [
        [1,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0],
        [0,0,0,1,0,0,0,0],
        [0,0,0,0,1,0,0,0],
        [0,0,0,0,0,1,1,0],
        [0,0,0,0,0,1,1,0],
        [0,0,0,0,0,0,0,1]]
      const objectObjectGraph = MediatorJoinReinforcementLearning.createObjectObjectView(subjectObjects);
      return expect(objectObjectGraph).toEqual(expectedOutput);
    });
    it('should create object-subject view', () =>{
      const expectedOutput: number[][] = [
        [1,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0],
        [0,0,0,1,1,1,0,0],
        [0,0,0,0,1,0,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,1]]
      const objectSubjectGraph = MediatorJoinReinforcementLearning.createObjectSubjectView(subjectObjects);
      console.log(objectSubjectGraph)
      return expect(objectSubjectGraph).toEqual(expectedOutput);
    });


  })
});

class DummyActor extends Actor<IAction, IDummyTest, IDummyTest> {
  public readonly id: number;

  public constructor(id: number, bus: Bus<DummyActor, IAction, IDummyTest, IDummyTest>) {
    super({ name: `dummy${id}`, bus });
    this.id = id;
  }

  public async test(action: IAction): Promise<IDummyTest> {
    return { field: this.id };
  }

  public async run(action: IAction): Promise<IDummyTest> {
    return { field: this.id };
  }
}

interface IDummyTest extends IActorTest, IActorOutput {
  field: number;
}
