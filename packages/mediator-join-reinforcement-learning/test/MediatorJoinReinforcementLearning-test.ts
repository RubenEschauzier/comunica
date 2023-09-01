import { Actor, Bus, IAction, IActorOutput, IActorTest, Mediator } from '@comunica/core';
import { MediatorJoinReinforcementLearning } from '../lib/MediatorJoinReinforcementLearning';
import * as rdfjs from '@rdfjs/types';

describe('MediatorJoinReinforcementLearning', () => {
  let bus: Bus<DummyActor, IAction, IDummyTest, IDummyTest>;

  beforeEach(() => {
    bus = new Bus({ name: 'bus' });
  });

  describe('An MediatorJoinReinforcementLearning instance', () => {
    let mediator: MediatorJoinReinforcementLearning<DummyActor, IAction, IDummyTest, IDummyTest>;

    beforeEach(() => {
      mediator = new MediatorJoinReinforcementLearning({ name: 'mediator', bus });
    });

    beforeEach(() => {
      bus.subscribe(new DummyActor(10, bus));
      bus.subscribe(new DummyActor(100, bus));
      bus.subscribe(new DummyActor(1, bus));
    });

    it('should mediate', () => {
      return expect(mediator.mediate({})).resolves.toEqual({ todo: true });
    });
  });
  describe('Creating graph views', () =>{

    beforeEach(()=>{
      const subjectObjects: rdfjs.Term[][] = 
      [
        [ { termType: 'Variable', value: 'v0', equals: () =>{return true;} }, { termType: 'Variable', value: 'v1'  , equals: ()=>{return true}}],
        [ { termType: 'Variable', value: 'v0' }, { termType: 'Variable', value: 'v2' } ],
        [ { termType: 'Variable', value: 'v0' }, { termType: 'Variable', value: 'v3' } ],
        [ { termType: 'Variable', value: 'v0' }, { termType: 'Variable', value: 'v4' } ],
        [ { termType: 'Variable', value: 'v4' }, { termType: 'Variable', value: 'v5' } ],
        [ { termType: 'Variable', value: 'v4' }, { termType: 'Variable', value: 'v6' } ],
        [ { termType: 'Variable', value: 'v7' }, { termType: 'Variable', value: 'v6' } ],
        [ { termType: 'Variable', value: 'v7' }, { termType: 'Variable', value: 'v8' } ]
      ]
      MediatorJoinReinforcementLearning.createObjectObjectView(subjectObjects)
    });
    it('should create subject-subject view', () =>{

    })
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
