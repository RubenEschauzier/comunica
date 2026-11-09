import type { MediatorContextPreprocess } from '@comunica/bus-context-preprocess';
import type { MediatorOptimizeQueryOperation } from '@comunica/bus-optimize-query-operation';
import type { MediatorQueryOperation } from '@comunica/bus-query-operation';
import type { MediatorQueryParse } from '@comunica/bus-query-parse';
import { KeysInitQuery, KeysStatistics } from '@comunica/context-entries';
import { ActionContext, Bus } from '@comunica/core';
import type { IActionContext } from '@comunica/types';
import { AlgebraFactory } from '@comunica/utils-algebra';
import type { Algebra } from '@comunica/utils-algebra';
import { BindingsFactory } from '@comunica/utils-bindings-factory';
import { DataFactory } from 'rdf-data-factory';
import { ActorQueryProcessSequential } from '../lib/ActorQueryProcessSequential';
import '@comunica/utils-jest';

const DF = new DataFactory();
const AF = new AlgebraFactory();
const BF = new BindingsFactory(DF);

describe('ActorQueryProcessSequential', () => {
  let bus: any;
  let mediatorContextPreprocess: MediatorContextPreprocess;
  let mediatorQueryParse: MediatorQueryParse;
  let mediatorOptimizeQueryOperation: MediatorOptimizeQueryOperation;
  let mediatorQueryOperation: MediatorQueryOperation;
  let mediatorMergeBindingsContext: any;

  beforeEach(() => {
    bus = new Bus({ name: 'bus' });
    mediatorContextPreprocess = <any> {
      mediate: jest.fn((action: any) => Promise.resolve(action)),
    };
    mediatorQueryParse = <any> {
      mediate: jest.fn((action: any) => Promise.resolve({
        operation: 'PARSED',
        context: action.context,
      })),
    };
    mediatorOptimizeQueryOperation = <any> {
      mediate: jest.fn((action: any) => Promise.resolve({
        operation: AF.createJoin([
          action.operation,
        ], false),
        context: action.context,
      })),
    };
    mediatorQueryOperation = <any> {
      mediate: jest.fn((action: any) => Promise.resolve({
        type: 'bindings',
        ...action,
      })),
    };
    mediatorMergeBindingsContext = {
      mediate: () => ({}),
    };
  });

  describe('An ActorQueryProcessSequential instance', () => {
    let actor: ActorQueryProcessSequential;
    let ctx: IActionContext;

    beforeEach(() => {
      ctx = new ActionContext({ [KeysInitQuery.dataFactory.name]: DF });
      actor = new ActorQueryProcessSequential({
        name: 'actor',
        bus,
        mediatorContextPreprocess,
        mediatorQueryParse,
        mediatorOptimizeQueryOperation,
        mediatorQueryOperation,
        mediatorMergeBindingsContext,
      });
    });

    describe('test', () => {
      it('rejects on explain in context', async() => {
        await expect(actor.test({
          query: 'q',
          context: new ActionContext({ [KeysInitQuery.dataFactory.name]: DF }).set(KeysInitQuery.explain, 'parsed'),
        }))
          .resolves.toFailTest(`actor is not able to explain queries.`);
      });

      it('rejects on raw explain in context', async() => {
        await expect(actor.test({
          query: 'q',
          context: new ActionContext({ [KeysInitQuery.dataFactory.name]: DF }).setRaw('explain', 'parsed'),
        }))
          .resolves.toFailTest(`actor is not able to explain queries.`);
      });

      it('handles no explain in context', async() => {
        await expect(actor.test({
          query: 'q',
          context: new ActionContext({ [KeysInitQuery.dataFactory.name]: DF }),
        })).resolves
          .toPassTestVoid();
      });
    });

    describe('run', () => {
      it('parses, optimizes, and evaluates a query', async() => {
        const op: Algebra.Operation = AF.createPattern(
          DF.namedNode('s'),
          DF.namedNode('p'),
          DF.namedNode('o'),
        );
        const { result } = await actor.run({
          query: op,
          context: ctx,
        });
        expect((<any> result).context).toEqual(new ActionContext({ [KeysInitQuery.dataFactory.name]: DF })
          .set(KeysInitQuery.queryExecutionScope, {})
          .set(KeysInitQuery.query, AF.createJoin([
            op,
          ], false)));
        expect((<any> result).operation).toEqual(AF.createJoin([
          op,
        ], false));

        expect(mediatorContextPreprocess.mediate).toHaveBeenCalledWith({
          context: new ActionContext({ [KeysInitQuery.dataFactory.name]: DF }),
          initialize: true,
        });
        expect(mediatorQueryParse.mediate).not.toHaveBeenCalled();
        expect(mediatorOptimizeQueryOperation.mediate).toHaveBeenCalledWith({
          operation: op,
          context: new ActionContext({ [KeysInitQuery.dataFactory.name]: DF })
            .set(KeysInitQuery.queryExecutionScope, {})
            .set(KeysInitQuery.query, op),
        });
        expect(mediatorQueryOperation.mediate).toHaveBeenCalledWith({
          operation: AF.createJoin([
            op,
          ], false),
          context: new ActionContext({ [KeysInitQuery.dataFactory.name]: DF })
            .set(KeysInitQuery.queryExecutionScope, {})
            .set(KeysInitQuery.query, AF.createJoin([
              op,
            ], false)),
        });
      });
    });

    describe('parse', () => {
      it('parses a query', async() => {
        const output = await actor.parse('query', ctx);
        expect(output.context).toEqual(new ActionContext({ [KeysInitQuery.dataFactory.name]: DF })
          .set(KeysInitQuery.queryString, 'query'));
        expect(output.operation).toBe('PARSED');

        expect(mediatorContextPreprocess.mediate).toHaveBeenCalledWith({
          context: new ActionContext({ [KeysInitQuery.dataFactory.name]: DF }),
          initialize: true,
        });
        expect(mediatorQueryParse.mediate).toHaveBeenCalledWith({
          context: new ActionContext({ [KeysInitQuery.dataFactory.name]: DF })
            .set(KeysInitQuery.queryString, 'query'),
          query: 'query',
          baseIRI: undefined,
        });
      });

      it('parses a query with query format', async() => {
        const output = await actor.parse('query', ctx);
        expect(output.context).toEqual(new ActionContext({ [KeysInitQuery.dataFactory.name]: DF })
          .set(KeysInitQuery.queryString, 'query'));
        expect(output.operation).toBe('PARSED');

        expect(mediatorContextPreprocess.mediate).toHaveBeenCalledWith({
          context: new ActionContext({ [KeysInitQuery.dataFactory.name]: DF }),
          initialize: true,
        });
        expect(mediatorQueryParse.mediate).toHaveBeenCalledWith({
          context: new ActionContext({ [KeysInitQuery.dataFactory.name]: DF })
            .set(KeysInitQuery.queryString, 'query'),
          query: 'query',
          baseIRI: undefined,
        });
      });

      it('parses a query with baseIRI in query', async() => {
        jest.spyOn((<any> mediatorQueryParse), 'mediate').mockImplementation((action: any) => Promise.resolve({
          operation: 'PARSED',
          context: action.context,
          baseIRI: 'BASE',
        }));

        const output = await actor.parse('query', ctx);
        expect(output.context).toEqual(new ActionContext({ [KeysInitQuery.dataFactory.name]: DF })
          .set(KeysInitQuery.queryString, 'query')
          .set(KeysInitQuery.baseIRI, 'BASE'));
        expect(output.operation).toBe('PARSED');

        expect(mediatorContextPreprocess.mediate).toHaveBeenCalledWith({
          context: new ActionContext({ [KeysInitQuery.dataFactory.name]: DF }),
          initialize: true,
        });
        expect(mediatorQueryParse.mediate).toHaveBeenCalledWith({
          context: new ActionContext({ [KeysInitQuery.dataFactory.name]: DF })
            .set(KeysInitQuery.queryString, 'query'),
          query: 'query',
          baseIRI: undefined,
        });
      });

      it('parses a query as algebra', async() => {
        const op: Algebra.Operation = AF.createPattern(
          DF.namedNode('s'),
          DF.namedNode('p'),
          DF.namedNode('o'),
        );
        const output = await actor.parse(op, ctx);
        expect(output.context).toEqual(new ActionContext({ [KeysInitQuery.dataFactory.name]: DF }));
        expect(output.operation).toEqual(op);

        expect(mediatorContextPreprocess.mediate).toHaveBeenCalledWith({
          context: new ActionContext({ [KeysInitQuery.dataFactory.name]: DF }),
          initialize: true,
        });
        expect(mediatorQueryParse.mediate).not.toHaveBeenCalled();
      });

      it('parses handles initial bindings', async() => {
        const op: Algebra.Operation = AF.createPattern(
          DF.namedNode('s'),
          DF.namedNode('p'),
          DF.variable('vo'),
        );
        const output = await actor.parse(op, ctx
          .set(KeysInitQuery.initialBindings, BF.fromRecord({ vo: DF.namedNode('o') })));
        expect(output.context).toEqual(new ActionContext({ [KeysInitQuery.dataFactory.name]: DF })
          .set(KeysInitQuery.initialBindings, BF.fromRecord({ vo: DF.namedNode('o') })));
        expect(output.operation).toEqual(AF.createPattern(
          DF.namedNode('s'),
          DF.namedNode('p'),
          DF.namedNode('o'),
        ));

        expect(mediatorContextPreprocess.mediate).toHaveBeenCalledWith({
          context: new ActionContext({ [KeysInitQuery.dataFactory.name]: DF })
            .set(KeysInitQuery.initialBindings, BF.fromRecord({ vo: DF.namedNode('o') })),
          initialize: true,
        });
        expect(mediatorQueryParse.mediate).not.toHaveBeenCalled();
      });
    });

    describe('optimize', () => {
      it('optimizes a query', async() => {
        const op: Algebra.Operation = AF.createPattern(
          DF.namedNode('s'),
          DF.namedNode('p'),
          DF.namedNode('o'),
        );

        const output = await actor.optimize(op, ctx);
        expect(output.context).toEqual(new ActionContext({ [KeysInitQuery.dataFactory.name]: DF })
          .set(KeysInitQuery.queryExecutionScope, {})
          .set(KeysInitQuery.query, AF.createJoin([
            op,
          ], false)));
        expect(output.operation).toEqual(AF.createJoin([
          op,
        ], false));

        expect(mediatorOptimizeQueryOperation.mediate).toHaveBeenCalledWith({
          operation: op,
          context: new ActionContext({ [KeysInitQuery.dataFactory.name]: DF })
            .set(KeysInitQuery.queryExecutionScope, {})
            .set(KeysInitQuery.query, op),
        });
      });

      it('keeps the query string where the optimizers left it, and the original one for statistics', async() => {
        const op: Algebra.Operation = AF.createPattern(DF.namedNode('s'), DF.namedNode('p'), DF.namedNode('o'));
        const queryCtx = ctx.set(KeysInitQuery.queryString, 'SELECT * WHERE { <s> <p> <o> }');

        // Kept by the optimizers, as when a single source takes the whole query
        let output = await actor.optimize(op, queryCtx);
        expect(output.context.get(KeysInitQuery.queryString)).toBe('SELECT * WHERE { <s> <p> <o> }');
        expect(output.context.get(KeysStatistics.originalQueryString)).toBe('SELECT * WHERE { <s> <p> <o> }');

        // Deleted by the optimizers, as when the query is spread over several sources, which must then each be
        // sent their own part of it rather than the query string
        jest.spyOn((<any> mediatorOptimizeQueryOperation), 'mediate').mockImplementation((action: any) =>
          Promise.resolve({ operation: action.operation, context: action.context.delete(KeysInitQuery.queryString) }));
        output = await actor.optimize(op, queryCtx);
        expect(output.context.has(KeysInitQuery.queryString)).toBe(false);
        expect(output.context.get(KeysStatistics.originalQueryString)).toBe('SELECT * WHERE { <s> <p> <o> }');
      });
    });

    describe('evaluate', () => {
      it('evaluates a query', async() => {
        const op: Algebra.Operation = AF.createPattern(
          DF.namedNode('s'),
          DF.namedNode('p'),
          DF.namedNode('o'),
        );

        const output = await actor.evaluate(op, ctx);
        expect(output).toEqual({
          type: 'bindings',
          operation: op,
          context: ctx,
        });

        expect(mediatorQueryOperation.mediate).toHaveBeenCalledWith({
          operation: op,
          context: new ActionContext({ [KeysInitQuery.dataFactory.name]: DF }),
        });
      });
    });
  });
});
