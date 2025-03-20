import { KeysInitQuery } from '@comunica/context-entries';
import { ActionContext, Bus } from '@comunica/core';
import type { IActionContext, IQuerySource } from '@comunica/types';
import { BindingsFactory } from '@comunica/utils-bindings-factory';
import { DataFactory } from 'rdf-data-factory';
import { ActorRdfJoinMultiIndexSampling } from '../lib/ActorRdfJoinMultiIndexSampling';
import '@comunica/utils-jest';
import { QuerySourceRdfJs } from '@comunica/actor-query-source-identify-rdfjs';
import { RdfStore } from 'rdf-stores';

const DF = new DataFactory();
const BF = new BindingsFactory(DF);

describe('ActorRdfJoinMultiIndexSampling', () => {
    let bus: any;
    let context: IActionContext;

    describe('aggregateCountFn and aggregateSampleFn', () => {
        let sources: IQuerySource[];
        let store1: RdfStore;
        let store2: RdfStore;
        let store3: RdfStore;
    
        beforeEach(() => {
            bus = new Bus({ name: 'bus' });
            context = new ActionContext({ [KeysInitQuery.dataFactory.name]: DF });
            
            store1 = RdfStore.createDefault(true);
            store2 = RdfStore.createDefault(true);
            store3 = RdfStore.createDefault(true);

            store1.addQuad(DF.quad(DF.namedNode('s1'), DF.namedNode('p'), DF.namedNode('o1')));
            store1.addQuad(DF.quad(DF.namedNode('s2'), DF.namedNode('p'), DF.namedNode('o2')));
            store1.addQuad(DF.quad(DF.namedNode('s3'), DF.namedNode('px'), DF.namedNode('o3')));
      
            store2.addQuad(DF.quad(DF.namedNode('s4'), DF.namedNode('p'), DF.namedNode('o1')));
            store2.addQuad(DF.quad(DF.namedNode('s5'), DF.namedNode('p'), DF.namedNode('o2')));
            store2.addQuad(DF.quad(DF.namedNode('s6'), DF.namedNode('px'), DF.namedNode('o3')));

            sources = [
                new QuerySourceRdfJs(store1, DF, BF),
                new QuerySourceRdfJs(store2, DF, BF)
            ]
        });

        it('should correctly count the number of entries', async () => {
            expect(await ActorRdfJoinMultiIndexSampling.aggregateCountFn(
                sources, undefined, DF.namedNode('p'), undefined, undefined
            )).toEqual([2,2])
        });
        it('should correctly count the number of entries without matches', async () => {
            expect(await ActorRdfJoinMultiIndexSampling.aggregateCountFn(
                sources, undefined, DF.namedNode('pf'), undefined, undefined
            )).toEqual([0,0]);
        })

        it('should correctly sample a single index', async () => {
            const counts = await ActorRdfJoinMultiIndexSampling.aggregateCountFn(
                sources, undefined, DF.namedNode('p'), undefined, undefined
            );
            const indexes = [1];
            expect(await ActorRdfJoinMultiIndexSampling.aggregateSampleFn(
                sources, indexes, counts, undefined, DF.namedNode('p'), undefined, undefined
            )).toEqual([DF.quad(DF.namedNode('s2'), DF.namedNode('p'), DF.namedNode('o2'))])
        });
        
        it('should correctly sample a multiple index in same store', async () => {
            const indexes = [0,1];
            const counts = await ActorRdfJoinMultiIndexSampling.aggregateCountFn(
                sources, undefined, DF.namedNode('p'), undefined, undefined
            );
            expect(await ActorRdfJoinMultiIndexSampling.aggregateSampleFn(
                sources, indexes, counts, undefined, DF.namedNode('p'), undefined, undefined
            )).toEqual([
                DF.quad(DF.namedNode('s1'), DF.namedNode('p'), DF.namedNode('o1')),
                DF.quad(DF.namedNode('s2'), DF.namedNode('p'), DF.namedNode('o2')),
            ])
        });

        it('should correctly sample a multiple index in different stores', async () => {
            const indexes = [0,1,2];
            const counts = await ActorRdfJoinMultiIndexSampling.aggregateCountFn(
                sources, undefined, DF.namedNode('p'), undefined, undefined
            );
            expect(await ActorRdfJoinMultiIndexSampling.aggregateSampleFn(
                sources, indexes, counts, undefined, DF.namedNode('p'), undefined, undefined
            )).toEqual([
                DF.quad(DF.namedNode('s1'), DF.namedNode('p'), DF.namedNode('o1')),
                DF.quad(DF.namedNode('s2'), DF.namedNode('p'), DF.namedNode('o2')),
                DF.quad(DF.namedNode('s4'), DF.namedNode('p'), DF.namedNode('o1')),
            ])
        });

        it('should correctly sample when some stores have no matches', async () => {
            const sourcesWithZeroCount = [
                new QuerySourceRdfJs(store1, DF, BF),
                new QuerySourceRdfJs(store3, DF, BF),
                new QuerySourceRdfJs(store2, DF, BF)
            ];
            const counts = await ActorRdfJoinMultiIndexSampling.aggregateCountFn(
                sourcesWithZeroCount, undefined, DF.namedNode('p'), undefined, undefined
            );
            const indexes = [0,1,2];
            expect(await ActorRdfJoinMultiIndexSampling.aggregateSampleFn(
                sourcesWithZeroCount, indexes, counts, undefined, DF.namedNode('p'), undefined, undefined
            )).toEqual([
                DF.quad(DF.namedNode('s1'), DF.namedNode('p'), DF.namedNode('o1')),
                DF.quad(DF.namedNode('s2'), DF.namedNode('p'), DF.namedNode('o2')),
                DF.quad(DF.namedNode('s4'), DF.namedNode('p'), DF.namedNode('o1')),
            ])
        });

        it('should correctly sample second store', async () => {
            const counts = await ActorRdfJoinMultiIndexSampling.aggregateCountFn(
                sources, undefined, DF.namedNode('p'), undefined, undefined
            );
            const indexes = [3];
            expect(await ActorRdfJoinMultiIndexSampling.aggregateSampleFn(
                sources, indexes, counts, undefined, DF.namedNode('p'), undefined, undefined
            )).toEqual([
                DF.quad(DF.namedNode('s5'), DF.namedNode('p'), DF.namedNode('o2')),
            ])
        });
        
        it('should return empty on invalid index', async () => {
            const counts = await ActorRdfJoinMultiIndexSampling.aggregateCountFn(
                sources, undefined, DF.namedNode('p'), undefined, undefined
            );
            const indexes = [10];
            expect(await ActorRdfJoinMultiIndexSampling.aggregateSampleFn(
                sources, indexes, counts, undefined, DF.namedNode('p'), undefined, undefined
            )).toEqual([
            ])
        });
    })
});
