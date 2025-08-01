import type { Readable } from 'node:stream';
import { ActorRdfMetadata } from '@comunica/bus-rdf-metadata';
import { ActionContext, Bus } from '@comunica/core';
import type { IActionContext } from '@comunica/types';
import type * as RDF from '@rdfjs/types';
import arrayifyStream from 'arrayify-stream';
import { streamifyArray } from 'streamify-array';
import { ActorRdfMetadataAll } from '../lib/ActorRdfMetadataAll';
import '@comunica/utils-jest';

const quad = require('rdf-quad');

describe('ActorRdfMetadataAll', () => {
  let bus: any;
  let context: IActionContext;

  beforeEach(() => {
    bus = new Bus({ name: 'bus' });
    context = new ActionContext();
  });

  describe('The ActorRdfMetadataAll module', () => {
    it('should be a function', () => {
      expect(ActorRdfMetadataAll).toBeInstanceOf(Function);
    });

    it('should be a ActorRdfMetadataAll constructor', () => {
      expect(new (<any> ActorRdfMetadataAll)({ name: 'actor', bus }))
        .toBeInstanceOf(ActorRdfMetadataAll);
      expect(new (<any> ActorRdfMetadataAll)({ name: 'actor', bus }))
        .toBeInstanceOf(ActorRdfMetadata);
    });

    it('should not be able to create new ActorRdfMetadataAll objects without \'new\'', () => {
      expect(() => {
        (<any> ActorRdfMetadataAll)();
      }).toThrow(`Class constructor ActorRdfMetadataAll cannot be invoked without 'new'`);
    });
  });

  describe('An ActorRdfMetadataAll instance', () => {
    let actor: ActorRdfMetadataAll;
    let input: Readable;
    let inputDifferent: Readable;

    beforeEach(() => {
      actor = new ActorRdfMetadataAll({ name: 'actor', bus });
      input = streamifyArray([
        quad('s1', 'p1', 'o1', ''),
        quad('o1', 'http://rdfs.org/ns/void#subset', 'o1?param', 'g1'),
        quad('g1', 'http://xmlns.com/foaf/0.1/primaryTopic', 'o1', 'g1'),
        quad('s2', 'p2', 'o2', 'g1'),
        quad('s3', 'p3', 'o3', ''),
      ]);
      inputDifferent = streamifyArray([
        quad('s1', 'p1', 'o1', ''),
        quad('g1', 'http://xmlns.com/foaf/0.1/primaryTopic', 'o2', 'g1'),
        quad('o2', 'http://rdfs.org/ns/void#subset', 'o2?param', 'g1'),
        quad('s2', 'p2', 'o2', 'g1'),
        quad('s3', 'p3', 'o3', ''),
      ]);
    });

    it('should test on a triple stream', async() => {
      await expect(actor.test({ url: '', quads: input, triples: true, context })).resolves.toPassTestVoid();
    });

    it('should test on a quad stream', async() => {
      await expect(actor.test({ url: '', quads: input, context })).resolves.toPassTestVoid();
    });

    it('should run', async() => {
      await actor.run({ url: 'o1?param', quads: input, context })
        .then(async(output) => {
          const data: RDF.Quad[] = await arrayifyStream(output.data);
          const metadata: RDF.Quad[] = await arrayifyStream(output.metadata);
          expect(data).toEqual([
            quad('s1', 'p1', 'o1', ''),
            quad('o1', 'http://rdfs.org/ns/void#subset', 'o1?param', 'g1'),
            quad('g1', 'http://xmlns.com/foaf/0.1/primaryTopic', 'o1', 'g1'),
            quad('s2', 'p2', 'o2', 'g1'),
            quad('s3', 'p3', 'o3', ''),
          ]);
          expect(metadata).toEqual([
            quad('s1', 'p1', 'o1', ''),
            quad('o1', 'http://rdfs.org/ns/void#subset', 'o1?param', 'g1'),
            quad('g1', 'http://xmlns.com/foaf/0.1/primaryTopic', 'o1', 'g1'),
            quad('s2', 'p2', 'o2', 'g1'),
            quad('s3', 'p3', 'o3', ''),
          ]);
        });
    });

    it('should run and delegate errors', async() => {
      await actor.run({ url: '', quads: input, context })
        .then((output) => {
          setImmediate(() => input.emit('error', new Error('RDF Meta Primary Topic error')));
          output.data.on('data', () => {
            // Do nothing
          });
          return Promise.all([ new Promise((resolve) => {
            output.data.on('error', resolve);
          }), new Promise((resolve) => {
            output.metadata.on('error', resolve);
          }) ]).then((errors) => {
            expect(errors).toHaveLength(2);
          });
        });
    });

    it('should run and delegate errors without data listener being attached', async() => {
      await actor.run({ url: '', quads: input, context })
        .then((output) => {
          setImmediate(() => input.emit('error', new Error('RDF Meta Primary Topic error')));
          return Promise.all([ new Promise((resolve) => {
            output.data.on('error', resolve);
          }), new Promise((resolve) => {
            output.metadata.on('error', resolve);
          }) ]).then((errors) => {
            expect(errors).toHaveLength(2);
          });
        });
    });

    it('should run and not re-attach listeners after calling .read again', async() => {
      await actor.run({ url: 'o1?param', quads: inputDifferent, context })
        .then(async(output) => {
          const data: RDF.Quad[] = await arrayifyStream(output.data);
          expect(data).toEqual([
            quad('s1', 'p1', 'o1', ''),
            quad('g1', 'http://xmlns.com/foaf/0.1/primaryTopic', 'o2', 'g1'),
            quad('o2', 'http://rdfs.org/ns/void#subset', 'o2?param', 'g1'),
            quad('s2', 'p2', 'o2', 'g1'),
            quad('s3', 'p3', 'o3', ''),
          ]);
          expect((<any> output.data)._read()).toBeFalsy();
        });
    });
  });
});
