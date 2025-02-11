import { ActorHttp } from '@comunica/bus-http';
import { KeysRdfUpdateQuads } from '@comunica/context-entries';
import { ActionContext } from '@comunica/core';
import type { IActionContext } from '@comunica/types';
import { stringify as stringifyStream } from '@jeswr/stream-to-string';
import { DataFactory } from 'rdf-data-factory';
import { Readable } from 'readable-stream';
import { QuadDestinationPutLdp } from '../lib/QuadDestinationPutLdp';

const DF = new DataFactory();

describe('QuadDestinationPutLdp', () => {
  let context: IActionContext;
  let mediaTypes: string[];
  let url: string;
  let mediatorHttp: any;
  let mediatorRdfSerializeMediatypes: any;
  let mediatorRdfSerialize: any;
  let destination: QuadDestinationPutLdp;

  beforeEach(() => {
    mediatorHttp = {
      mediate: jest.fn(async () => ({
        response: {
          status: 200,
        }
      })),
    };
    mediatorRdfSerializeMediatypes = {
      mediate: jest.fn(() => ({
        mediaTypes: {
          'text/turtle': 0.5,
          'application/ld+json': 0.7,
          'application/trig': 0.9,
        },
      })),
    };
    mediatorRdfSerialize = {
      mediate: jest.fn(() => ({
        handle: {
          data: Readable.from([ 'TRIPLES' ]),
        },
      })),
    };
    context = new ActionContext({ [KeysRdfUpdateQuads.destination.name]: 'abc' });
    mediaTypes = [ 'text/turtle', 'application/ld+json' ];
    url = 'abc';
    destination = new QuadDestinationPutLdp(
      url,
      context,
      mediaTypes,
      mediatorHttp,
      mediatorRdfSerializeMediatypes,
      mediatorRdfSerialize,
    );
  });

  describe('insert', () => {
    it('should handle a valid insert', async() => {
      await destination.update({ insert: <any> 'QUADS' });

      expect(mediatorRdfSerializeMediatypes.mediate).toHaveBeenCalledWith({
        context,
        mediaTypes: true,
      });

      expect(mediatorRdfSerialize.mediate).toHaveBeenCalledWith({
        context,
        handle: { context, quadStream: 'QUADS' },
        handleMediaType: 'text/turtle',
      });

      expect(mediatorHttp.mediate).toHaveBeenCalledWith({
        context,
        init: {
          headers: new Headers({ 'content-type': 'text/turtle' }),
          method: 'PUT',
          body: expect.anything(),
        },
        input: 'abc',
      });
      await expect(stringifyStream(ActorHttp.toNodeReadable(mediatorHttp.mediate.mock.calls[0][0].init.body))).resolves
        .toBe('TRIPLES');
    });

    it('should handle a valid insert when the server does not provide any suggestions', async() => {
      mediaTypes = [];
      destination = new QuadDestinationPutLdp(
        url,
        context,
        mediaTypes,
        mediatorHttp,
        mediatorRdfSerializeMediatypes,
        mediatorRdfSerialize,
      );

      await destination.update({ insert: <any> 'QUADS' });

      expect(mediatorRdfSerializeMediatypes.mediate).toHaveBeenCalledWith({
        context,
        mediaTypes: true,
      });

      expect(mediatorRdfSerialize.mediate).toHaveBeenCalledWith({
        context,
        handle: { context, quadStream: 'QUADS' },
        handleMediaType: 'application/trig',
      });

      expect(mediatorHttp.mediate).toHaveBeenCalledWith({
        context,
        init: {
          headers: new Headers({ 'content-type': 'application/trig' }),
          method: 'PUT',
          body: expect.anything(),
        },
        input: 'abc',
      });
      await expect(stringifyStream(ActorHttp.toNodeReadable(mediatorHttp.mediate.mock.calls[0][0].init.body))).resolves
        .toBe('TRIPLES');
    });

    it('should throw on a server error', async() => {
      mediatorHttp.mediate = async () => ({response: { status: 400 }});
      await expect(destination.update({ insert: <any> 'QUADS' })).rejects
        .toThrow('Could not update abc (HTTP status 400):\nempty response');
    });

    it('should close body if available', async() => {
      const cancel = jest.fn();
      mediatorHttp.mediate = async () => ( { response: {
        status: 200,
        body: { cancel },
      }});
      await destination.update({ insert: <any> 'QUADS' });
      expect(cancel).toHaveBeenCalledTimes(1);
    });
  });

  describe('delete', () => {
    it('should always throw', async() => {
      await expect(destination.update({ delete: <any> 'QUADS' }))
        .rejects.toThrow(`Put-based LDP destinations don't support deletions`);
    });
  });

  describe('update', () => {
    it('should always throw', async() => {
      await expect(destination.update({ delete: <any> 'QUADS', insert: <any> 'QUADS' }))
        .rejects.toThrow(`Put-based LDP destinations don't support deletions`);
    });
  });

  describe('deleteGraphs', () => {
    it('should always throw', async() => {
      await expect(destination.deleteGraphs('ALL', true, true))
        .rejects.toThrow(`Put-based LDP destinations don't support named graphs`);
    });
  });

  describe('createGraph', () => {
    it('should always throw', async() => {
      await expect(destination.createGraphs([ DF.namedNode('a') ], true))
        .rejects.toThrow(`Put-based LDP destinations don't support named graphs`);
    });
  });
});
