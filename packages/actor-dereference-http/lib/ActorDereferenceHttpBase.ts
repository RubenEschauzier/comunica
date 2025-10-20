import type { IActionDereference, IActorDereferenceArgs, IActorDereferenceOutput } from '@comunica/bus-dereference';
import { ActorDereference, emptyReadable } from '@comunica/bus-dereference';
import type { IActorHttpOutput, MediatorHttp } from '@comunica/bus-http';
import { ActorHttp } from '@comunica/bus-http';
import type { IActorTest, TestResult } from '@comunica/core';
import { failTest, passTestVoid } from '@comunica/core';
import { stringify as stringifyStream } from '@jeswr/stream-to-string';
import { resolve as resolveRelative } from 'relative-to-absolute-iri';

const REGEX_MEDIATYPE = /^[^ ;]*/u;

export function mediaTypesToAcceptString(mediaTypes: Record<string, number>, maxLength: number): string {
  const wildcard = '*/*;q=0.1';
  const parts: string[] = [];
  const sortedMediaTypes = Object.entries(mediaTypes)
    .map(([ mediaType, priority ]) => ({ mediaType, priority }))
    .sort((left, right) => right.priority === left.priority ?
      left.mediaType.localeCompare(right.mediaType) :
      right.priority - left.priority);
  // Take into account the ',' characters joining each type
  let partsLength = sortedMediaTypes.length - 1;
  for (const { mediaType, priority } of sortedMediaTypes) {
    const part = mediaType + (priority === 1 ? '' : `;q=${priority.toFixed(3).replace(/0*$/u, '')}`);
    if (partsLength + part.length > maxLength) {
      while (partsLength + wildcard.length > maxLength) {
        const last = parts.pop() ?? '';
        // Don't forget the ','
        partsLength -= last.length + 1;
      }
      parts.push(wildcard);
      break;
    }
    parts.push(part);
    partsLength += part.length;
  }
  return parts.length === 0 ? '*/*' : parts.join(',');
}

/**
 * An actor that listens on the 'dereference' bus.
 *
 * It resolves the URL using the HTTP bus using an accept header compiled from the available media types.
 */
export abstract class ActorDereferenceHttpBase extends ActorDereference implements IActorDereferenceHttpArgs {
  public readonly mediatorHttp: MediatorHttp;
  public readonly maxAcceptHeaderLength: number;
  public readonly maxAcceptHeaderLengthBrowser: number;

  public constructor(args: IActorDereferenceHttpArgs) {
    super(args);
  }

  public async test({ url }: IActionDereference): Promise<TestResult<IActorTest>> {
    if (!/^https?:/u.test(url)) {
      return failTest(`Cannot retrieve ${url} because it is not an HTTP(S) URL.`);
    }
    return passTestVoid();
  }

  public async run(action: IActionDereference): Promise<IActorDereferenceOutput> {
    let exists = true;

    // Append any custom passed headers
    const headers = new Headers(action.headers);

    // Resolve HTTP URL using appropriate accept header
    headers.append(
      'Accept',
      mediaTypesToAcceptString(await action.mediaTypes?.() ?? {}, this.getMaxAcceptHeaderLength()),
    );

    let httpResponse: IActorHttpOutput;
    const requestTimeStart = Date.now();
    try {
      httpResponse = await this.mediatorHttp.mediate({
        context: action.context,
        init: { headers, method: action.method },
        input: action.url,
        validate: action.validate,
      });
      if (!httpResponse.response && action.validate) {
        return {
          url: action.url,
          data: emptyReadable(),
          exists,
          requestTime: 0,
          validationOutput: httpResponse.validationOutput,
          mediaType: 'text/turtle',
        };
      }
      if (!httpResponse.response) {
        return this.handleDereferenceErrors(action, new Error('Response undefined in dereference actor'));
      }
    } catch (error: unknown) {
      return this.handleDereferenceErrors(action, error);
    }
    // The response URL can be relative to the given URL
    const url = resolveRelative(httpResponse.response.url, action.url);
    const requestTime = Date.now() - requestTimeStart;

    // Only parse if retrieval was successful
    if (httpResponse.response.status !== 200) {
      exists = false;
      // Consume the body, to avoid process to hang
      const bodyString = httpResponse.response.body ?
        await stringifyStream(ActorHttp.toNodeReadable(httpResponse.response.body)) :
        'empty response';

      if (!action.acceptErrors) {
        const error = new Error(`Could not retrieve ${action.url} (HTTP status ${httpResponse.response.status}):\n${bodyString}`);
        return this.handleDereferenceErrors(action, error, httpResponse.response.headers, requestTime);
      }
    }

    const mediaType = REGEX_MEDIATYPE.exec(httpResponse.response.headers.get('content-type') ?? '')?.[0];

    // Return the parsed quad stream and whether or not only triples are supported
    return {
      url,
      data: exists ? ActorHttp.toNodeReadable(httpResponse.response.body) : emptyReadable(),
      exists,
      requestTime,
      headers: httpResponse.response.headers,
      mediaType: mediaType === 'text/plain' ? undefined : mediaType,
    };
  }

  protected abstract getMaxAcceptHeaderLength(): number;
}

export interface IActorDereferenceHttpArgs extends IActorDereferenceArgs {
  /**
   * The HTTP mediator.
   */
  mediatorHttp: MediatorHttp;
  /**
   * The maximum allowed accept header value length for non-browser environments.
   * @range {integer}
   * @default {1024}
   */
  maxAcceptHeaderLength: number;
  /**
   * The maximum allowed accept header value length for browser environments.
   * @range {integer}
   * @default {128}
   */
  maxAcceptHeaderLengthBrowser: number;
}
