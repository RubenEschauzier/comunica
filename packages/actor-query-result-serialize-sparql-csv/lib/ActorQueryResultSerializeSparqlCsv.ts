import type {
  IActionSparqlSerialize,
  IActorQueryResultSerializeFixedMediaTypesArgs,
  IActorQueryResultSerializeOutput,
} from '@comunica/bus-query-result-serialize';
import {
  ActorQueryResultSerializeFixedMediaTypes,
} from '@comunica/bus-query-result-serialize';
import type { TestResult } from '@comunica/core';
import { failTest, passTestVoid } from '@comunica/core';
import type { Bindings, IActionContext, IQueryOperationResultBindings } from '@comunica/types';
import type * as RDF from '@rdfjs/types';
import { Readable } from 'readable-stream';

/**
 * A comunica SPARQL CSV Query Result Serialize Actor.
 */
export class ActorQueryResultSerializeSparqlCsv extends ActorQueryResultSerializeFixedMediaTypes {
  /**
   * @param args -
   *   \ @defaultNested {{
   *       "text/csv": 0.75
   *     }} mediaTypePriorities
   *   \ @defaultNested {{
   *       "text/csv": "http://www.w3.org/ns/formats/SPARQL_Results_CSV"
   *     }} mediaTypeFormats
   */
  public constructor(args: IActorQueryResultSerializeFixedMediaTypesArgs) {
    super(args);
  }

  /**
   * Converts an RDF term to its CSV representation.
   * @param {RDF.Term} value An RDF term.
   * @return {string} A string representation of the given value.
   */
  public static bindingToCsvBindings(value?: RDF.Term): string {
    if (!value) {
      return '';
    }

    let stringValue = value.value;

    if (value.termType === 'Literal') {
      // This is a lossy representation, since language and datatype are not encoded in here.
      stringValue = `${stringValue}`;
    } else if (value.termType === 'BlankNode') {
      stringValue = `_:${stringValue}`;
    } else if (value.termType === 'Quad') {
      let object = ActorQueryResultSerializeSparqlCsv.bindingToCsvBindings(value.object);
      if (value.object.termType === 'Literal') {
        // If object is a literal, it must be put in quotes, and internal quotes must be escaped
        object = `"${object.replaceAll('"', '""')}"`;
      }
      stringValue = `<< ${ActorQueryResultSerializeSparqlCsv.bindingToCsvBindings(value.subject)} ${ActorQueryResultSerializeSparqlCsv.bindingToCsvBindings(value.predicate)} ${object} >>`;
    } else {
      stringValue = `<${stringValue}>`;
    }

    // If a value contains certain characters, put it between double quotes
    if (/[",\n\r]/u.test(stringValue)) {
      // Within quote strings, " is written using a pair of quotation marks "".
      stringValue = `"${stringValue.replaceAll('"', '""')}"`;
    }

    return stringValue;
  }

  public override async testHandleChecked(
    action: IActionSparqlSerialize,
    _context: IActionContext,
  ): Promise<TestResult<boolean>> {
    if (action.type !== 'bindings') {
      return failTest('This actor can only handle bindings streams.');
    }
    return passTestVoid();
  }

  public async runHandle(action: IActionSparqlSerialize, _mediaType: string | undefined, _context: IActionContext):
  Promise<IActorQueryResultSerializeOutput> {
    const bindingsAction = <IQueryOperationResultBindings> action;

    const data = new Readable();

    const metadata = await bindingsAction.metadata();

    // Write head
    data.push(`${metadata.variables.map(variable => variable.variable.value).join(',')}\r\n`);

    // Write body
    data.wrap(<any> bindingsAction.bindingsStream.map((bindings: Bindings) => `${metadata.variables
        .map(key => ActorQueryResultSerializeSparqlCsv.bindingToCsvBindings(bindings.get(key.variable)))
        .join(',')}\r\n`));

    return { data };
  }
}
