import type * as RDF from '@rdfjs/types';
import { termToString } from 'rdf-string';
import type { Bindings } from './Bindings';

/**
 * Stringify a bindings object.
 * @param bindings A bindings object.
 */
export function bindingsToString(bindings: RDF.Bindings): string {
  const raw: Record<string, string> = {};
  for (const key of bindings.keys()) {
    raw[key.value] = termToString(bindings.get(key))!;
  }
  const bindingsWithContext = <Bindings> bindings;
  raw.context = JSON.stringify(bindingsWithContext.context);
  return JSON.stringify(raw, null, '  ');
}
