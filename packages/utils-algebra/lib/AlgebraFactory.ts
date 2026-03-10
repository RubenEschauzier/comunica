import type * as RDF from '@rdfjs/types';
import { AlgebraFactory as AlgebraFactoryBase } from '@traqula/algebra-transformations-1-2';
import type { QuadTermName } from 'rdf-terms';
import type { DistinctTerms, HintedGroup, Nodes, Operation } from './Algebra';
import { TypesComunica } from './TypesComunica';

export class AlgebraFactory extends AlgebraFactoryBase {
  public createNodes(graph: RDF.Term, variable: RDF.Variable): Nodes {
    return {
      type: TypesComunica.NODES,
      graph,
      variable,
    };
  }

  public createDistinctTerms(
    variables: RDF.Variable[],
    terms: Record<string, QuadTermName>,
  ): DistinctTerms {
    return {
      type: TypesComunica.DISTINCT_TERMS,
      variables,
      terms,
    };
  }

  public createHintedGroup(input: Operation[]): HintedGroup {
    return {
      type: TypesComunica.HINTED_GROUP,
      input,
    };
  }
}
