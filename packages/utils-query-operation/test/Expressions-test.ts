import { getOperatorTerms } from './../lib/Expressions';
import { translate } from 'sparqlalgebrajs';

describe('extractTermsFromOperation', () => {
  describe('Basic Graph Patterns', () => {
    it('should extract IRIs from a simple triple pattern', () => {
      const query = `
        PREFIX : <http://example.org/>
        SELECT * WHERE {
          :alice :knows :bob .
        }
      `;
      const algebra = translate(query);
      const terms = getOperatorTerms(algebra);
      
      expect(terms).toContain('subject:iri:http://example.org/alice');
      expect(terms).toContain('predicate:iri:http://example.org/knows');
      expect(terms).toContain('object:iri:http://example.org/bob');
    });

    it('should not extract variables', () => {
      const query = `
        PREFIX : <http://example.org/>
        SELECT * WHERE {
          ?person :knows :bob .
        }
      `;
      const algebra = translate(query);
      const terms = getOperatorTerms(algebra);
      
      expect(terms).not.toContain('subject:var:person');
      expect(terms).toContain('predicate:iri:http://example.org/knows');
      expect(terms).toContain('object:iri:http://example.org/bob');
    });

    it('should extract multiple constants from BGP', () => {
      const query = `
        PREFIX : <http://example.org/>
        SELECT * WHERE {
          :alice :knows :bob .
          :bob :likes :charlie .
        }
      `;
      const algebra = translate(query);
      const terms = getOperatorTerms(algebra);
      
      expect(terms).toContain('subject:iri:http://example.org/alice');
      expect(terms).toContain('subject:iri:http://example.org/bob');
      expect(terms).toContain('object:iri:http://example.org/bob');
      expect(terms).toContain('object:iri:http://example.org/charlie');
    });
  });

  describe('Literals', () => {
    it('should extract plain literals', () => {
      const query = `
        PREFIX : <http://example.org/>
        SELECT * WHERE {
          ?person :name "Alice" .
        }
      `;
      const algebra = translate(query);
      const terms = getOperatorTerms(algebra);
      
      expect(Array.from(terms).some(t => t.includes('literal:Alice'))).toBe(true);
    });

    it('should distinguish typed literals', () => {
      const query = `
        PREFIX : <http://example.org/>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        SELECT * WHERE {
          ?person :age "25"^^xsd:integer .
        }
      `;
      const algebra = translate(query);
      const terms = getOperatorTerms(algebra);
      
      expect(Array.from(terms).some(t => 
        t.includes('literal:25') && t.includes('XMLSchema#integer')
      )).toBe(true);
    });

    it('should distinguish language-tagged literals', () => {
      const query = `
        PREFIX : <http://example.org/>
        SELECT * WHERE {
          ?person :name "Alice"@en .
        }
      `;
      const algebra = translate(query);
      const terms = getOperatorTerms(algebra);
      
      expect(Array.from(terms).some(t => 
        t.includes('literal:Alice') && t.includes('@en')
      )).toBe(true);
    });
  });

  describe('Position distinction', () => {
    it('should distinguish same IRI in different positions', () => {
      const query = `
        PREFIX : <http://example.org/>
        SELECT * WHERE {
          :Person :type :Person .
        }
      `;
      const algebra = translate(query);
      const terms = getOperatorTerms(algebra);
      
      expect(terms).toContain('subject:iri:http://example.org/Person');
      expect(terms).toContain('predicate:iri:http://example.org/type');
      expect(terms).toContain('object:iri:http://example.org/Person');
      expect(terms.size).toBeGreaterThanOrEqual(3);
    });
  });

  describe('GRAPH operation', () => {
    it('should extract named graph IRI', () => {
      const query = `
        PREFIX : <http://example.org/>
        SELECT * WHERE {
          GRAPH :metadata {
            ?person :age ?age .
          }
        }
      `;
      const algebra = translate(query);
      const terms = getOperatorTerms(algebra);
      
      expect(terms).toContain('graph:iri:http://example.org/metadata');
      expect(terms).toContain('predicate:iri:http://example.org/age');
    });

    it('should not extract graph variable', () => {
      const query = `
        PREFIX : <http://example.org/>
        SELECT * WHERE {
          GRAPH ?g {
            ?person :age ?age .
          }
        }
      `;
      const algebra = translate(query);
      const terms = getOperatorTerms(algebra);
      
      expect(Array.from(terms).every(t => !t.includes('var:g'))).toBe(true);
      expect(terms).toContain('predicate:iri:http://example.org/age');
    });

    it('should extract constants from inside GRAPH', () => {
      const query = `
        PREFIX : <http://example.org/>
        SELECT * WHERE {
          GRAPH :g1 {
            :alice :knows :bob .
          }
        }
      `;
      const algebra = translate(query);
      const terms = getOperatorTerms(algebra);
      
      expect(terms).toContain('graph:iri:http://example.org/g1');
      expect(terms).toContain('subject:iri:http://example.org/alice');
      expect(terms).toContain('object:iri:http://example.org/bob');
    });
  });

  describe('UNION operation', () => {
    it('should extract constants from both sides of UNION and a BGP', () => {
      const query = `
        PREFIX : <http://example.org/>
        SELECT * WHERE {
          ?v1 :knows :bob .
          ?v1 :posts ?message .
          {
            ?message :type :Post .
          }
          UNION
          {
            ?message :type :Comment .
          }
        }
      `;
      const algebra = translate(query);
      const terms = getOperatorTerms(algebra);
        
      expect(terms).toContain('object:iri:http://example.org/bob');
      expect(terms).toContain('predicate:iri:http://example.org/knows');
      expect(terms).toContain('predicate:iri:http://example.org/posts');
      expect(terms).toContain('predicate:iri:http://example.org/type');
      expect(terms).toContain('object:iri:http://example.org/Post');
      expect(terms).toContain('object:iri:http://example.org/Comment');
    });
  });

  describe('FILTER with constants', () => {
    it('should extract constants from FILTER expression', () => {
      const query = `
        PREFIX : <http://example.org/>
        SELECT * WHERE {
          ?person :name ?name .
          FILTER(?name != "Bob")
        }
      `;
      const algebra = translate(query);
      const terms = getOperatorTerms(algebra);
      
      expect(terms).toContain('predicate:iri:http://example.org/name');
      expect(Array.from(terms).some(t => t.includes('literal:Bob'))).toBe(true);
    });

    it('should extract constants from EXISTS subquery', () => {
      const query = `
        PREFIX : <http://example.org/>
        SELECT * WHERE {
          ?person :name ?name .
          FILTER EXISTS {
            ?person :livesIn :Brussels .
          }
        }
      `;
      const algebra = translate(query);
      const terms = getOperatorTerms(algebra);
      
      expect(terms).toContain('predicate:iri:http://example.org/livesIn');
      expect(terms).toContain('object:iri:http://example.org/Brussels');
    });
  });

  describe('BIND operation', () => {
    it('should extract constants from BIND expression', () => {
      const query = `
        PREFIX : <http://example.org/>
        SELECT * WHERE {
          ?person :name ?name .
          BIND(:DefaultCity AS ?city)
        }
      `;
      const algebra = translate(query);
      const terms = getOperatorTerms(algebra);
      
      expect(terms).toContain('predicate:iri:http://example.org/name');
      expect(Array.from(terms).some(t => t.includes('DefaultCity'))).toBe(true);
    });
  });

  describe('VALUES operation', () => {
    it('should extract constants from VALUES clause', () => {
      const query = `
        PREFIX : <http://example.org/>
        SELECT * WHERE {
          VALUES (?person ?city) {
            (:alice :Brussels)
            (:bob :Paris)
          }
          ?person :livesIn ?city .
        }
      `;
      const algebra = translate(query);
      const terms = getOperatorTerms(algebra);
      
      expect(Array.from(terms).some(t => t.includes('alice'))).toBe(true);
      expect(Array.from(terms).some(t => t.includes('Brussels'))).toBe(true);
      expect(Array.from(terms).some(t => t.includes('bob'))).toBe(true);
      expect(Array.from(terms).some(t => t.includes('Paris'))).toBe(true);
    });
  });

  describe('SERVICE operation', () => {
    it('should extract service endpoint IRI', () => {
      const query = `
        PREFIX : <http://example.org/>
        SELECT * WHERE {
          SERVICE <http://dbpedia.org/sparql> {
            ?person :name ?name .
          }
        }
      `;
      const algebra = translate(query);
      const terms = getOperatorTerms(algebra);
      
      expect(terms).toContain('service:iri:http://dbpedia.org/sparql');
    });

    it('should not extract service variable', () => {
      const query = `
        PREFIX : <http://example.org/>
        SELECT * WHERE {
          SERVICE ?endpoint {
            ?person :name ?name .
          }
        }
      `;
      const algebra = translate(query);
      const terms = getOperatorTerms(algebra);
      
      expect(Array.from(terms).every(t => !t.includes('var:endpoint'))).toBe(true);
    });
  });

  describe('Property paths', () => {
    it('should extract IRIs from property paths', () => {
      const query = `
        PREFIX : <http://example.org/>
        SELECT * WHERE {
          :alice :knows+ :bob .
        }
      `;
      const algebra = translate(query);
      const terms = getOperatorTerms(algebra);
      console.log(terms)
      
      expect(terms).toContain('subject:iri:http://example.org/alice');
      expect(terms).toContain('object:iri:http://example.org/bob');
      expect(terms).toContain('predicate:iri:http://example.org/knows');
    });
  });

  describe('Individual Operations', () => {
    it('should extract all terms from a BGP', () => {
      const query = `
        PREFIX : <http://example.org/>
        SELECT * WHERE {
          :alice :knows :bob .
          :alice :messages ?v1 .
          :bob :messages ?v2
          {
            ?message :type :Post .
          }
          UNION
          {
            ?message :type :Comment .
          }
        }
      `;
      const algebra = translate(query);
      console.log(algebra.input)

    });
  })
  describe('Edge cases', () => {
    it('should return empty set for query with only variables', () => {
      const query = `
        SELECT * WHERE {
          ?s ?p ?o .
        }
      `;
      const algebra = translate(query);
      const terms = getOperatorTerms(algebra);
      
      expect(terms.size).toBe(0);
    });

    it('should handle blank nodes', () => {
      const query = `
        PREFIX : <http://example.org/>
        SELECT * WHERE {
          _:b1 :knows ?person .

        }
      `;
      const algebra = translate(query);
      const terms = getOperatorTerms(algebra);
      
      expect(Array.from(terms).some(t => t.includes('bnode:'))).toBe(true);
    });
  });
});