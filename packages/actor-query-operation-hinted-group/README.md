# Comunica Hinted Group Query Operation Actor

[![npm version](https://badge.fury.io/js/%40comunica%2Factor-query-operation-hinted-group.svg)](https://www.npmjs.com/package/@comunica/actor-query-operation-hinted-group)

A [Query Operation](https://github.com/comunica/comunica/tree/master/packages/bus-query-operation) actor
that handles `hinted-group` algebra operations. These operations are produced by the parser when the query hint
triple `comunica:hint comunica:optimizer "None" .` is present and nested `{ }` group graph patterns are used.

The actor executes sub-operations strictly in the order they appear in the input array,
preserving the query structure as authored and bypassing heuristic join reordering.
Nested `hinted-group` operations are supported for bushy execution plans.

This module is part of the [Comunica framework](https://github.com/comunica/comunica),
and should only be used by [developers that want to build their own query engine](https://comunica.dev/docs/modify/).

[Click here if you just want to query with Comunica](https://comunica.dev/docs/query/).

## Install

```bash
$ yarn add @comunica/actor-query-operation-hinted-group
```

## Configure

After installing, this package can be added to your engine's configuration as follows:
```text
{
  "@context": [
    ...
    "https://linkedsoftwaredependencies.org/bundles/npm/@comunica/actor-query-operation-hinted-group/^5.0.0/components/context.jsonld"
  ],
  "actors": [
    ...
    {
      "@id": "urn:comunica:default:query-operation/actors#hinted-group",
      "@type": "ActorQueryOperationHintedGroup",
      "mediatorQueryOperation": { "@id": "urn:comunica:default:query-operation/mediators#main" },
      "mediatorJoin": { "@id": "urn:comunica:default:rdf-join/mediators#main" }
    }
  ]
}
```
