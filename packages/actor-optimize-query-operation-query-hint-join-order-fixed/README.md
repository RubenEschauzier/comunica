# Comunica Query Hint Join Order Fixed Optimize Query Operation Actor

[![npm version](https://badge.fury.io/js/%40comunica%2Factor-optimize-query-operation-query-hint-join-order-fixed.svg)](https://www.npmjs.com/package/@comunica/actor-optimize-query-operation-query-hint-join-order-fixed)

An [Optimize Query Operation](https://github.com/comunica/comunica/tree/master/packages/bus-optimize-query-operation) actor
that detects the query hint triple `comunica:hint comunica:optimizer "None" .` (where `comunica:` resolves to `https://comunica.dev/`),
removes it from the algebra, and sets a context entry to indicate that join order should not be optimized.

This module is part of the [Comunica framework](https://github.com/comunica/comunica),
and should only be used by [developers that want to build their own query engine](https://comunica.dev/docs/modify/).

[Click here if you just want to query with Comunica](https://comunica.dev/docs/query/).

## Install

```bash
$ yarn add @comunica/actor-optimize-query-operation-query-hint-join-order-fixed
```

## Configure

After installing, this package can be added to your engine's configuration as follows:
```text
{
  "@context": [
    ...
    "https://linkedsoftwaredependencies.org/bundles/npm/@comunica/actor-optimize-query-operation-query-hint-join-order-fixed/^5.0.0/components/context.jsonld"
  ],
  "actors": [
    ...
    {
      "@id": "urn:comunica:default:optimize-query-operation/actors#query-hint-join-order-fixed",
      "@type": "ActorOptimizeQueryOperationQueryHintJoinOrderFixed"
    }
  ]
}
```
