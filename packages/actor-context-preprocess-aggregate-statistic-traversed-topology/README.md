# Comunica Aggregate Statistic Traversed Topology Context Preprocess Actor

[![npm version](https://badge.fury.io/js/%40comunica%2Factor-context-preprocess-aggregate-statistic-traversed-topology.svg)](https://www.npmjs.com/package/@comunica/actor-context-preprocess-aggregate-statistic-traversed-topology)

A comunica Aggregate Statistic Traversed Topology Context Preprocess Actor. This statistic requires both link discover and dereference events to be tracked.
It listens for new link events and constructs the topology of the dereferenced subweb.

This module is part of the [Comunica framework](https://github.com/comunica/comunica),
and should only be used by [developers that want to build their own query engine](https://comunica.dev/docs/modify/).

[Click here if you just want to query with Comunica](https://comunica.dev/docs/query/).

## Install

```bash
$ yarn add @comunica/actor-context-preprocess-aggregate-statistic-traversed-topology
```

## Configure

After installing, this package can be added to your engine's configuration as follows:
```text
{
  "@context": [
    ...
    "https://linkedsoftwaredependencies.org/bundles/npm/@comunica/actor-context-preprocess-aggregate-statistic-traversed-topology/^1.0.0/components/context.jsonld"
  ],
  "actors": [
    ...
    {
      "@id": "urn:comunica:default:context-preprocess/actors#aggregate-statistic-traversed-topology",
      "@type": "ActorContextPreprocessAggregateStatisticTraversedTopology"
    }
  ]
}
```
