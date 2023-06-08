# Comunica Mediator Model Inference

[![npm version](https://badge.fury.io/js/%40comunica%2Fmediator-model-inference.svg)](https://www.npmjs.com/package/@comunica/mediator-model-inference)

A comunica model inference mediator, that will select the model that can perform a certain task

This module is part of the [Comunica framework](https://github.com/comunica/comunica),
and should only be used by [developers that want to build their own query engine](https://comunica.dev/docs/modify/).

[Click here if you just want to query with Comunica](https://comunica.dev/docs/query/).

## Install

```bash
$ yarn add @comunica/mediator-model-inference
```

## Configure

After installing, this mediator can be instantiated as follows:
```text
{
  "@context": [
    ...
    "https://linkedsoftwaredependencies.org/bundles/npm/@comunica/mediator-all/^1.0.0/components/context.jsonld"  
  ],
  "actors": [
    ...
    {
      "@type": "SomeActor",
      "someMediator": {
        "@id": "#myMediator",
        "@type": "MediatorModelInference",
        "bus": { "@id": "ActorQueryOperation:_default_bus" }
      }
    }
  ]
}
```

### Config Parameters

TODO: fill in parameters (this section can be removed if there are none)

* `bus`: Identifier of the bus to mediate over.

