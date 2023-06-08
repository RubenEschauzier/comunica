# Comunica Bus Model Inference

[![npm version](https://badge.fury.io/js/%40comunica%2Fbus-model-inference.svg)](https://www.npmjs.com/package/@comunica/bus-model-inference)

A comunica bus for model inference. Other parts of comunica can use actors subscribed to this bus to perform machine learning tasks.

This module is part of the [Comunica framework](https://github.com/comunica/comunica),
and should only be used by [developers that want to build their own query engine](https://comunica.dev/docs/modify/).

[Click here if you just want to query with Comunica](https://comunica.dev/docs/query/).

## Install

```bash
$ yarn add @comunica/bus-model-inference
```

## Usage

## Bus usage

* **Context**: `"https://linkedsoftwaredependencies.org/bundles/npm/@comunica/bus-model-inference/^1.0.0/components/context.jsonld"`
* **Bus name**: `ActorModelInference:_default_bus`

## Creating actors on this bus

Actors extending [`ActorModelInference`](TODO:jsdoc_url) are automatically subscribed to this bus.
