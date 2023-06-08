# Comunica Bus Model Train

[![npm version](https://badge.fury.io/js/%40comunica%2Fbus-model-train.svg)](https://www.npmjs.com/package/@comunica/bus-model-train)

A comunica bus for model training logic. Actors subscribed to this bus execute optimization passes for specific models and purposes.

This module is part of the [Comunica framework](https://github.com/comunica/comunica),
and should only be used by [developers that want to build their own query engine](https://comunica.dev/docs/modify/).

[Click here if you just want to query with Comunica](https://comunica.dev/docs/query/).

## Install

```bash
$ yarn add @comunica/bus-model-train
```

## Usage

## Bus usage

* **Context**: `"https://linkedsoftwaredependencies.org/bundles/npm/@comunica/bus-model-train/^1.0.0/components/context.jsonld"`
* **Bus name**: `ActorModelTrain:_default_bus`

## Creating actors on this bus

Actors extending [`ActorModelTrain`](TODO:jsdoc_url) are automatically subscribed to this bus.
