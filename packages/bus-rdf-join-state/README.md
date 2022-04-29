# Comunica Bus RDF Join State

[![npm version](https://badge.fury.io/js/%40comunica%2Fbus-rdf-join-state.svg)](https://www.npmjs.com/package/@comunica/bus-rdf-join-state)

A comunica bus for rdf-join-state events. This bus is able to pass the current state of the join operation to the actor. Specifically designed to be used with Actor-rdf-join-inner-multi-reinforcement-learning.

This module is part of the [Comunica framework](https://github.com/comunica/comunica),
and should only be used by [developers that want to build their own query engine](https://comunica.dev/docs/modify/).

[Click here if you just want to query with Comunica](https://comunica.dev/docs/query/).

## Install

```bash
$ yarn add @comunica/bus-rdf-join-state
```

## Usage

## Bus usage

* **Context**: `"https://linkedsoftwaredependencies.org/bundles/npm/@comunica/bus-rdf-join-state/^1.0.0/components/context.jsonld"`
* **Bus name**: `ActorRdfJoinState:_default_bus`

## Creating actors on this bus

Actors extending [`ActorRdfJoinState`](TODO:jsdoc_url) are automatically subscribed to this bus.
