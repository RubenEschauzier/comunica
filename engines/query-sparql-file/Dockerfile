FROM node:18

# Install location
WORKDIR /home/comunica/query-sparql-file/
RUN chown -R node:node .

USER node

# Copy the engine files (generated from package.json!files)
COPY --chown=node:node config config/
COPY --chown=node:node lib/*.js lib/
COPY --chown=node:node bin/*.js bin/
COPY --chown=node:node engine-default.js package.json ./

# Set the npm registry
ARG NPM_REGISTRY=https://registry.npmjs.org/

RUN npm config set @comunica:registry $NPM_REGISTRY
# RUN npm config set registry $NPM_REGISTRY
RUN npm config ls -l 

# Install the node module

RUN npm install --omit=dev --loglevel verbose

ENTRYPOINT ["node", "./bin/query.js"]

# Default command
CMD ["--help"]
