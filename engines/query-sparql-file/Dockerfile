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
# RUN cat engine-default.js 


# Set the npm registry
ARG NPM_REGISTRY=https://registry.npmjs.org/

RUN echo ${NPM_REGISTRY}
RUN npm config set @comunica:registry $NPM_REGISTRY
RUN npm config ls -l 

# Install the node module
RUN npm install --omit=dev

# RUN ls
# RUN ls node_modules/@comunica
# RUN cat engine-default.js
# Run base binary (generated from package.json!bin)
ENTRYPOINT ["node", "./bin/query.js"]

# Default command
CMD ["--help"]
