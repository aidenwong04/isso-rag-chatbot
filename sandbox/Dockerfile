# The HW2 sandbox image: both agents, the MCP server, and the frozen inputs.
# Build from the repository root (the context), with Colima's Docker:
#   docker build -f sandbox/Dockerfile -t isso-hw2:1 .
#
# Layout (brief section 2.2, with the prepared inputs kept as one folder):
#   /app/              code, schemas, configs, mcp.json, and a venv from the pinned lock
#   /target/prepared/  frozen inputs written by prepare.py       (policy: read-only)
#   /target/labels/    the HW1 dev answer key                     (policy: no grant)
#   /target/private/   original chunks.json with embeddings       (policy: no grant)
#   /sandbox/work/     writable: evidence, run, scratch           (owned by uid 1000)
# Everything under /app and /target is owned by root and readable by all, so
# any access denial inside the sandbox comes from the policy, not the files.
# sandbox/Dockerfile.dockerignore allow-lists the context: no .env, no keys.

FROM python:3.13-slim

# Dependencies first, so code changes reuse this layer. Installed at build
# time because the sandbox has no access to a package index.
COPY agent/requirements.lock /app/agent/requirements.lock
RUN python -m venv /app/.venv \
 && /app/.venv/bin/pip install --no-cache-dir --no-deps -r /app/agent/requirements.lock

# Code: the HW2 agents and server, the HW1 modules they import, schemas, configs.
COPY agent/src/hw1/ /app/agent/src/hw1/
COPY agent/src/hw2/ /app/agent/src/hw2/
COPY agent/schemas/ /app/agent/schemas/
COPY agent/isso-agent-hw2 /app/agent/isso-agent-hw2
COPY configs/ /app/configs/
COPY mcp.json /app/mcp.json
# The policy itself, read-only: the CodeAct harness checks worker-reported denials against its grants.
COPY policy.yaml /app/policy.yaml
# The injection experiment's overlay (used only when a config enables [experiment]).
COPY fixtures/injection.json /app/fixtures/injection.json

# Frozen inputs, and the withheld material the policy does not grant.
COPY agent/prepared/ /target/prepared/
COPY agent/labels/ /target/labels/
COPY data/embeddings/chunks.json /target/private/chunks.json

# The launcher expects the venv at <repo>/env.
RUN ln -s /app/.venv /app/env \
 && chmod -R a+rX /app /target \
 && useradd --uid 1000 --create-home sandbox \
 && mkdir -p /sandbox/work \
 && chown -R 1000:1000 /sandbox

USER 1000
WORKDIR /sandbox
