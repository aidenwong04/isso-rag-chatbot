#!/bin/sh
# One HW2 agent run in a fresh OpenShell sandbox, then its output copied out.
#
#   sandbox/run.sh <mcp|codeact> <dev|heldout-a|heldout-b> <run-name>
#   e.g. sandbox/run.sh mcp dev mcp-dev
#
# Creates a sandbox from isso-hw2:1 with policy.yaml and the "openai" provider
# (so the real key never enters the sandbox), runs the agent with the image's
# frozen inputs, downloads /sandbox/work/run (report.json, trace.jsonl) and
# /sandbox/work/evidence (the final packets) to agent/runs/sandbox/<run-name>/,
# and deletes the sandbox. Every launch starts from a fresh sandbox, so the
# packets always start empty (the agent also resets them first).
set -eu

kind="$1"
query_set="$2"
name="$3"
image="${ISSO_IMAGE:-isso-hw2:1}"
repo="$(cd "$(dirname "$0")/.." && pwd)"
out="$repo/agent/runs/sandbox/$name"
# Sandbox names are limited to 19 characters, so this one is short and random;
# the run name is kept in the output folder instead.
sandbox="hw2-$$-$(od -An -N2 -tx2 /dev/urandom | tr -d ' \n')"

case "$kind" in mcp|codeact) ;; *) echo "kind must be mcp or codeact" >&2; exit 2 ;; esac
case "$query_set" in dev|heldout-a|heldout-b) ;; *) echo "unknown query set $query_set" >&2; exit 2 ;; esac
if [ -e "$out" ]; then echo "$out already exists; choose another run name" >&2; exit 2; fi

cleanup() { openshell sandbox delete "$sandbox" >/dev/null 2>&1 || true; }
trap cleanup EXIT

openshell sandbox create --name "$sandbox" --from "$image" --policy "$repo/policy.yaml" \
    --provider openai --detach -- sleep infinity
sleep 3

status=0
openshell sandbox exec -n "$sandbox" --no-login-shell --no-tty -- \
    /app/agent/isso-agent-hw2 --input "/target/prepared/$query_set" \
    --output /sandbox/work/run --config "/app/configs/$kind.toml" || status=$?

mkdir -p "$out/evidence"
openshell sandbox download "$sandbox" /sandbox/work/run "$out"
openshell sandbox download "$sandbox" /sandbox/work/evidence "$out/evidence"
echo "run $name ($kind on $query_set) exit $status: $out"
exit "$status"
