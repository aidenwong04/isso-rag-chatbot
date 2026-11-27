"""
fault_probe - one agent run with the real model, where two of the model's tool
calls are corrupted on their way to the dispatcher (malformed-argument example,
brief section 7, comparison 2).

The model, host, dispatcher, MCP client, server, and backend are the real ones;
only the model's reply is changed, in two places, after the API returns it:

  item 1 (first in frozen order), first classify call: the label is sent as a
      string ("1" instead of 1), which the tool's schema does not allow.
  item 2, first read call: the arguments are cut off mid-JSON ('{"chunk_id": ').

Everything after that is the model's own behavior, including how it recovers
from the refusals. Each corruption is recorded in <output>/faults.jsonl (the
original and the corrupted arguments), so the trace and report stay in their
schema formats and every changed call can be found.

  HW2 (sandbox or local), same arguments as agent/isso-agent-hw2:
    python agent/src/hw2/fault_probe.py hw2 --input <set> --output <dir> --config <toml> [--mcp-json <file>]
  HW1 (local), same arguments as agent/isso-agent:
    python agent/src/hw2/fault_probe.py hw1 --input agent/queries/dev --output <dir> --config agent/configs/full.toml
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HW1_DIR = HERE.parent / "hw1"


class Faults:
    """Decides which tool calls to corrupt, and records each corruption."""

    PLAN = {0: "classify", 1: "read"}   # item index -> the tool whose first call is corrupted

    def __init__(self):
        self.items: list[str] = []      # each conversation's first user message, in order
        self.done: set[int] = set()
        self.calls = 0
        self.log: list[dict] = []

    def item_index(self, messages) -> int:
        first_user = next(m["content"] for m in messages if m["role"] == "user")
        if first_user not in self.items:
            self.items.append(first_user)
        return self.items.index(first_user)

    def apply(self, response, messages) -> None:
        self.calls += 1
        i = self.item_index(messages)
        tool = self.PLAN.get(i)
        calls = response.choices[0].message.tool_calls or []
        if tool is None or i in self.done or not calls or calls[0].function.name != tool:
            return
        function = calls[0].function
        original = function.arguments
        if tool == "classify":
            arguments = json.loads(original)
            arguments["label"] = str(arguments.get("label"))
            corrupted = json.dumps(arguments)
        else:
            corrupted = '{"chunk_id": '
        function.arguments = corrupted
        self.done.add(i)
        self.log.append({"model_call": self.calls, "item_index": i + 1, "tool_call_id": calls[0].id,
                         "tool": tool, "original_arguments": original, "corrupted_arguments": corrupted,
                         "item_opening_message": self.items[i]})

    def write(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "faults.jsonl", "w", encoding="utf-8") as f:
            for entry in self.log:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class _Completions:
    def __init__(self, inner, faults, is_async):
        self.inner, self.faults, self.is_async = inner, faults, is_async

    def create(self, **kwargs):
        if self.is_async:
            async def go():
                response = await self.inner.create(**kwargs)
                self.faults.apply(response, kwargs["messages"])
                return response
            return go()
        response = self.inner.create(**kwargs)
        self.faults.apply(response, kwargs["messages"])
        return response


class FaultyClient:
    """Same interface as (Async)OpenAI for chat.completions.create."""

    def __init__(self, client, faults: Faults, is_async: bool):
        completions = _Completions(client.chat.completions, faults, is_async)
        self.chat = type("Chat", (), {"completions": completions})()


def output_arg(argv: list[str]) -> Path:
    return Path(argv[argv.index("--output") + 1])


def run_hw2(argv: list[str], faults: Faults) -> None:
    import argparse
    import asyncio
    sys.path.insert(0, str(HERE))
    import cli
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mcp-json", type=Path)
    args = parser.parse_args(argv)
    client = FaultyClient(cli.default_model_client(), faults, is_async=True)
    try:
        status = asyncio.run(cli.run_agent(args.input, args.output, args.config,
                                           model_client=client, mcp_json=args.mcp_json))
    finally:
        faults.write(args.output)
    print(f"{status}: {args.output}")


def run_hw1(argv: list[str], faults: Faults) -> None:
    # HW1's modules use flat imports with the same names as HW2's (cli, nodes),
    # so only its folder may be on the path.
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
    sys.path.insert(0, str(HW1_DIR))
    import nodes
    nodes._client_cache = FaultyClient(nodes.openai_client(), faults, is_async=False)
    import cli
    sys.argv = ["agent/isso-agent", *argv]
    try:
        cli.main()
    finally:
        faults.write(output_arg(argv))


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("hw1", "hw2"):
        sys.exit(__doc__)
    faults = Faults()
    (run_hw1 if sys.argv[1] == "hw1" else run_hw2)(sys.argv[2:], faults)
    print(f"{len(faults.log)} tool calls corrupted; see faults.jsonl")


if __name__ == "__main__":
    main()
