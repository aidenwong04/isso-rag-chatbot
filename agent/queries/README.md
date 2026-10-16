# Query sets

Each folder is one target: the agent's `--input` points at a folder, and reads its `queries.json`.

- `dev/` - the development set. Its labels are in `../labels/dev.worksheet.json`, and are used to measure and tune the agent.
- `heldout-a/`, `heldout-b/` - held-out sets. Never tune against them.

The answer keys live in `../labels/`, outside every query folder, so the agent is never handed a path that contains them.

## Format

`queries.json` is a list of entries:

```json
[
  { "id": "dev-q01", "text": "the question, as a student would type it", "audience": "students" }
]
```

- `id` - the set name, then `-q`, then a two-digit number: `dev-q01`, `heldout-a-q01`.
  Zero-padded so that plain string order is also numeric order.
  It is the first part of every `alert_id` and of the report's row order.
- `text` - the question. Written from memory, not with the source page open.
- `audience` - who is asking.
