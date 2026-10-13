"""
dump_worksheet - write the labeling worksheet for one query set.

Runs discovery on a query set and writes every alert to a JSON file with an
empty label, for labeling by hand. No model is called.

    env/bin/python agent/src/dump_worksheet.py agent/queries/dev agent/labels/dev.worksheet.json

The cosine score is left out on purpose, and entries are ordered by page and
chunk index rather than by rank: the labels are the ground truth the retriever
is judged against, so the labeler should not see the retriever's opinion.
"""

import argparse
import json
import sys
from pathlib import Path

from discovery import discover
from query import corpus_version


def sort_key(alert):
    # The chunk index is the part of the chunk id after the last "::". It is
    # compared as an integer so that 2 comes before 14.
    index = int(alert.chunk.chunk_id.rsplit("::", 1)[1])
    return (alert.query_id, alert.chunk.source, index)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="query set folder, holding queries.json")
    parser.add_argument("output", type=Path, help="worksheet file to create")
    parser.add_argument("-k", type=int, default=3, help="chunks retrieved per query")
    args = parser.parse_args()

    # A worksheet holds hand labels once it has been worked on, so an existing
    # one is never overwritten.
    if args.output.exists():
        sys.exit(f"{args.output} already exists; move or delete it first")

    alerts = sorted(discover(args.input / "queries.json", args.k), key=sort_key)

    worksheet = {
        "corpus_version": corpus_version(),
        "k": args.k,
        "alerts": [
            {
                "alert_id": alert.alert_id,
                "query": alert.query,
                "audience": alert.audience,
                "page": alert.chunk.source,
                "heading": alert.chunk.heading,
                "url": alert.chunk.url,
                "text": alert.chunk.text,
                "label": None,
                "note": "",
            }
            for alert in alerts
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as file:
        json.dump(worksheet, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print(f"{len(alerts)} alerts -> {args.output} (corpus {worksheet['corpus_version']})")


if __name__ == "__main__":
    main()
