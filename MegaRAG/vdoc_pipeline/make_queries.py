"""Generate queries.jsonl from eval items and copy eval items to the output dir."""
from __future__ import annotations

import json

from vdoc_pipeline.settings import (
    EVAL_ITEMS_OUT,
    EVAL_ITEMS_TMP,
    QUERIES,
    ensure_dirs,
)


def main() -> None:
    ensure_dirs()

    with open(EVAL_ITEMS_TMP, "r", encoding="utf-8") as f:
        eval_items = [json.loads(line) for line in f if line.strip()]

    with open(QUERIES, "w", encoding="utf-8") as f:
        for item in eval_items:
            f.write(json.dumps({"question": item["question"]}, ensure_ascii=False) + "\n")

    with open(EVAL_ITEMS_OUT, "w", encoding="utf-8") as f:
        for item in eval_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved queries to {QUERIES}")
    print(f"Saved eval items to {EVAL_ITEMS_OUT} ({len(eval_items)} questions)")


if __name__ == "__main__":
    main()
