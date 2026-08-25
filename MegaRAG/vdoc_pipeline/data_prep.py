"""Download the Vietnamese document-image QA dataset and build page assets.

Produces:
  - pages_content_preocr.json  (pages with empty text -> filled by the OCR stage)
  - eval_items.jsonl           (gold QA items used for evaluation)
"""
from __future__ import annotations

import io
import json
import pathlib
import random
import shutil

from PIL import Image
from datasets import load_dataset

from vdoc_pipeline.settings import (
    CONFIG_NAME,
    DATASET_ID,
    EVAL_ITEMS_TMP,
    HF_CACHE,
    IMG_DIR,
    MAX_EVAL_QUESTIONS,
    MAX_QUESTIONS_PER_PAGE,
    PAGES_PREOCR,
    SAMPLE_PAGES,
    SPLIT,
    ensure_dirs,
)

Image.MAX_IMAGE_PIXELS = None


def save_image(img, path) -> None:
    path = pathlib.Path(path)
    if isinstance(img, Image.Image):
        img.convert("RGB").save(path, quality=90)
    elif isinstance(img, dict):
        if img.get("bytes") is not None:
            Image.open(io.BytesIO(img["bytes"])).convert("RGB").save(path, quality=90)
        elif img.get("path") and pathlib.Path(img["path"]).exists():
            shutil.copy(img["path"], path)
        elif img.get("src"):
            import requests

            r = requests.get(img["src"], timeout=30)
            Image.open(io.BytesIO(r.content)).convert("RGB").save(path, quality=90)
        else:
            raise ValueError("Cannot interpret image dict")
    elif isinstance(img, (str, pathlib.Path)):
        p = pathlib.Path(img)
        if p.exists():
            shutil.copy(p, path)
        else:
            import requests

            r = requests.get(img, timeout=30)
            Image.open(io.BytesIO(r.content)).convert("RGB").save(path, quality=90)
    else:
        raise ValueError(f"Unknown image type: {type(img)}")


def main() -> None:
    random.seed(42)
    ensure_dirs()

    print(f"Loading dataset {DATASET_ID} / {CONFIG_NAME} ...")
    ds = load_dataset(DATASET_ID, CONFIG_NAME, split=SPLIT, cache_dir=HF_CACHE)
    print(f"Loaded {len(ds)} rows")

    groups: dict = {}
    for i, row in enumerate(ds):
        img = row.get("image", None)
        if img is None:
            continue

        pid = str(row.get("id", i))
        q = row.get("question", "")
        a = row.get("model_answer", "")
        if not q or not a:
            continue

        if pid not in groups:
            groups[pid] = {"image": img, "questions": []}
        groups[pid]["questions"].append(
            {
                "question": q,
                "model_answer": a,
                "model_reasoning": row.get("model_reasoning", ""),
            }
        )

    print(f"Unique page ids: {len(groups)}")

    page_ids = list(groups.keys())
    page_ids.sort(key=lambda pid: len(groups[pid]["questions"]), reverse=True)
    selected_page_ids = page_ids[:SAMPLE_PAGES]
    print(f"Selected {len(selected_page_ids)} pages")

    pages_content: dict = {}
    eval_items: list = []
    qid = 0

    for page_idx, pid in enumerate(selected_page_ids):
        img_path = IMG_DIR / f"page_{page_idx:03d}.jpg"
        save_image(groups[pid]["image"], img_path)

        pages_content[str(page_idx)] = {
            "text": "",
            "page_image": str(img_path),
            "figure_images": [],
            "page_id": pid,
        }

        for q in groups[pid]["questions"][:MAX_QUESTIONS_PER_PAGE]:
            if len(eval_items) >= MAX_EVAL_QUESTIONS:
                break
            eval_items.append(
                {
                    "index": qid,
                    "page_idx": page_idx,
                    "page_id": pid,
                    "question": q["question"],
                    "gold_answer": q["model_answer"],
                }
            )
            qid += 1

    if len(eval_items) == 0:
        raise RuntimeError("No evaluation questions were created. Increase SAMPLE_PAGES.")

    PAGES_PREOCR.write_text(json.dumps(pages_content, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(EVAL_ITEMS_TMP, "w", encoding="utf-8") as f:
        for item in eval_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved pre-OCR pages: {PAGES_PREOCR}")
    print(f"Saved eval items: {EVAL_ITEMS_TMP} ({len(eval_items)} questions)")


if __name__ == "__main__":
    main()
