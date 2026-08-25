"""Central path & config settings for the vdoc pipeline.

Everything is overridable via environment variables so the same code runs on
Kaggle (stage 2 of megarag-mmkg-kaggle-v2.ipynb) and locally.
"""
from __future__ import annotations

import os
from pathlib import Path


def _default_base() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working") / "reproduce_MegaRAG"
    return Path(__file__).resolve().parent.parent


BASE_DIR = Path(os.environ.get("VDOC_BASE_DIR", str(_default_base())))
TMP_DIR = Path(os.environ.get("VDOC_TMP_DIR", BASE_DIR / "vdoc_tmp"))
OUT_DIR = Path(os.environ.get("VDOC_OUT_DIR", BASE_DIR / "vdoc_outputs"))
IMG_DIR = TMP_DIR / "images"

DATASET_ID = os.environ.get("VDOC_DATASET_ID", "trannhiem/TranNhiem-Vietnamese-DocumentImage-Reasoning")
CONFIG_NAME = os.environ.get("VDOC_DATASET_CONFIG", "preview")
SPLIT = os.environ.get("VDOC_DATASET_SPLIT", "train")
HF_CACHE = os.environ.get("HF_DATASETS_CACHE", str(TMP_DIR / "hf" / "datasets"))

SAMPLE_PAGES = int(os.environ.get("SAMPLE_PAGES", "30"))
MAX_QUESTIONS_PER_PAGE = int(os.environ.get("MAX_QUESTIONS_PER_PAGE", "3"))
MAX_EVAL_QUESTIONS = int(os.environ.get("MAX_EVAL_QUESTIONS", "90"))

USE_OCR = os.environ.get("USE_OCR", "1") == "1"
OCR_MAX_CHARS = int(os.environ.get("OCR_MAX_CHARS", "3000"))

PAGES_PREOCR = TMP_DIR / "pages_content_preocr.json"
PAGES_FINAL = TMP_DIR / "pages_content.json"
EVAL_ITEMS_TMP = TMP_DIR / "eval_items.jsonl"
QUERIES = TMP_DIR / "queries.jsonl"

EVAL_ITEMS_OUT = OUT_DIR / "eval_items.jsonl"
RESULTS = OUT_DIR / "results" / "results.jsonl"
BASELINE_RESULTS = OUT_DIR / "results" / "baseline_results.jsonl"
JUDGES = OUT_DIR / "judges.jsonl"
METRICS = OUT_DIR / "metrics.json"
PREDICTIONS_CSV = OUT_DIR / "predictions.csv"


def ensure_dirs() -> None:
    """Create all working/output directories."""
    for d in [BASE_DIR, TMP_DIR, OUT_DIR, IMG_DIR, RESULTS.parent]:
        d.mkdir(parents=True, exist_ok=True)
