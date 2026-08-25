"""OCR stage: fill page texts with Qwen3-VL OCR or a placeholder.

Runs egs/utils/ocr_pages.py as a subprocess (so the heavy transformers import
and model load live in a clean process) unless USE_OCR=0.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from vdoc_pipeline.settings import (
    OCR_MAX_CHARS,
    PAGES_FINAL,
    PAGES_PREOCR,
    USE_OCR,
    ensure_dirs,

)

UTILS_DIR = Path(__file__).resolve().parent.parent / "egs" / "utils"
REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ensure_dirs()

    if not USE_OCR:
        print("USE_OCR=0 -> using placeholder text.")
        pages = json.loads(PAGES_PREOCR.read_text(encoding="utf-8"))
        for _, v in pages.items():
            if not str(v.get("text", "")).strip():
                v["text"] = "Ảnh trang tài liệu tiếng Việt. Vui lòng xem hình."
        PAGES_FINAL.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Final pages_content path: {PAGES_FINAL}")
        return

    print("Running OCR with local Qwen3-VL... This may take several minutes.")
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(REPO_ROOT))

    cmd = [
        sys.executable,
        str(UTILS_DIR / "ocr_pages.py"),
        "--input", str(PAGES_PREOCR),
        "--output", str(PAGES_FINAL),
        "--max-chars", str(OCR_MAX_CHARS),
    ]
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(UTILS_DIR), env=env, check=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        raise RuntimeError(f"ocr_pages.py failed with return code {result.returncode}")

    print(f"OCR finished in {elapsed:.1f}s -> {PAGES_FINAL}")


if __name__ == "__main__":
    main()
