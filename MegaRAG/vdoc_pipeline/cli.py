"""CLI orchestrator for the vdoc pipeline (MegaRAG Stage 2).

Stages:
  prep     - download dataset, build pages_content_preocr.json + eval items
  ocr      - fill page texts (Qwen3-VL OCR or placeholder)
  queries  - generate queries.jsonl for MegaRAG
  build    - construct MMKG (construct_mmkg.py, local backends via env vars)
  query    - run MegaRAG queries (query_mmkg.py)
  baseline - no-RAG Qwen3-VL answers (baseline_direct.py)
  judge    - LLM-judge evaluation (judge_qwen.py)
  metrics  - compute EM / Token F1 / Recall and comparison table

Example:
  python -m vdoc_pipeline prep
  python -m vdoc_pipeline all
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
UTILS_DIR = REPO_ROOT / "egs" / "utils"
CONFIG_FILE = REPO_ROOT / "egs" / "vdoc" / "conf" / "addon_params.yaml"
# WORKING_DIR = REPO_ROOT / "egs" / "vdoc" / "exp" / "vdoc"
WORKING_DIR = Path(os.environ.get("VDOC_RUN_DIR", str(REPO_ROOT / "vdoc_run" / "exp" / "vdoc")))

def _stage_env() -> dict:
    """Env for subprocess stages; defaults mirror the Kaggle stage-2 notebook."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in [str(REPO_ROOT), str(UTILS_DIR), env.get("PYTHONPATH", "")] if p
    )
    env.setdefault("MEGARAG_LLM_BACKEND", "qwen_local")
    env.setdefault("MEGARAG_EMBED_BACKEND", "gme_compat")
    env.setdefault("EMBED_DEVICE", "cpu")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def _run(cmd: list, env_overrides: dict | None = None) -> None:
    print("[vdoc] Running:", " ".join(str(c) for c in cmd))
    env = _stage_env()
    for k, v in (env_overrides or {}).items():
        env[k] = v
    result = subprocess.run([str(c) for c in cmd], cwd=str(REPO_ROOT), env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Stage failed with return code {result.returncode}: {cmd}")


def stage_prep() -> None:
    from vdoc_pipeline import data_prep

    data_prep.main()


def stage_ocr() -> None:
    from vdoc_pipeline import run_ocr

    run_ocr.main()


def make_queries() -> None:
    from vdoc_pipeline import make_queries

    make_queries.main()


def stage_build(max_retries: int = 1, retry_delay: int = 5) -> None:
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    _run([
        sys.executable,
        UTILS_DIR / "construct_mmkg.py",
        "--config-file", CONFIG_FILE,
        "--working-dir", WORKING_DIR,
        "--input-dir", str(_pages_file()),
        "--max-retries", max_retries,
        "--retry-delay", retry_delay,
    ])


def stage_query(max_retries: int = 1, retry_delay: float = 5.0, concurrency: int = 1) -> None:
    results_path = _results_file()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    _run([
        sys.executable,
        UTILS_DIR / "query_mmkg.py",
        "--config-file", CONFIG_FILE,
        "--working-dir", WORKING_DIR,
        "--input-queries", _queries_file(),
        "--output-file", results_path,
        "--output-format", "jsonl",
        "--concurrency", concurrency,
        "--max-retries", max_retries,
        "--retry-delay", retry_delay,
        "--resume",
    ])


def stage_baseline() -> None:
    # Spread Qwen3-VL across all visible GPUs (device_map="auto") for the baseline.
    _run(
        [
            sys.executable,
            UTILS_DIR / "baseline_direct.py",
            "--eval-items", _eval_items_file(),
            "--pages", str(_pages_file()),
            "--output", _baseline_results_file(),
            "--resume",
        ],
        env_overrides={"CUDA_VISIBLE_DEVICES": "0,1"},
    )


def stage_judge(sample: int = 20) -> None:
    _run([
        sys.executable,
        UTILS_DIR / "judge_qwen.py",
        "--eval-items", _eval_items_file(),
        "--results", _results_file(),
        "--output", _judges_file(),
        "--sample", sample,
    ])


def stage_metrics() -> None:
    from vdoc_pipeline import metrics

    metrics.main()


# Lazy path imports keep settings overridable until a stage actually runs.
def _pages_file():
    from vdoc_pipeline.settings import PAGES_FINAL

    return Path(PAGES_FINAL)


def _queries_file():
    from vdoc_pipeline.settings import QUERIES

    return Path(QUERIES)


def _eval_items_file():
    from vdoc_pipeline.settings import EVAL_ITEMS_OUT

    return Path(EVAL_ITEMS_OUT)


def _results_file():
    from vdoc_pipeline.settings import RESULTS

    return Path(RESULTS)


def _baseline_results_file():
    from vdoc_pipeline.settings import BASELINE_RESULTS

    return Path(BASELINE_RESULTS)


def _judges_file():
    from vdoc_pipeline.settings import JUDGES

    return Path(JUDGES)


STAGES = {
    "prep": stage_prep,
    "ocr": stage_ocr,
    "make_queries": make_queries,
    "build": stage_build,
    "query": stage_query,
    "baseline": stage_baseline,
    "judge": stage_judge,
    "metrics": stage_metrics,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="vdoc pipeline orchestrator")
    parser.add_argument("stages", nargs="+", choices=list(STAGES) + ["all"],
                        help="One or more stages to run, in order.")
    args = parser.parse_args()

    selected: list = []
    for s in args.stages:
        if s == "all":
            selected = ["prep", "ocr", "queries", "build", "query"]
            if os.environ.get("RUN_BASELINE", "0") == "1":
                selected.append("baseline")
            if os.environ.get("RUN_LLM_JUDGE", "0") == "1":
                selected.append("judge")
            selected.append("metrics")
            break
        selected.append(s)

    for s in selected:
        print(f"\n===== [vdoc] Stage: {s} =====")
        STAGES[s]()
    print("\n[vdoc] All requested stages completed.")


if __name__ == "__main__":
    main()
