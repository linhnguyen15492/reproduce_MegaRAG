"""Evaluation metrics for vdoc Stage 2: EM, Token F1, Token Recall + LLM-judge.

Pure functions (testable) plus a CLI entrypoint that produces metrics.json and
predictions.csv comparing MegaRAG against the no-RAG baseline.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from vdoc_pipeline.settings import (
    BASELINE_RESULTS,
    EVAL_ITEMS_OUT,
    JUDGES,
    METRICS,
    PREDICTIONS_CSV,
    RESULTS,
    ensure_dirs,
)


def normalize_answer(s) -> str:
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r"```.*?```", " ", s, flags=re.S)
    s = re.sub(r"[*_#`>|]", " ", s)
    s = unicodedata.normalize("NFC", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: str):
    return re.findall(r"\w+", s, flags=re.UNICODE)


def token_f1(pred: str, gold: str) -> float:
    p_toks = tokenize(pred)
    g_toks = tokenize(gold)
    if not p_toks or not g_toks:
        return 0.0
    common = Counter(p_toks) & Counter(g_toks)
    num = sum(common.values())
    if num == 0:
        return 0.0
    p = num / len(p_toks)
    r = num / len(g_toks)
    return 2 * p * r / (p + r)


def token_recall(pred: str, gold: str) -> float:
    p_toks = tokenize(pred)
    g_toks = tokenize(gold)
    if not g_toks:
        return 0.0
    common = Counter(p_toks) & Counter(g_toks)
    num = sum(common.values())
    return num / len(g_toks)


def count_cuda_errors(text) -> int:
    if not text:
        return 0
    t = str(text).lower()
    return t.count("cuda out of memory") + t.count("outofmemoryerror")


def load_jsonl(path) -> dict:
    out: dict = {}
    path = Path(path)
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            out[rec.get("index")] = rec
    return out


def compute_rows(eval_items, pred_map, baseline_map, judge_map):
    rows = []
    for item in eval_items:
        idx = item["index"]
        gold = item["gold_answer"]

        megarag_rec = pred_map.get(idx, {})
        megarag_pred = "" if megarag_rec.get("error") else megarag_rec.get("answer", "")

        base_rec = baseline_map.get(idx, {})
        base_pred = "" if base_rec.get("error") else base_rec.get("answer", "")

        norm_gold = normalize_answer(gold)
        row = {
            "index": idx,
            "page_id": item.get("page_id", ""),
            "question": item["question"],
            "gold_answer": gold,
            "megarag_answer": megarag_pred,
            "baseline_answer": base_pred,
            "megarag_em": int(norm_gold != "" and normalize_answer(megarag_pred) == norm_gold),
            "megarag_f1": token_f1(normalize_answer(megarag_pred), norm_gold),
            "megarag_recall": token_recall(normalize_answer(megarag_pred), norm_gold),
            "baseline_em": int(norm_gold != "" and normalize_answer(base_pred) == norm_gold),
            "baseline_f1": token_f1(normalize_answer(base_pred), norm_gold),
            "baseline_recall": token_recall(normalize_answer(base_pred), norm_gold),
            "megarag_cuda_errors": count_cuda_errors(megarag_pred),
        }
        if idx in judge_map:
            row["llm_judge"] = judge_map[idx]
        rows.append(row)
    return rows


def compute_metrics(df):
    metrics = {
        "num_eval": len(df),
        "megarag_exact_match": float(df["megarag_em"].mean()) if len(df) else 0.0,
        "megarag_token_f1": float(df["megarag_f1"].mean()) if len(df) else 0.0,
        "megarag_token_recall": float(df["megarag_recall"].mean()) if len(df) else 0.0,
        "megarag_cuda_error_answers": int((df["megarag_cuda_errors"] > 0).sum()),
        "baseline_exact_match": float(df["baseline_em"].mean()) if len(df) else 0.0,
        "baseline_token_f1": float(df["baseline_f1"].mean()) if len(df) else 0.0,
        "baseline_token_recall": float(df["baseline_recall"].mean()) if len(df) else 0.0,
    }

    if "llm_judge" in df.columns:
        judged = df[df["llm_judge"].isin(["YES", "NO"])]
        if len(judged) > 0:
            metrics["megarag_llm_judge_accuracy"] = float((judged["llm_judge"] == "YES").mean())
            metrics["num_llm_judged"] = int(len(judged))
    return metrics


def build_comparison_table(metrics):
    comparison = {
        "Method": ["MegaRAG + Qwen3-VL-8B", "Baseline Qwen3-VL-8B (no RAG)"],
        "EM": [metrics["megarag_exact_match"], metrics["baseline_exact_match"]],
        "Token F1": [metrics["megarag_token_f1"], metrics["baseline_token_f1"]],
        "Token Recall": [metrics["megarag_token_recall"], metrics["baseline_token_recall"]],
    }
    if "megarag_llm_judge_accuracy" in metrics:
        comparison["LLM-Judge Acc"] = [
            metrics.get("megarag_llm_judge_accuracy", float("nan")),
            float("nan"),
        ]
    return comparison


def main() -> None:
    import pandas as pd

    ensure_dirs()

    with open(EVAL_ITEMS_OUT, "r", encoding="utf-8") as f:
        eval_items = [json.loads(line) for line in f if line.strip()]

    pred_map = load_jsonl(RESULTS)
    baseline_map = load_jsonl(BASELINE_RESULTS)

    judge_map: dict = {}
    if Path(JUDGES).exists():
        with open(JUDGES, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    judge_map[rec["index"]] = rec.get("verdict", "NO")

    rows = compute_rows(eval_items, pred_map, baseline_map, judge_map)
    df = pd.DataFrame(rows)
    metrics = compute_metrics(df)

    METRICS.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(PREDICTIONS_CSV, index=False)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    comparison = pd.DataFrame(build_comparison_table(metrics))
    print("\nFINAL COMPARISON")
    print(comparison.to_string(index=False))
    print(f"\nSaved metrics to {METRICS}")
    print(f"Saved predictions to {PREDICTIONS_CSV}")


if __name__ == "__main__":
    main()
