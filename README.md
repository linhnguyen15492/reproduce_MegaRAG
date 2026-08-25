# reproduce_MegaRAG

Reproduction of **MegaRAG** ([arXiv:2512.20626](https://arxiv.org/abs/2512.20626), accepted at ACL 2026) on Kaggle: Multimodal Knowledge Graph–based RAG that enables _global_ question answering over long documents by constructing a MMKG from parsed document pages.

The repo ships **two runnable pipelines** driven by a single Kaggle notebook (`megarag-mmkg-kaggle-v2.ipynb`):

|                   | Stage 1 (Sections 1–8)                                             | Stage 2 (Section 9, `vdoc`)                                                                       |
| ----------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| Input             | `World_History_Volume_1.pdf` (Kaggle dataset `world-history-tiny`) | `trannhiem/TranNhiem-Vietnamese-DocumentImage-Reasoning` (HF dataset, Vietnamese document images) |
| Document parsing  | MinerU (`magic-pdf` CLI) → page images → assets manifest           | Direct page images + Qwen3-VL OCR                                                                 |
| LLM backend       | OpenAI `gpt-4o-mini` API                                           | **Local Qwen3-VL-8B** (bnb-4bit, `device_map="auto"`)                                             |
| Embedding backend | GME-Qwen2-VL-2B (`trust_remote_code`, transformers ≤ 4.51)         | `gme_compat.py` wrapper (transformers ≥ 4.57 compatible)                                          |
| Evaluation        | Manual inspection + KG stats                                       | EM / Token-F1 / Token-Recall + LLM-as-Judge vs no-RAG baseline                                    |

---

## Repository Structure

```
reproduce_MegaRAG/
├── megarag-mmkg-kaggle-v2.ipynb   # Main notebook: Stage 1 (Sec. 1–8) + Stage 2 (Sec. 9)
├── MegaRAG/                       # MegaRAG package + example scripts (editable install)
│   ├── megarag/                   # Core library (operate, storage, prompt, llms, kg)
│   ├── egs/
│   │   ├── utils/                 # Pipeline CLIs
│   │   │   ├── construct_mmkg.py      # Build MMKG (--input-dir = pages_content.json)
│   │   │   ├── query_mmkg.py          # Batch querying (resume/pacing/cache persistence)
│   │   │   ├── qwen_llm.py            # Local Qwen3-VL async wrapper        [Stage 2]
│   │   │   ├── gme_compat.py          # GME embedder for transformers>=4.57  [Stage 2]
│   │   │   ├── ocr_pages.py           # Page-image OCR via Qwen3-VL         [Stage 2]
│   │   │   ├── baseline_direct.py     # No-RAG baseline QA                  [Stage 2]
│   │   │   └── judge_qwen.py          # LLM-as-judge evaluation             [Stage 2]
│   │   ├── vdoc/conf/addon_params.yaml                                [Stage 2]
│   │   └── world_history_tiny/conf/addon_params.yaml                  [Stage 1]
│   └── vdoc_pipeline/             # Data-prep + evaluation package         [Stage 2]
│       └── cli.py                 # python -m vdoc_pipeline {prep|ocr|queries|build|query|baseline|judge|metrics|all}
├── LightRAG/                      # Upstream LightRAG v1.4.x (editable install) + reproduce steps
├── MinerU/                        # Upstream MinerU/magic-pdf (editable install)
├── archive/                       # Archived experiment notebooks (vdoc prototype, kaggle run logs)
├── implementation_plan.md         # Rate-limit fix plan (OpenAI 429s)
├── improvement_details.md         # Changelog of all fixes/improvements
└── walkthrough.md                 # Notes on direct-execution error logging
```

---

## Requirements

### Hardware

- **Stage 1:** any single GPU (P100/T4 16 GB fine — LLM calls go to the OpenAI API).
- **Stage 2:** 2× T4 16 GB recommended. Qwen3-VL-8B (4-bit) is sharded across visible GPUs; the GME embedding model runs on CPU by default (`EMBED_DEVICE=cpu`) to keep VRAM free.

### Software / accounts

- Kaggle account with Internet enabled and GPU accelerator selected (Session options).
- An OpenAI API key (Stage 1 only), stored as a Kaggle Secret named `OPENAI_API_KEY`.
- Access to Hugging Face (dataset download in Stage 2). No HF token required for the preview split.

---

## Running on Kaggle

1. **Upload** `megarag-mmkg-kaggle-v2.ipynb` to Kaggle (File → Import Notebook) and select a GPU accelerator (2× T4 for Stage 2).
2. **Add the input dataset** `nguyenngochonglinh/world-history-tiny` (Add Input → Datasets). Cell 0 also pulls it via `kagglehub`.
3. **Add the secret** `OPENAI_API_KEY` (Add-ons → Secrets). Used only by Stage 1.
4. **Run cells in order.**

### Stage 1 — World History MMKG with OpenAI (Sections 1–8)

| Section | What happens                                                                                                                                                           |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1–2     | Clones this repo into `/kaggle/working/reproduce_MegaRAG`, installs pinned deps (`transformers==4.51.3` for MinerU) and editable packages (MinerU, LightRAG, MegaRAG). |
| 3       | Downloads MinerU weights (PDF-Extract-Kit, LayoutReader), writes `magic-pdf.json` (formula recognition disabled → faster, no MBart issues).                            |
| 4       | Loads the OpenAI key, writes `MegaRAG/env.sh`.                                                                                                                         |
| 5       | Copies the PDF + `queries_short.txt` into `data/`.                                                                                                                     |
| 6.1–6.4 | Parses PDF pages 0–9 with MinerU → renders page images (150 dpi) → builds `pages_content.json` manifest → **builds the MMKG** via `construct_mmkg.py`.                 |
| 7       | Runs batch queries via `query_mmkg.py` (concurrency 1, retry pacing, `--resume`).                                                                                      |
| 8       | Displays answers + knowledge-graph stats; zips the run folder as `results.zip`.                                                                                        |

Artifacts: `World_History_Volume_1_run/exp/World_History_Volume_1/`
(`results/results.json`, `*.graphml`, KV stores, vector DBs).

### Stage 2 — Vietnamese Document QA with local Qwen3-VL (Section 9)

> ⚠️ **Version constraint:** Stage 1 pins `transformers==4.51.3`; Qwen3-VL requires `>=4.57`.
> Finish **all** Stage 1 sections first. Section 9 upgrades transformers — after that,
> do **not** re-run MinerU sections (3, 6.1) in the same session.

| Cell | What happens                                                                                                                                       |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| 9.0  | Sets Stage-2 env (paths, sampling flags, backends) and defines the `run_stage()` helper that streams subprocess logs.                              |
| 9.0b | Installs `transformers>=4.57,<5` + `bitsandbytes`, `qwen_vl_utils`, `datasets`, ... (hard-fails if < 4.57).                                        |
| 9.1  | Downloads the HF dataset preview → selects top pages/questions → `pages_content_preocr.json` + `eval_items.jsonl`.                                 |
| 9.2  | OCR page images with local Qwen3-VL (`USE_OCR=0` → placeholder text instead).                                                                      |
| 9.3  | Builds the vdoc MMKG (`construct_mmkg.py` with `MEGARAG_LLM_BACKEND=qwen_local`, `MEGARAG_EMBED_BACKEND=gme_compat`); clears stale `egs/vdoc/exp`. |
| 9.4  | Queries MegaRAG over the generated `queries.jsonl`.                                                                                                |
| 9.5  | Optional no-RAG baseline + LLM-as-judge (toggle `RUN_BASELINE` / `RUN_LLM_JUDGE` in 9.0).                                                          |
| 9.6  | Computes EM / Token-F1 / Token-Recall (+ judge accuracy) → prints comparison table MegaRAG vs baseline.                                            |
| 9.7  | Zips outputs as `vdoc_results.zip`.                                                                                                                |

Artifacts: `<repo>/vdoc_outputs/`
(`results/results.jsonl`, `results/baseline_results.jsonl`, `judges.jsonl`, `metrics.json`, `predictions.csv`, `eval_items.jsonl`).

Tunable knobs live in the 9.0 cell: `SAMPLE_PAGES` (default 30), `MAX_QUESTIONS_PER_PAGE` (3), `MAX_EVAL_QUESTIONS` (90), `JUDGE_SAMPLE` (120), `USE_OCR`, `EMBED_DEVICE`.

---

## Configuration Reference

The notebook is the only supported way to run the pipelines today; local/standalone
execution of the CLI stages has **not been tested yet**. All knobs are set in the
notebook (Stage 2: Section 9.0 cell; Stage 1: `egs/world_history_tiny/conf/addon_params.yaml`)
and can be overridden via environment variables:

| Variable                                                         | Default                                  | Purpose                                                        |
| ---------------------------------------------------------------- | ---------------------------------------- | -------------------------------------------------------------- |
| `VDOC_TMP_DIR` / `VDOC_OUT_DIR`                                  | repo-local `vdoc_tmp/` / `vdoc_outputs/` | Working & output dirs                                          |
| `MEGARAG_LLM_BACKEND`                                            | `openai`                                 | `openai` or `qwen_local` (used by construct/query scripts)     |
| `MEGARAG_EMBED_BACKEND`                                          | `hf_gme`                                 | `hf_gme` (legacy loader) or `gme_compat` (transformers ≥ 4.57) |
| `EMBED_DEVICE`                                                   | `cpu` on Kaggle                          | Where the GME embedding model runs                             |
| `QWEN_MODEL_ID`                                                  | `unsloth/Qwen3-VL-8B-Instruct-bnb-4bit`  | Local vision-LLM                                               |
| `QWEN_MAX_PIXELS` / `QWEN_MAX_INPUT_CHARS`                       | `768*28*28` / `24000`                    | Vision token & prompt caps                                     |
| `USE_OCR`                                                        | `1`                                      | OCR pages before MMKG build (`0` = placeholder text)           |
| `SAMPLE_PAGES` / `MAX_QUESTIONS_PER_PAGE` / `MAX_EVAL_QUESTIONS` | `30` / `3` / `90`                        | Eval-set sizing                                                |
| `RUN_BASELINE` / `RUN_LLM_JUDGE` / `JUDGE_SAMPLE`                | `1` / `1` / `120`                        | Optional eval extras                                           |

Per-experiment knobs (entity types, language, token budgets, concurrency) live in
`egs/<recipe>/conf/addon_params.yaml`.

---

## Documentation

- [`improvement_details.md`](improvement_details.md) — dated changelog (rate-limit fixes, Stage 2 integration).
- [`implementation_plan.md`](implementation_plan.md) — analysis & fix plan for OpenAI 429 errors.
- [`walkthrough.md`](walkthrough.md) — direct-execution/unmasked-error logging notes.
- `archive/` — earlier prototypes (vdoc+Qwen3 prototype, raw Kaggle run dump).

## Credits

- [MegaRAG](https://github.com/AI-Application-and-Integration-Lab/MegaRAG) — AI Application and Integration Lab
- [LightRAG](https://github.com/HKUDS/LightRAG) (v1.4.3) — HKUDS
- [MinerU](https://github.com/opendatalab/MinerU) — OpenDataLab
- Models: `unsloth/Qwen3-VL-8B-Instruct-bnb-4bit`, `Alibaba-NLP/gme-Qwen2-VL-2B-Instruct`,
  dataset `trannhiem/TranNhiem-Vietnamese-DocumentImage-Reasoning`.

## AI Assistance Disclosure

Quá trình reproduce này có sự trợ giúp của AI (LLM coding assistant) trong các công việc:
refactor code từ notebook thành module Python, tích hợp Stage 2 vào notebook, viết tài liệu
(README, changelog) và debug. Toàn bộ thay đổi đều được kiểm tra/lưu ý và phê duyệt bởi
người thực hiện trước khi đưa vào repo.
