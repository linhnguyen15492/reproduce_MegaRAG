# Improvement Details

## 2026-08-20

### 1. Reduced OpenAI burst in two-step graph query

- File: `MegaRAG/megarag/operate.py`
- Change: `kg_two_step_query()` no longer runs `kg_query` and `naive_query` with `asyncio.gather()`.
- New behavior: the KG branch is awaited first, then the image/naive branch is awaited second.
- Why: the parallel branches doubled peak OpenAI traffic for every query and made `RateLimitError` much more likely during batch runs.

### 2. Added a regression test for sequential two-step execution

- File: `MegaRAG/test/test_two_step_query.py`
- Change: added an async unit test that verifies `kg_two_step_query()` executes the KG branch before the second branch.
- Why: this locks in the reduced-concurrency behavior and prevents accidental reintroduction of parallel fan-out.

### 3. Lowered query-time defaults to reduce API usage

- File: `MegaRAG/egs/utils/query_mmkg.py`
- Changes:
  - `--max-retries` default changed from `3` to `1`
  - `--retry-delay` default changed from `3.0` seconds to `1.0` second
  - `--concurrency` default changed from `4` to `1`
  - default query mode changed from `mix_two_step` to `global`
  - default `chunk_top_k` lowered from `6` to `2`
- Why: this keeps graph reasoning enabled while minimizing duplicate requests and retry cost.

### 4. Lowered Kaggle example config to the minimum practical load

- File: `MegaRAG/egs/world_history_tiny/conf/addon_params.yaml`
- Changes:
  - added `llm_model_max_async: 1`
  - changed `chunk_top_k` from `6` to `2`
  - set `query_mode: global`
  - changed `embed_parallel_limit` from `12` to `1`
- Why: this reduces parallel embedding calls and keeps graph query load as low as possible for Kaggle/API budget constraints.

### 5. Final operating mode

- Current behavior: graph reasoning is still used, but only through the lighter `global` query path.
- Not used by default: `mix_two_step`, because it adds an extra naive branch and increases OpenAI usage significantly.
- Practical result: lower API cost, fewer 429s, and less wasted retry traffic.

## 2026-08-22

### 1. Resolved MMKG Construction 429 errors
- File: `MegaRAG/egs/utils/construct_mmkg.py`
- Change: Refactored `initialize_rag` to unpack and pass all parameters (such as `llm_model_max_async: 1`) from `addon_params.yaml` directly to the `MegaRAG` constructor.
- Why: Previously, custom parameters from the yaml file were not forwarded to the MegaRAG instance, causing graph construction to run with a default concurrency of 4 and exceed the TPM limit.

### 2. Enhanced API retry strategy & randomized jitter
- File: `MegaRAG/megarag/llms/openai.py`
- Change: Modified `@retry` on `openai_complete_if_cache` to run up to 6 times (up from 3) with an exponential wait up to 60 seconds (up from 10) combined with a random jitter (`wait_random`).
- Why: Gives the sliding 60-second OpenAI rate limit window sufficient time to cool down and prevents synchronized request retry bursts.

### 3. Integrated low-detail vision token footprint reduction
- Files: `MegaRAG/megarag/llms/openai.py`, `MegaRAG/egs/utils/construct_mmkg.py`, `MegaRAG/egs/utils/query_mmkg.py`
- Change: Exposed an `image_detail` parameter (defaulting to `low` in `addon_params.yaml`) to specify `"detail": image_detail` in OpenAI image payloads.
- Why: Drops image token usage from 400+ down to exactly 85 tokens per document page/figure image.

### 4. Added pacing, resume checkpointing, and cache disk persistence to querying
- File: `MegaRAG/egs/utils/query_mmkg.py`
- Change: 
  - Added `--query-delay` (default 5s) to pace sequential requests.
  - Added `--resume` flag to skip questions with existing successful responses in output.
  - Explicitly called `await rag.llm_response_cache.index_done_callback()` at the end of `async_main`.
- Why: Pacing protects TPM limits, resume prevents costly reprocessing of finished items if interrupted, and cache persistence ensures subsequent runs hit the cache instantly with zero token consumption.

### 5. Configured defaults & updated Kaggle notebook invocation
- Files: `MegaRAG/egs/world_history_tiny/conf/addon_params.yaml`, `megarag-mmkg-kaggle.ipynb`
- Change: Added token parameters and `image_detail` to the yaml config, modified Section 7 cell command invocation to pass query pacing, retry, and resume CLI arguments, and updated Section 1 repository setup cell to run `git pull` if the repository folder already exists.
- Why: Forces the Kaggle environment to fetch the latest codebase changes from GitHub on every execution instead of skipping updates and using outdated cached files.

## 2026-08-25

Integrated the Vietnamese-document QA flow (`archive_notebooks/megarag-vietnamdoc-mmkg-qwen3.ipynb`, local Qwen3-VL) as **Stage 2** of the production notebook `megarag-mmkg-kaggle-v2.ipynb`, and eliminated all notebook-runtime code generation/patching by moving logic into version-controlled files under `MegaRAG/`.

### 1. Promoted notebook-generated scripts into real repo files

- Source: `%%writefile` payload cells in `archive_notebooks/megarag-vietnamdoc-mmkg-qwen3.ipynb` (cells 5, 7, 9, 11, 24), extracted byte-for-byte (UTF-8 preserved).
- Files added:
  - `MegaRAG/egs/utils/qwen_llm.py` — async local Qwen3-VL wrapper (`qwen_gpt_4o_mini_complete`) compatible with the LightRAG/MegaRAG `llm_func` interface; lazy model load, generation lock, OOM fallback without images.
  - `MegaRAG/egs/utils/ocr_pages.py` — OCR page images via Qwen3-VL.
  - `MegaRAG/egs/utils/baseline_direct.py` — no-RAG baseline QA on page images, with resume support.
  - `MegaRAG/egs/utils/judge_qwen.py` — LLM-as-judge (YES/NO verdict) evaluation.
  - `MegaRAG/egs/utils/gme_compat.py` — GME-Qwen2-VL-2B embedding wrapper compatible with transformers >= 4.57; exposes `get_text_embeddings` / `get_image_embeddings` / `get_fused_embeddings` so it is a drop-in replacement consumed by the existing `hf_gme_embed`.
  - `MegaRAG/egs/vdoc/conf/addon_params.yaml` — vdoc recipe config (Vietnamese entity types, batch size 1, reduced token budgets).
- Why: Code written at runtime via `%%writefile` cannot be linted, tested, reviewed, or diffed; it also drifted between the notebook and the repo.

### 2. Removed runtime regex patching of upstream files

- Files: `MegaRAG/egs/utils/construct_mmkg.py`, `MegaRAG/egs/utils/query_mmkg.py`
- Change: Replaced the old approach (notebook cells regex-patching imports and rewriting `initialize_model()` after cloning) with first-class backend switches:
  - `MEGARAG_LLM_BACKEND=openai|qwen_local` (default `openai`) resolved lazily in `_resolve_llm_complete()`.
  - `MEGARAG_EMBED_BACKEND=hf_gme|gme_compat` (default `hf_gme`) resolved in `initialize_model()`.
- Compatibility: defaults preserve Stage 1 behavior exactly; diff is ~+17 lines per file.
- Why: Regex patching failed silently whenever upstream formatting changed, was invisible to review/version control, and made reproduction fragile.

### 3. Added `vdoc_pipeline` Python package (data prep + evaluation)

- Files added under `MegaRAG/vdoc_pipeline/`: `settings.py`, `data_prep.py`, `run_ocr.py`, `make_queries.py`, `metrics.py`, `cli.py`, `__main__.py`, `__init__.py`.
- Details:
  - `settings.py` centralizes all paths/constants previously scattered as hardcoded `/kaggle/...` strings across notebook cells; overridable via `VDOC_*` env vars.
  - `data_prep.py` downloads `TranNhiem/Vietnamese-DocumentImage-Reasoning`, selects top pages by question count, writes `pages_content_preocr.json` + `eval_items.jsonl` (sampling controlled by `SAMPLE_PAGES`, `MAX_QUESTIONS_PER_PAGE`, `MAX_EVAL_QUESTIONS`).
  - `run_ocr.py` orchestrates Qwen3-VL OCR as a clean subprocess, or writes placeholder text when `USE_OCR=0`.
  - `metrics.py` splits the EM / Token-F1 / Token-Recall computation into pure testable functions plus a CLI producing `metrics.json` and `predictions.csv` comparing MegaRAG vs baseline (+ LLM-judge accuracy).
  - `cli.py` exposes one entrypoint: `python -m vdoc_pipeline {prep|ocr|queries|build|query|baseline|judge|metrics|all}` with subprocess stages streaming logs live; sets backend env defaults; baseline stage spreads Qwen3-VL across all visible GPUs.
- Why: Makes the pipeline runnable headless/CI without the notebook, removes duplicated orchestration, and gives every artifact (metrics, predictions) a stable location under `VDOC_OUT_DIR`.

### 4. Appended Stage 2 (Section 9) to the production notebook

- File: `megarag-mmkg-kaggle-v2.ipynb` (cells 28–44 appended; backup of previous version kept).
- Cells added:
  - Section intro documenting the flow and prerequisites.
  - Config cell: `VDOC_TMP_DIR=/kaggle/tmp/megarag_vdoc`, `VDOC_OUT_DIR=<repo>/vdoc_outputs`, HF caches on `/kaggle/tmp`, sampling flags, optional `RUN_BASELINE` / `RUN_LLM_JUDGE` / `JUDGE_SAMPLE`, Qwen settings (`QWEN_MODEL_ID=unsloth/Qwen3-VL-8B-Instruct-bnb-4bit`, max pixels/input chars), backend env switches, and a reusable `run_stage()` helper that streams subprocess logs into the notebook.
  - transformers upgrade cell: installs `transformers>=4.57,<5` plus Qwen3-VL deps (`bitsandbytes`, `qwen_vl_utils`, ...) and hard-fails if < 4.57.
  - One thin cell per stage: prep → OCR → build MMKG (removes stale `egs/vdoc/exp` index first) → query → optional baseline/judge → metrics + per-question preview.
  - Optional download cell zipping `vdoc_outputs` as `vdoc_results.zip` (reuses the existing `download_kaggle_working()` helper).
- Known constraint documented in-notebook: Stage 1 pins `transformers==4.51.3` for MinerU while Qwen3-VL requires >= 4.57, so all MinerU steps must complete before the upgrade cell, and must not be re-run afterwards in the same session.
- Why: Keeps the running notebook as a thin launcher while all logic lives in the repo.

### 5. Verification performed

- `py_compile` passed for all new/modified Python files.
- Notebook JSON revalidated (nbformat 4); every code cell AST-parses (magics excluded).
- Unit checks passed for `normalize_answer` / `token_f1` / `token_recall` and settings env overrides.
- `python -m vdoc_pipeline --help` and stage routing verified locally (pandas-dependent metrics stage verified structurally only, since pandas is not installed in the local environment).

