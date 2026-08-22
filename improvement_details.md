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

