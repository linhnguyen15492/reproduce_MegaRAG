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
