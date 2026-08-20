# Implementation Plan - Resolve OpenAI API 429 Rate Limit Errors in MegaRAG Querying

This plan details technical improvements proposed to resolve and prevent OpenAI API `RateLimitError` (429) failures when running queries on the Multimodal Knowledge Graph (MMKG) in Kaggle/Colab environments, building upon previous sequential execution updates.

## User Review Required

> [!IMPORTANT]
> The default token budgets in LightRAG/MegaRAG allow prompts to grow up to **32,000 tokens** per request. Under Tier 1 or free-tier OpenAI accounts, a few concurrent or back-to-back requests of this size will immediately trigger the 200,000 TPM (Tokens Per Minute) limit. We propose lowering the token budgets in the configuration and introducing query pacing to keep usage within rate limits.

---

## Technical Analysis of the 429 Error

1. **Large Prompt Size**: The average prompt token usage recorded was ~79,306 tokens per LLM call (combining keyword extraction and final answer generation). High-volume prompts quickly exhaust the OpenAI Tokens Per Minute (TPM) limit.
2. **Lack of Query-Level Retries**: In `query_mmkg.py`, `--max-retries` defaults to `1`. If an LLM call fails (e.g. after internal retries), the query script fails immediately for that question without retrying at the task level.
3. **Thundering Herd / Continuous Exhaustion**: When queries are processed sequentially but with no delay between them, the rate limit window never has time to reset.
4. **Insufficient Retry Wait Times**: The internal OpenAI completion wrapper uses a maximum exponential backoff wait of only `10 seconds`, which is too short to clear the 60-second sliding rate limit window.

---

## Proposed Changes

### 1. Configuration Tuning (Token & Image Budgets)

#### [MODIFY] [addon_params.yaml](file:///d:/workspace/reproduce_MegaRAG/MegaRAG/egs/world_history_tiny/conf/addon_params.yaml)
* Add configuration overrides to limit the maximum tokens allowed for entities, relations, and the total prompt context.
* Add `image_detail` override to default image quality to `low` (reducing vision tokens per image from 400+ to 85).

```yaml
max_entity_tokens: 4000
max_relation_tokens: 4000
max_total_tokens: 16000
image_detail: low
```

---

### 2. Core Python Updates

#### [MODIFY] [openai.py](file:///d:/workspace/reproduce_MegaRAG/MegaRAG/megarag/llms/openai.py)
* Update the retry decorator for `openai_complete_if_cache` to increase maximum retries and backoff wait times, and add random jitter (`wait_random`) to prevent synchronized requests.
* Expose `image_detail` to `_img_item` to allow low-detail mode for base64 vision images.

```diff
-from tenacity import (
-    retry,
-    stop_after_attempt,
-    wait_exponential,
-    retry_if_exception_type,
-)
+from tenacity import (
+    retry,
+    stop_after_attempt,
+    wait_exponential,
+    wait_random,
+    retry_if_exception_type,
+)
```

```diff
 @retry(
-    stop=stop_after_attempt(3),
-    wait=wait_exponential(multiplier=1, min=4, max=10),
+    stop=stop_after_attempt(6),
+    wait=wait_exponential(multiplier=2, min=5, max=60) + wait_random(0, 2),
     retry=(
         retry_if_exception_type(RateLimitError)
         | retry_if_exception_type(APIConnectionError)
         | retry_if_exception_type(APITimeoutError)
         | retry_if_exception_type(InvalidResponseError)
     ),
 )
 async def openai_complete_if_cache(
```

```diff
     # Strip kwargs meant only for upstream callers
     kwargs.pop("hashing_kv", None)
     kwargs.pop("keyword_extraction", None)
+    image_detail = kwargs.pop("image_detail", "low")
 
     messages: List[dict[str, Any]] = []
     if system_prompt:
         messages.append({"role": "system", "content": system_prompt})
     messages.extend(history_messages)
 
     # Helper: turn a path / URL into the vision content item
     def _img_item(path: str) -> dict[str, Any]:
         if Path(path).is_file():  # local ⇒ convert to data: URL
             mime, _ = mimetypes.guess_type(path)
             with open(path, "rb") as f:
                 b64 = base64.b64encode(f.read()).decode()
             url = f"data:{mime or 'image/png'};base64,{b64}"
         else:                      # already remote
             url = path
-        return {"type": "image_url", "image_url": {"url": url}}
+        return {"type": "image_url", "image_url": {"url": url, "detail": image_detail}}
```

#### [MODIFY] [query_mmkg.py](file:///d:/workspace/reproduce_MegaRAG/MegaRAG/egs/utils/query_mmkg.py)
* Read the newly defined token budget fields from configuration and pass them into `QueryParam`.
* Pass `image_detail` from `addon_params` into `llm_func`.
* Change the default for `--max-retries` from `1` to `3`.
* Add `--query-delay` parameter (defaulting to `5.0` seconds) to enforce a cooldown delay between queries.
* Add a `--resume` parameter to skip queries that already have a successful response recorded in the output file.
* Explicitly persist the `llm_response_cache` to disk at the end of the query execution so cache hits can be reused in future notebook runs.

```diff
     async def llm_func(
         prompt,
         input_images=None,
         system_prompt=None,
         history_messages=None,
         keyword_extraction=False,
         **kwargs,
     ):
         return await gpt_4o_mini_complete(
             prompt=prompt,
             input_images=input_images,
             system_prompt=system_prompt,
             history_messages=history_messages,
             keyword_extraction=keyword_extraction,
             token_tracker=token_tracker,
+            image_detail=addon_params.get("image_detail", "low"),
             **kwargs,
         )
```

```diff
     param = QueryParam(
         mode=addon_params.get("query_mode", "global"),
         chunk_top_k=addon_params.get("chunk_top_k", 2),
         enable_rerank=False,
+        max_entity_tokens=addon_params.get("max_entity_tokens", 4000),
+        max_relation_tokens=addon_params.get("max_relation_tokens", 4000),
+        max_total_tokens=addon_params.get("max_total_tokens", 16000),
     )
```

```diff
-    parser.add_argument("--max-retries", type=int, default=1, metavar="N")
+    parser.add_argument("--max-retries", type=int, default=3, metavar="N")
     parser.add_argument("--retry-delay", type=float, default=1.0, metavar="SECONDS")
+    parser.add_argument(
+        "--query-delay",
+        type=float,
+        default=5.0,
+        metavar="SECONDS",
+        help="Delay between query requests to prevent rate limit limits.",
+    )
+    parser.add_argument(
+        "--resume",
+        action="store_true",
+        help="Skip already processed and successful queries in the output file.",
+    )
```

```diff
     questions = extract_queries(Path(args.input_queries))
     if not questions:
         raise SystemExit(f"No questions found in {args.input_queries}")
 
+    processed_questions = set()
+    out_path = Path(args.output_file)
+    if args.resume and out_path.exists():
+        try:
+            if args.output_format == "jsonl":
+                with out_path.open("r", encoding="utf-8") as f:
+                    for line in f:
+                        if line.strip():
+                            rec = json.loads(line)
+                            if rec.get("question") and not rec.get("error"):
+                                processed_questions.add(rec["question"])
+            else:
+                with out_path.open("r", encoding="utf-8") as f:
+                    payload = json.load(f)
+                    for rec in payload.get("results", []):
+                        if rec.get("question") and not rec.get("error"):
+                            processed_questions.add(rec["question"])
+            print(f"Loaded {len(processed_questions)} already completed queries from output file.")
+        except Exception as e:
+            print(f"⚠️ Resume check failed, running all: {e}")
+
     sem = asyncio.Semaphore(max(1, args.concurrency))
 
     async def worker(idx: int, q: str):
+        if q in processed_questions:
+            # Return placeholder for skipped tasks, handled on merge
+            return idx, None
         started_at = _now_iso()
         async with sem:
             res, dt, err = await run_query_with_retries(
                 rag, q, param, max_retries=args.max_retries, retry_delay=args.retry_delay
             )
+            if args.query_delay > 0:
+                await asyncio.sleep(args.query_delay)
         finished_at = _now_iso()
```

```diff
     for fut in asyncio.as_completed(tasks):
-        idx, rec = await fut
-        finished[idx] = rec
-        if pbar:
-            if rec["error"]:
-                err_count += 1
-                pbar.set_postfix_str(f"errors={err_count}")
-            pbar.update(1)
+        idx, rec = await fut
+        if rec is None:
+            # Query was skipped
+            if pbar:
+                pbar.update(1)
+            continue
+        finished[idx] = rec
+        if pbar:
+            if rec["error"]:
+                err_count += 1
+                pbar.set_postfix_str(f"errors={err_count}")
+            pbar.update(1)
```

```diff
     # If not streaming earlier, print in input order now
     if not args.print_as_complete:
         for idx, rec in enumerate(finished):
+            if rec is None:
+                continue
             if rec["error"]:
```

```diff
+    # Load existing finished records if resuming in JSON format
+    if args.resume and args.output_format == "json" and out_path.exists():
+        try:
+            with out_path.open("r", encoding="utf-8") as f:
+                existing_payload = json.load(f)
+                existing_results = existing_payload.get("results", [])
+                # Merge finished results with existing results
+                for i, r in enumerate(finished):
+                    if r is None:
+                        # Get existing result
+                        match = [x for x in existing_results if x.get("question") == questions[i]]
+                        if match:
+                            finished[i] = match[0]
+        except Exception as e:
+            print(f"⚠️ Failed to merge resume results: {e}")
+
+    # Remove unresolved None records
+    finished = [rec for rec in finished if rec is not None]
```

```diff
     print(f"Final Token Usage: {token_tracker}.")
 
+    # Save LLM Cache back to disk
+    if hasattr(rag, "llm_response_cache") and rag.llm_response_cache:
+        print("Persisting query-time LLM response cache to disk...")
+        await rag.llm_response_cache.index_done_callback()
```

---

### 3. Notebook Execution Updates

#### [MODIFY] [megarag-mmkg-kaggle.ipynb](file:///d:/workspace/reproduce_MegaRAG/megarag-mmkg-kaggle.ipynb)
* Modify the CLI parameters for `query_mmkg.py` to invoke query pacing and resume features:

```bash
/usr/bin/python3 /content/reproduce_MegaRAG/MegaRAG/egs/utils/query_mmkg.py \
    --config-file /content/reproduce_MegaRAG/MegaRAG/egs/world_history_tiny/conf/addon_params.yaml \
    --working-dir /content/reproduce_MegaRAG/World_History_Volume_1_run/exp/World_History_Volume_1 \
    --input-queries /content/reproduce_MegaRAG/data/queries_short.txt \
    --output-file /content/reproduce_MegaRAG/World_History_Volume_1_run/exp/World_History_Volume_1/results/results.json \
    --concurrency 1 \
    --max-retries 3 \
    --query-delay 5.0 \
    --resume
```

---

## Verification Plan

### Automated Tests
* Run `pytest MegaRAG/test/test_two_step_query.py` to ensure core correctness.
* Execute the queries locally on the sample query dataset using `--concurrency 1`, `--max-retries 3`, `--query-delay 5.0`, and `--resume` to verify query completion and retry logging.
* Confirm that repeating the query command hits the persisted `llm_response_cache` and completes instantly with 0 additional OpenAI API token usage.
