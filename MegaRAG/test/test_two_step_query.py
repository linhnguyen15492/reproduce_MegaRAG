import asyncio
from unittest.mock import patch

import pytest

from lightrag.base import QueryParam
from megarag.operate import kg_two_step_query


class _DummyHashKV:
    global_config: dict = {}


@pytest.mark.asyncio
async def test_kg_two_step_query_runs_branches_sequentially():
    events: list[str] = []

    async def fake_kg_query(*args, **kwargs):
        events.append("kg_start")
        await asyncio.sleep(0)
        events.append("kg_end")
        return "kg-answer"

    async def fake_naive_query(*args, **kwargs):
        assert events == ["kg_start", "kg_end"]
        events.append("naive_start")
        await asyncio.sleep(0)
        events.append("naive_end")
        return "naive-answer"

    async def fake_llm(prompt, **kwargs):
        return "final-answer"

    async def fake_handle_cache(*args, **kwargs):
        return None, None, None, None

    query_param = QueryParam(mode="mix_two_step", chunk_top_k=1, enable_rerank=False)
    global_config = {"llm_model_func": fake_llm}

    with (
        patch("megarag.operate.kg_query", fake_kg_query),
        patch("megarag.operate.naive_query", fake_naive_query),
        patch("megarag.operate.handle_cache", fake_handle_cache),
    ):
        result = await kg_two_step_query(
            query="test query",
            knowledge_graph_inst=object(),
            entities_vdb=object(),
            relationships_vdb=object(),
            text_chunks_db=object(),
            query_param=query_param,
            global_config=global_config,
            hashing_kv=_DummyHashKV(),
        )

    assert result == "final-answer"
    assert events == ["kg_start", "kg_end", "naive_start", "naive_end"]
