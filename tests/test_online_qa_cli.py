import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document
from langgraph.checkpoint.memory import MemorySaver

from src.app.config import GraphRAGConfig
from src.app.online_qa import cli
from src.app.online_qa.graphs import online_qa_graph
from src.app.online_qa.nodes.rerank_node import make_rerank_node


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("has_docs", [False, True])
def test_cli_query_runs_async_answer_and_saves_session(monkeypatch, capsys, stream, has_docs):
    # 禁止构建真实外部工具；保留 asyncio 在 Windows 上使用的本机 socketpair。
    for tool in (cli.Neo4jGraphTool, cli.MilvusVectorTool, cli.LLMGenerationTool):
        monkeypatch.setattr(
            tool, "__init__", Mock(side_effect=AssertionError("禁止构建真实外部工具"))
        )
    monkeypatch.setattr(online_qa_graph, "get_checkpointer", lambda _: MemorySaver())
    monkeypatch.setattr(
        online_qa_graph,
        "make_rerank_node",
        lambda *args, **kwargs: make_rerank_node(
            *args, **kwargs, cross_encoder_model=Mock()
        ),
    )
    docs = (
        [Document(page_content="合成测试资料", metadata={"node_id": "test-drug"})]
        if has_docs else []
    )
    bm25 = SimpleNamespace(invoke=Mock(return_value=[]))
    milvus = SimpleNamespace(invoke=Mock(return_value=docs))
    neo4j = SimpleNamespace(_graph=SimpleNamespace(query=Mock(return_value=[])))
    prompts = []

    async def astream(messages):
        prompts.append(messages[0].content)
        yield SimpleNamespace(content="测试")
        yield SimpleNamespace(content="回答")

    def invoke_text(prompt):
        if "JSON schema" in prompt:
            return json.dumps({"intent": "theory", "complexity_level": "simple"})
        if "充足性评估器" in prompt:
            return "sufficient"
        return '[]'

    llm = SimpleNamespace(invoke_text=invoke_text, _llm=SimpleNamespace(astream=astream))
    graph = online_qa_graph.build_graph(
        GraphRAGConfig(), bm25_tool=bm25, milvus_tool=milvus,
        neo4j_tool=neo4j, llm_tool=llm,
    )
    monkeypatch.setattr(cli, "_build_system", lambda _: graph)
    args = ["--query", "合成测试问题", "--session-id", "cli-test", "--log-level", "ERROR"]
    if stream:
        args.append("--stream")
    cli.main(args)

    output = capsys.readouterr().out
    expected = "测试回答" if has_docs else "抱歉，没有检索到可用的中医药资料"
    assert expected in output
    state = graph.get_state({"configurable": {"thread_id": "cli-test"}}).values
    assert state["history"][-1]["content"] == state["answer"]
    assert state["history"][-2] == {"role": "user", "content": "合成测试问题"}
    assert state["metrics"]["fuse_ready"] is True
    assert state["metrics"]["rerank_mode"] == "skipped"
    assert bm25.invoke.call_count == milvus.invoke.call_count == (1 if has_docs else 2)
    assert neo4j._graph.query.call_count == (1 if has_docs else 0)
    if has_docs:
        assert "合成测试资料" in prompts[0]
    else:
        assert prompts == []
        assert state["iteration_count"] == 1
