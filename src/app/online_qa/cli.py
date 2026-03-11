import argparse
import asyncio
import logging

from src.app.config import DEFAULT_CONFIG, GraphRAGConfig
from src.app.online_qa.graphs.online_qa_graph import OnlineQAGraph
from src.app.online_qa.tools.answer_generation import AnswerGenerationTool
from src.app.online_qa.tools.graph_rag_search import GraphRAGSearchTool
from src.app.online_qa.tools.hybrid_search import HybridSearchTool
from src.legacy.rag_modules.generation_integration import GenerationIntegrationModule
from src.legacy.rag_modules.graph_data_preparation import GraphDataPreparationModule
from src.legacy.rag_modules.graph_rag_retrieval import GraphRAGRetrieval
from src.legacy.rag_modules.hybrid_retrieval import HybridRetrievalModule
from src.legacy.rag_modules.intelligent_query_router import IntelligentQueryRouter
from src.legacy.rag_modules.milvus_index_construction import MilvusIndexConstructionModule


def _build_system(config: GraphRAGConfig):
    data_module = GraphDataPreparationModule(
        uri=config.neo4j_uri,
        user=config.neo4j_user,
        password=config.neo4j_password,
        database=config.neo4j_database,
    )
    index_module = MilvusIndexConstructionModule(
        host=config.milvus_host,
        port=config.milvus_port,
        collection_name=config.milvus_collection_name,
        dimension=config.milvus_dimension,
        model_name=config.embedding_model,
    )
    generation_module = GenerationIntegrationModule(
        model_name=config.llm_model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    traditional_retrieval = HybridRetrievalModule(
        config=config,
        milvus_module=index_module,
        data_module=data_module,
        llm_client=generation_module.client,
    )
    graph_rag_retrieval = GraphRAGRetrieval(config=config, llm_client=generation_module.client)
    query_router = IntelligentQueryRouter(
        traditional_retrieval=traditional_retrieval,
        graph_rag_retrieval=graph_rag_retrieval,
        llm_client=generation_module.client,
        config=config,
    )

    online_graph = OnlineQAGraph(
        router=query_router,
        hybrid_tool=HybridSearchTool(traditional_retrieval),
        graph_tool=GraphRAGSearchTool(graph_rag_retrieval),
        answer_tool=AnswerGenerationTool(generation_module),
        top_k=config.top_k,
    )
    return (
        data_module,
        index_module,
        generation_module,
        online_graph,
        query_router,
        traditional_retrieval,
        graph_rag_retrieval,
    )


def _ensure_kb(config: GraphRAGConfig, data_module, index_module):
    if index_module.has_collection() and index_module.load_collection():
        data_module.load_graph_data()
        data_module.build_recipe_documents()
        chunks = data_module.chunk_documents(config.chunk_size, config.chunk_overlap)
        return chunks

    data_module.load_graph_data()
    data_module.build_recipe_documents()
    chunks = data_module.chunk_documents(config.chunk_size, config.chunk_overlap)
    index_module.build_vector_index(chunks)
    return chunks


def _print_route_summary(state):
    analysis = state.get("analysis")
    if not analysis:
        return
    strategy_icons = {
        "hybrid_traditional": "🔍",
        "graph_rag": "🕸️",
        "combined": "🔄",
    }
    strategy = analysis.recommended_strategy.value
    icon = strategy_icons.get(strategy, "❓")
    print(f"{icon} 使用策略: {strategy}")
    print(f"📊 复杂度: {analysis.query_complexity:.2f}, 关系密集度: {analysis.relationship_intensity:.2f}")


def _print_docs_summary(state, limit: int = 3):
    docs = state.get("docs_final") or []
    if not docs:
        return
    items = []
    for doc in docs[:limit]:
        recipe_name = doc.metadata.get("recipe_name", "未知内容")
        search_type = doc.metadata.get("search_type") or doc.metadata.get("search_source") or doc.metadata.get("route_strategy") or "unknown"
        score = doc.metadata.get("final_score", doc.metadata.get("relevance_score", doc.metadata.get("score", 0.0)))
        try:
            score_str = f"{float(score):.3f}"
        except Exception:
            score_str = str(score)
        items.append(f"{recipe_name}({search_type}, {score_str})")
    print(f"📋 找到 {len(docs)} 个相关文档: {', '.join(items)}")
    if len(docs) > limit:
        print(f"    等 {len(docs)} 个结果...")


def _print_metrics(state):
    metrics = state.get("metrics") or {}
    keys = ["analyze_seconds", "retrieve_seconds", "fuse_seconds", "generate_seconds"]
    parts = []
    for k in keys:
        if k in metrics:
            parts.append(f"{k}={metrics[k]:.3f}s")
    if parts:
        print("⏱️ " + ", ".join(parts))


async def _ask_once(online_graph: OnlineQAGraph, generation_module, query: str, stream: bool, verbose: bool, show_metrics: bool):
    state = await online_graph.ainvoke(query, stream=stream)
    docs = state.get("docs_final", [])
    if verbose:
        _print_route_summary(state)
        _print_docs_summary(state)
        if show_metrics:
            _print_metrics(state)
    if stream:
        print("回答:")
        for chunk in generation_module.generate_adaptive_answer_stream(query, docs):
            print(chunk, end="", flush=True)
        print()
        return
    print("回答:")
    print(state.get("answer") or "")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="online_qa", description="LangGraph 在线问答")
    parser.add_argument("--top-k", type=int, default=DEFAULT_CONFIG.top_k)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--show-metrics", action="store_true")
    parser.add_argument("--log-level", type=str, default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    config = GraphRAGConfig(**{**DEFAULT_CONFIG.to_dict(), "top_k": args.top_k})
    (
        data_module,
        index_module,
        generation_module,
        online_graph,
        _,
        traditional_retrieval,
        graph_rag_retrieval,
    ) = _build_system(config)
    chunks = _ensure_kb(config, data_module, index_module)
    traditional_retrieval.initialize(chunks)
    graph_rag_retrieval.initialize()

    if args.query:
        asyncio.run(_ask_once(online_graph, generation_module, args.query, args.stream, args.verbose, args.show_metrics))
        return

    while True:
        q = input("\n您的问题(quit退出): ").strip()
        if not q:
            continue
        if q.lower() == "quit":
            break
        asyncio.run(_ask_once(online_graph, generation_module, q, args.stream, args.verbose, args.show_metrics))


if __name__ == "__main__":
    main()
