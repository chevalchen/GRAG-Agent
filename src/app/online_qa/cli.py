import argparse
import logging
import sqlite3
import sys
import uuid

from src.app.config import Config, DEFAULT_CONFIG, GraphRAGConfig
from src.app.online_qa.graphs.online_qa_graph import build_graph
from src.core.tools.graph.neo4j_tool import Neo4jGraphTool
from src.core.tools.llm.generation_tool import LLMGenerationTool
from src.core.tools.retrieval.bm25_tool import build_bm25_from_milvus
from src.core.tools.vector.milvus_tool import MilvusVectorTool


def _build_system(config: GraphRAGConfig):
    neo4j_tool = Neo4jGraphTool(config)
    milvus_tool = MilvusVectorTool(config)
    _ = milvus_tool.load_collection()
    bm25_tool = build_bm25_from_milvus(
        milvus_tool.client,
        top_k=getattr(config, "bm25_top_k", config.top_k),
        collection_name=milvus_tool.collection_name,
    )
    llm_tool = LLMGenerationTool(config)
    return build_graph(
        config,
        bm25_tool=bm25_tool,
        milvus_tool=milvus_tool,
        neo4j_tool=neo4j_tool,
        llm_tool=llm_tool,
    )


def _print_route_summary(state):
    routing = state.get("routing")
    if not routing:
        return
    intent_icons = {
        "drug_specific": "💊",
        "symptom_disease": "🩺",
        "tcm_theory": "📚",
        "clinical_case": "🧪",
        "health_advice": "🌿",
    }
    intent = routing.get("query_intent", "unknown")
    icon = intent_icons.get(intent, "❓")
    print(f"{icon} 识别意图: {intent}")
    print(
        f"🧭 路由: graph={routing.get('use_graph')}, drug={routing.get('use_drug_vec')}, "
        f"lit={routing.get('use_lit_vec')}, health={routing.get('use_health_vec')}"
    )


def _print_docs_summary(state, limit: int = 3):
    docs = state.get("docs_final") or []
    if not docs:
        return
    items = []
    for doc in docs[:limit]:
        recipe_name = (
            doc.metadata.get("drug_name")
            or doc.metadata.get("recipe_name")
            or doc.metadata.get("source")
            or "未知内容"
        )
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
    time_keys = ["analyze_seconds", "retrieve_seconds", "fuse_seconds", "rerank_seconds", "generate_seconds", "total_seconds"]
    count_keys = ["graph_rows", "graph_docs"]
    parts = []
    for k in time_keys:
        if k in metrics:
            parts.append(f"{k}={metrics[k]:.3f}s")
    for k in count_keys:
        if k in metrics:
            parts.append(f"{k}={int(metrics[k])}")
    if parts:
        print("⏱️ " + ", ".join(parts))


def _ask_once(
    online_graph,
    query: str,
    stream: bool,
    verbose: bool,
    show_metrics: bool,
    session_id: str | None,
):
    sid = session_id or str(uuid.uuid4())
    state = online_graph.invoke(
        {"query": query, "metrics": {"stream": stream}},
        config={"configurable": {"thread_id": sid}},
    )
    docs = state.get("docs_final", [])
    if verbose:
        _print_route_summary(state)
        _print_docs_summary(state)
        if show_metrics:
            _print_metrics(state)
    if stream:
        answer = state.get("answer") or ""
        print("回答:")
        for ch in answer:
            print(ch, end="", flush=True)
        print()
        return
    print("回答:")
    print(state.get("answer") or "")


def _list_history_sessions(db_path: str = ".checkpoints/tcm.db") -> list[tuple[str, str]]:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT c.thread_id, c.checkpoint_id
            FROM checkpoints c
            JOIN (
                SELECT thread_id, MAX(rowid) AS last_rowid
                FROM checkpoints
                GROUP BY thread_id
            ) last ON last.thread_id = c.thread_id AND last.last_rowid = c.rowid
            ORDER BY c.rowid DESC
            """
        )
        rows = cur.fetchall() or []
        conn.close()
    except Exception:
        return []
    sessions = []
    for thread_id, checkpoint_id in rows:
        sessions.append((str(thread_id), str(checkpoint_id)))
    return sessions


def _delete_session(db_path: str, session_id: str) -> bool:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("DELETE FROM writes WHERE thread_id = ?", (session_id,))
        cur.execute("DELETE FROM checkpoints WHERE thread_id = ?", (session_id,))
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


def _choose_session_id(cli_session_id: str) -> str:
    if cli_session_id and cli_session_id != "default":
        return cli_session_id
    configured = Config.SESSION_ID
    if configured:
        use_it = input(f"检测到配置的会话：{configured}，是否使用？(y/n) ").strip().lower()
        if use_it == "y":
            return configured
    db_path = Config.CHECKPOINTER_PATH
    while True:
        sessions = _list_history_sessions(db_path=db_path)
        print("历史会话列表（按最近使用排序）：")
        for idx, (sid, last_checkpoint_id) in enumerate(sessions, start=1):
            print(f"  {idx}. {sid}（最后 checkpoint：{last_checkpoint_id}）")
        print("  0. 创建新会话")
        print("  d. 删除会话")
        print("  q. 退出")
        choice = input("请选择会话编号：").strip().lower()
        if choice == "q":
            raise SystemExit(0)
        if choice == "d":
            if not sessions:
                continue
            target = input("请输入要删除的会话编号或 session_id：").strip()
            if not target:
                continue
            target_sid = target
            if target.isdigit():
                idx = int(target)
                if idx <= 0 or idx > len(sessions):
                    continue
                target_sid = sessions[idx - 1][0]
            confirm = input(f"确认删除会话 {target_sid}？(y/n) ").strip().lower()
            if confirm != "y":
                continue
            ok = _delete_session(db_path=db_path, session_id=target_sid)
            if ok:
                print(f"已删除会话：{target_sid}")
            else:
                print("删除失败。")
            continue
        if choice == "0" or not sessions:
            sid = input("请输入新 session_id（留空自动生成）：").strip() or str(uuid.uuid4())
            print(f"如需下次自动使用此会话，请在 .env 中设置：\nC9_SESSION_ID={sid}")
            return sid
        try:
            selected = sessions[int(choice) - 1][0]
        except Exception:
            selected = sessions[0][0]
        print(f"如需下次自动使用此会话，请在 .env 中设置：\nC9_SESSION_ID={selected}")
        return selected


def run_history_mode(system, args, verbose_enabled: bool) -> None:
    session_id = _choose_session_id(args.session_id)
    while True:
        q = input("\n请输入您的问题: ").strip()
        if not q:
            continue
        if q.lower() in {"quit", "exit"}:
            break
        _ask_once(
            system,
            q,
            args.stream,
            verbose_enabled,
            args.show_metrics,
            session_id,
        )


def run_single_mode(system, args, verbose_enabled: bool) -> None:
    q = input("\n请输入问题: ").strip()
    if not q:
        return
    _ask_once(
        system,
        q,
        args.stream,
        verbose_enabled,
        args.show_metrics,
        None,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="online_qa", description="LangGraph 在线问答")
    parser.add_argument("--top-k", type=int, default=DEFAULT_CONFIG.top_k)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--session-id", type=str, default="default")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--show-metrics", action="store_true")
    parser.add_argument("--log-level", type=str, default="WARN", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args(argv)

    log_level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    logging.getLogger("src").setLevel(log_level)
    logging.getLogger("src.legacy").setLevel(log_level)
    logging.getLogger("src.app").setLevel(log_level)
    verbose_enabled = True

    config = GraphRAGConfig(
        **{
            **DEFAULT_CONFIG.to_dict(),
            "top_k": args.top_k,
            "neo4j_uri": Config.NEO4J_URI,
            "neo4j_user": Config.NEO4J_USER,
            "neo4j_password": Config.NEO4J_PASSWORD,
            "neo4j_database": Config.NEO4J_DATABASE,
            "milvus_host": Config.MILVUS_HOST,
            "milvus_port": Config.MILVUS_PORT,
            "milvus_uri": Config.MILVUS_URI,
            "milvus_collection_name": Config.MILVUS_COLLECTION_NAME,
            "embedding_model": Config.FAST_EMBEDDING_MODEL if Config.FAST_MODE else Config.EMBEDDING_MODEL,
            "llm_model": Config.FAST_LLM_MODEL if Config.FAST_MODE else Config.LLM_MODEL,
            "reranker_model": Config.FAST_RERANKER_MODEL if Config.FAST_MODE else Config.RERANKER_MODEL,
            "bm25_top_k": Config.BM25_TOP_K,
            "retrieval_balance_strategy": Config.RETRIEVAL_BALANCE_STRATEGY,
            "retrieval_timeout_seconds": Config.RETRIEVAL_TIMEOUT_SECONDS,
            "retrieve_expand_factor": Config.RETRIEVE_EXPAND_FACTOR,
            "lit_expand_factor": Config.LIT_EXPAND_FACTOR,
            "graph_expand_factor": Config.GRAPH_EXPAND_FACTOR,
            "max_retrieval_top_k": Config.MAX_RETRIEVAL_TOP_K,
            "max_graph_rows": Config.MAX_GRAPH_ROWS,
            "graph_fallback_max_nodes": Config.GRAPH_FALLBACK_MAX_NODES,
            "rerank_simple_skip_threshold": Config.RERANK_SIMPLE_SKIP_THRESHOLD,
            "rerank_simple_candidate_limit": Config.RERANK_SIMPLE_CANDIDATE_LIMIT,
            "rerank_complex_candidate_limit": Config.RERANK_COMPLEX_CANDIDATE_LIMIT,
            "simple_context_budget_chars": Config.SIMPLE_CONTEXT_BUDGET_CHARS,
            "complex_context_budget_chars": Config.COMPLEX_CONTEXT_BUDGET_CHARS,
            "simple_per_doc_chars": Config.SIMPLE_PER_DOC_CHARS,
            "complex_per_doc_chars": Config.COMPLEX_PER_DOC_CHARS,
            "checkpointer_path": Config.CHECKPOINTER_PATH,
        }
    )
    online_graph = _build_system(config)

    if args.query:
        effective_session_id = args.session_id
        if effective_session_id == "default" and Config.SESSION_ID:
            effective_session_id = Config.SESSION_ID
        _ask_once(
            online_graph,
            args.query,
            args.stream,
            verbose_enabled,
            args.show_metrics,
            effective_session_id,
        )
        return

    print("请选择模式：")
    print("  1. 历史对话")
    print("  2. 临时对话（不保存历史）")
    choice = input("请输入选项 (1/2)：").strip()
    MODES = {
        "1": run_history_mode,
        "2": run_single_mode,
    }
    runner = MODES.get(choice, run_history_mode)
    runner(online_graph, args, verbose_enabled)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
