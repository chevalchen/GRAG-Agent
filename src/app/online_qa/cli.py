import argparse
import logging
import sqlite3
import sys
import uuid

from langchain_core.documents import Document

from src.app.config import Config, DEFAULT_CONFIG, GraphRAGConfig
from src.app.online_qa.graphs.online_qa_graph import build_graph
from src.core.tools.graph.neo4j_tool import Neo4jGraphTool
from src.core.tools.llm.generation_tool import LLMGenerationTool
from src.core.tools.retrieval.bm25_tool import BM25Tool
from src.core.tools.vector.milvus_tool import MilvusVectorTool


def _load_bm25_chunks(config: GraphRAGConfig, *, limit: int = 5000) -> list[Document]:
    try:
        from langchain_community.graphs import Neo4jGraph

        graph = Neo4jGraph(
            url=config.neo4j_uri,
            username=config.neo4j_user,
            password=config.neo4j_password,
            database=config.neo4j_database,
        )
        rows = graph.query(
            """
            MATCH (r:Concept {conceptType: "Recipe"})
            OPTIONAL MATCH (r)-[]->(i:Concept {conceptType: "Ingredient"})
            OPTIONAL MATCH (r)-[]->(s:Concept {conceptType: "CookingStep"})
            WITH r,
                 collect(DISTINCT i.name)[..50] AS ingredients,
                 collect(DISTINCT s.description)[..50] AS steps
            RETURN r.nodeId AS node_id,
                   r.name AS recipe_name,
                   r.category AS category,
                   r.cuisineType AS cuisine_type,
                   r.difficulty AS difficulty,
                   ingredients AS ingredients,
                   steps AS steps
            LIMIT $limit
            """,
            {"limit": int(limit)},
        )
    except Exception:
        return []

    docs: list[Document] = []
    chunk_size = int(config.chunk_size)
    chunk_overlap = int(config.chunk_overlap)
    if chunk_overlap < 0:
        chunk_overlap = 0
    if chunk_overlap >= chunk_size:
        chunk_overlap = 0
    for row in rows or []:
        recipe_name = row.get("recipe_name") or "未知菜谱"
        ingredients = row.get("ingredients") or []
        steps = row.get("steps") or []
        full_text = "\n".join(
            [
                f"# {recipe_name}",
                "",
                "## 所需食材",
                *[f"- {x}" for x in ingredients if x],
                "",
                "## 关键步骤",
                *[f"- {x}" for x in steps if x],
            ]
        ).strip()
        base_meta = {
            "node_id": row.get("node_id") or "",
            "node_type": "Recipe",
            "recipe_name": recipe_name,
            "category": row.get("category") or "",
            "cuisine_type": row.get("cuisine_type") or "",
            "difficulty": row.get("difficulty") or 0,
            "doc_type": "recipe",
        }
        step = max(1, chunk_size - chunk_overlap)
        chunks = [full_text[i : i + chunk_size] for i in range(0, len(full_text), step)] or [full_text]
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{base_meta['node_id']}_{idx}"
            docs.append(Document(page_content=chunk, metadata={**base_meta, "chunk_id": chunk_id, "parent_id": base_meta["node_id"]}))
    return docs


def _build_system(config: GraphRAGConfig):
    neo4j_tool = Neo4jGraphTool(config)
    milvus_tool = MilvusVectorTool(config)
    _ = milvus_tool.load_collection()
    bm25_docs = _load_bm25_chunks(config)
    bm25_tool = BM25Tool(bm25_docs)
    llm_tool = LLMGenerationTool(config)
    return build_graph(
        config,
        bm25_tool=bm25_tool,
        milvus_tool=milvus_tool,
        neo4j_tool=neo4j_tool,
        llm_tool=llm_tool,
    )


def _print_route_summary(state):
    analysis = state.get("analysis")
    if not analysis:
        return
    strategy_icons = {
        "hybrid": "🔍",
        "graph_rag": "🕸️",
        "combined": "🔄",
    }
    strategy = analysis.recommended_strategy
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


def _list_history_sessions(db_path: str = ".checkpoints/c9.db") -> list[tuple[str, str]]:
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
    db_path = ".checkpoints/c9.db"
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

    config = GraphRAGConfig(**{**DEFAULT_CONFIG.to_dict(), "top_k": args.top_k})
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
    print("  2. 单次对话（不保存历史）")
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
