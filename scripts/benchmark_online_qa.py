from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid
from pathlib import Path

from src.app.config import Config, DEFAULT_CONFIG, GraphRAGConfig
from src.app.online_qa.cli import _build_system


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    values_sorted = sorted(float(v) for v in values)
    rank = max(0, min(len(values_sorted) - 1, int(round((len(values_sorted) - 1) * p))))
    return float(values_sorted[rank])


def _build_config(top_k: int) -> GraphRAGConfig:
    return GraphRAGConfig(
        **{
            **DEFAULT_CONFIG.to_dict(),
            "top_k": top_k,
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


def _load_queries(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, list):
            return [str(x).strip() for x in payload if str(x).strip()]
        if isinstance(payload, dict) and isinstance(payload.get("queries"), list):
            return [str(x).strip() for x in payload.get("queries") if str(x).strip()]
    return [line.strip() for line in text.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="在线问答性能基准")
    parser.add_argument("--queries-file", type=str, required=True, help="问题集文件（txt/json）")
    parser.add_argument("--runs", type=int, default=1, help="问题集重复轮数")
    parser.add_argument("--top-k", type=int, default=DEFAULT_CONFIG.top_k)
    parser.add_argument("--p95-target", type=float, default=6.0)
    parser.add_argument("--output", type=str, default="", help="可选：输出结果 JSON 路径")
    args = parser.parse_args()

    queries = _load_queries(Path(args.queries_file))
    if not queries:
        print("未加载到任何 query，请检查 --queries-file。")
        return 1

    system = _build_system(_build_config(args.top_k))
    totals: list[float] = []
    stage_bottlenecks: dict[str, int] = {}
    records: list[dict] = []

    for run_idx in range(max(int(args.runs), 1)):
        for q_idx, query in enumerate(queries):
            sid = f"bench-{run_idx}-{q_idx}-{uuid.uuid4()}"
            t0 = time.perf_counter()
            state = system.invoke(
                {"query": query, "metrics": {"via": "benchmark"}},
                config={"configurable": {"thread_id": sid}},
            )
            wall = time.perf_counter() - t0
            metrics = (state or {}).get("metrics") or {}
            total = float(metrics.get("total_seconds") or wall)
            totals.append(total)
            stage_times = {
                "analyze": float(metrics.get("analyze_seconds", 0.0)),
                "retrieve": float(metrics.get("retrieve_seconds", 0.0)),
                "fuse": float(metrics.get("fuse_seconds", 0.0)),
                "rerank": float(metrics.get("rerank_seconds", 0.0)),
                "generate": float(metrics.get("generate_seconds", 0.0)),
            }
            bottleneck = max(stage_times.items(), key=lambda x: x[1])[0]
            stage_bottlenecks[bottleneck] = stage_bottlenecks.get(bottleneck, 0) + 1
            records.append(
                {
                    "query": query,
                    "total_seconds": total,
                    "complexity": ((state or {}).get("routing") or {}).get("complexity_level", ""),
                    "rerank_mode": metrics.get("rerank_mode", ""),
                    "stages": stage_times,
                    "bottleneck_stage": bottleneck,
                }
            )

    p50 = _percentile(totals, 0.50)
    p95 = _percentile(totals, 0.95)
    mean_v = statistics.mean(totals) if totals else 0.0
    print(f"samples={len(totals)}, mean={mean_v:.3f}s, p50={p50:.3f}s, p95={p95:.3f}s, target={args.p95_target:.3f}s")
    print("bottleneck_distribution:", json.dumps(stage_bottlenecks, ensure_ascii=False))
    if p95 > float(args.p95_target):
        print("性能未达标：P95 超过阈值，请优先排查 bottleneck_distribution 中占比最高阶段。")
    else:
        print("性能达标：P95 在目标阈值内。")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(
                {
                    "samples": len(totals),
                    "mean_seconds": mean_v,
                    "p50_seconds": p50,
                    "p95_seconds": p95,
                    "p95_target_seconds": float(args.p95_target),
                    "bottleneck_distribution": stage_bottlenecks,
                    "records": records,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"已输出报告: {args.output}")

    return 0 if p95 <= float(args.p95_target) else 2


if __name__ == "__main__":
    raise SystemExit(main())
