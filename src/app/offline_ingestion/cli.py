import argparse
import sys

from src.app.config import Config
from src.app.offline_ingestion.graphs.ingestion_graph import OfflineIngestionGraph


def _collect_neo4j_stats() -> dict:
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            Config.NEO4J_URI,
            auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD),
        )
        session = driver.session(database=Config.NEO4J_DATABASE)
        node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rel_count = session.run("MATCH ()-[r]-() RETURN count(r) AS c").single()["c"]
        session.close()
        driver.close()
        return {"node_count": int(node_count), "relationship_count": int(rel_count), "error": ""}
    except Exception as e:
        return {"node_count": -1, "relationship_count": -1, "error": str(e)}


def _collect_milvus_stats() -> dict:
    try:
        from pymilvus import MilvusClient

        client = MilvusClient(uri=Config.MILVUS_URI)
        if not client.has_collection(Config.MILVUS_COLLECTION_NAME):
            return {"doc_count": 0, "error": "collection_not_found"}
        stats = client.get_collection_stats(collection_name=Config.MILVUS_COLLECTION_NAME) or {}
        row_count = int(stats.get("row_count") or 0)
        return {"doc_count": row_count, "error": ""}
    except Exception as e:
        return {"doc_count": -1, "error": str(e)}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="offline_ingestion", description="LangGraph 离线 ingestion")
    parser.add_argument("recipe_dir", type=str)
    parser.add_argument("-o", "--output-dir", type=str, default="./ai_output")
    parser.add_argument("--output-format", type=str, default="neo4j", choices=["neo4j", "csv"])
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--parse-concurrency", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    graph = OfflineIngestionGraph(
        batch_size=args.batch_size,
        parse_concurrency=args.parse_concurrency,
    )
    print(f"[offline_ingestion] start recipe_dir={args.recipe_dir} output_dir={args.output_dir}")
    state = graph.invoke(args.recipe_dir, args.output_dir, output_format=args.output_format, resume=args.resume) or {}
    error = state.get("error")
    metrics = state.get("metrics") or {}
    if error:
        print(f"[offline_ingestion] failed error={error}")
        print(f"[offline_ingestion] metrics={metrics}")
        sys.exit(1)
    print(
        "[offline_ingestion] done "
        f"processed={state.get('processed', 0)} "
        f"indexed={state.get('indexed_count', 0)} "
        f"failed={state.get('failed', 0)} "
        f"failed_writes={len(state.get('failed_writes') or [])}"
    )
    neo4j_stats = _collect_neo4j_stats()
    milvus_stats = _collect_milvus_stats()
    print(
        "[offline_ingestion] stats "
        f"neo4j_nodes={neo4j_stats['node_count']} "
        f"neo4j_relationships={neo4j_stats['relationship_count']} "
        f"milvus_docs={milvus_stats['doc_count']}"
    )
    if neo4j_stats["error"]:
        print(f"[offline_ingestion] stats_warn neo4j={neo4j_stats['error']}")
    if milvus_stats["error"]:
        print(f"[offline_ingestion] stats_warn milvus={milvus_stats['error']}")
    print(f"[offline_ingestion] metrics={metrics}")


if __name__ == "__main__":
    main()

