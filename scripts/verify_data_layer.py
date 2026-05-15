"""
scripts/verify_data_layer.py
-----------------------------
One-shot data-layer smoke check.

Pings Neo4j (Concept node distribution) and Milvus (both collections),
then writes a JSON report to benchmark/data_layer_report.json.

Usage:
    uv run python -m scripts.verify_data_layer

Exit codes:
    0 — all checks pass
    1 — one or more checks failed (see report for details)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from src.app.config import Config, DEFAULT_CONFIG, GraphRAGConfig


def _check_neo4j(config: GraphRAGConfig) -> dict:
    result: dict = {"ok": False, "concept_distribution": {}, "error": None}
    try:
        from langchain_neo4j import Neo4jGraph

        graph = Neo4jGraph(
            url=config.neo4j_uri,
            username=config.neo4j_user,
            password=config.neo4j_password,
            database=config.neo4j_database,
        )
        rows = graph.query(
            "MATCH (n:Concept) RETURN n.conceptType AS type, count(*) AS cnt"
        )
        dist = {str(r.get("type") or "Unknown"): int(r.get("cnt") or 0) for r in (rows or [])}
        result["concept_distribution"] = dist
        result["total_concepts"] = sum(dist.values())
        result["ok"] = True
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _check_milvus_collection(config: GraphRAGConfig, collection_name: str) -> dict:
    result: dict = {"ok": False, "collection": collection_name, "row_count": None, "error": None}
    try:
        from pymilvus import MilvusClient

        uri = str(getattr(config, "milvus_uri", "") or "").strip()
        client = MilvusClient(uri=uri) if uri else MilvusClient(
            uri=f"http://{config.milvus_host}:{config.milvus_port}"
        )
        if not client.has_collection(collection_name):
            result["error"] = f"Collection '{collection_name}' does not exist"
            return result
        stats = client.get_collection_stats(collection_name)
        row_count = int((stats or {}).get("row_count") or 0)
        result["row_count"] = row_count
        result["ok"] = True
    except Exception as exc:
        result["error"] = str(exc)
    return result


def main() -> int:
    config = GraphRAGConfig(
        **{
            **DEFAULT_CONFIG.to_dict(),
            "neo4j_uri": Config.NEO4J_URI,
            "neo4j_user": Config.NEO4J_USER,
            "neo4j_password": Config.NEO4J_PASSWORD,
            "neo4j_database": Config.NEO4J_DATABASE,
            "milvus_host": Config.MILVUS_HOST,
            "milvus_port": Config.MILVUS_PORT,
            "milvus_uri": Config.MILVUS_URI,
            "milvus_collection_name": Config.MILVUS_COLLECTION_NAME,
            "milvus_health_collection_name": Config.MILVUS_HEALTH_COLLECTION_NAME,
        }
    )

    print("=== Data Layer Verification ===\n")

    t0 = time.time()

    print("1. Checking Neo4j ...")
    neo4j_result = _check_neo4j(config)
    if neo4j_result["ok"]:
        print(f"   OK — {neo4j_result['total_concepts']} Concept nodes")
        for ctype, cnt in sorted(neo4j_result["concept_distribution"].items()):
            print(f"      {ctype}: {cnt}")
    else:
        print(f"   FAIL — {neo4j_result['error']}")

    print("\n2. Checking Milvus (tcm_knowledge) ...")
    milvus_main = _check_milvus_collection(config, Config.MILVUS_COLLECTION_NAME)
    if milvus_main["ok"]:
        print(f"   OK — {milvus_main['row_count']} rows")
    else:
        print(f"   FAIL — {milvus_main['error']}")

    print("\n3. Checking Milvus (health_knowledge) ...")
    milvus_health = _check_milvus_collection(config, Config.MILVUS_HEALTH_COLLECTION_NAME)
    if milvus_health["ok"]:
        print(f"   OK — {milvus_health['row_count']} rows")
    else:
        print(f"   FAIL — {milvus_health['error']}")

    elapsed = round(time.time() - t0, 2)
    all_ok = neo4j_result["ok"] and milvus_main["ok"] and milvus_health["ok"]

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed,
        "all_ok": all_ok,
        "neo4j": neo4j_result,
        "milvus_main": milvus_main,
        "milvus_health": milvus_health,
    }

    out_path = Path("benchmark/data_layer_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport written to {out_path}  (elapsed {elapsed}s)")
    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
