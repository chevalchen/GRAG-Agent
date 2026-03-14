import concurrent.futures
import os
import time
from typing import Optional

from langgraph.graph import END, StateGraph

from src.app.config import Config, DEFAULT_CONFIG, GraphRAGConfig
from src.app.offline_ingestion.nodes.milvus_index_node import make_milvus_index_node
from src.app.offline_ingestion.nodes.neo4j_write_node import make_neo4j_write_node
from src.app.offline_ingestion.state import OfflineIngestionState
from src.app.offline_ingestion.tools.build_tool import GraphRecordBuilderTool
from src.app.offline_ingestion.tools.export_tool import ExporterTool
from src.app.offline_ingestion.tools.parse_tool import RecipeParseTool, create_builder
from src.app.offline_ingestion.tools.progress_tool import ProgressStoreTool
from src.app.offline_ingestion.tools.scan_files import scan_recipe_files
from src.utils.env_utils import KIMI_API_KEY, MOONSHOT_API_KEY


class OfflineIngestionGraph:
    def __init__(
        self,
        batch_size: int = 20,
        parse_concurrency: int = 3,
        save_every: int = 5,
        base_url: str = "https://api.moonshot.cn/v1",
    ):
        self._batch_size = batch_size
        self._parse_concurrency = parse_concurrency
        self._save_every = save_every
        self._base_url = base_url

        self._api_key = KIMI_API_KEY or MOONSHOT_API_KEY
        self._builder = None
        self._parse_tool: Optional[RecipeParseTool] = None
        self._build_tool: Optional[GraphRecordBuilderTool] = None
        self._export_tool: Optional[ExporterTool] = None
        self._progress_tool: Optional[ProgressStoreTool] = None
        self._neo4j_write_node = None
        self._milvus_index_node = None
        self._rag_config: GraphRAGConfig | None = None
        self._graph = self._build()

    def _build(self):
        g = StateGraph(OfflineIngestionState)
        g.add_node("init", self._init)
        g.add_node("scan", self._scan)
        g.add_node("parse_batch", self._parse_batch)
        g.add_node("flush", self._flush)
        g.add_node("finalize", self._finalize)
        g.add_node("neo4j_write", self._neo4j_write)
        g.add_node("milvus_index", self._milvus_index)
        g.add_node("export", self._export)

        g.set_entry_point("init")
        g.add_edge("init", "scan")
        g.add_edge("scan", "parse_batch")
        g.add_conditional_edges("parse_batch", self._after_parse, {"continue": "parse_batch", "flush": "flush", "finalize": "finalize"})
        g.add_conditional_edges("flush", self._after_flush, {"continue": "parse_batch", "finalize": "finalize"})
        g.add_edge("finalize", "neo4j_write")
        g.add_edge("neo4j_write", "milvus_index")
        g.add_edge("milvus_index", "export")
        g.add_edge("export", END)
        return g.compile()

    def invoke(self, recipe_dir: str, output_dir: str, output_format: str = "neo4j", resume: bool = True):
        state: OfflineIngestionState = {
            "recipe_dir": recipe_dir,
            "output_dir": output_dir,
            "output_format": output_format,
            "resume": resume,
            "total_files": 0,
            "file_list": [],
            "next_index": 0,
            "processed": 0,
            "failed": 0,
            "current_batch_count": 0,
            "indexed_count": 0,
            "failed_writes": [],
            "metrics": {},
            "error": None,
        }
        return self._graph.invoke(state)

    def _init(self, state: OfflineIngestionState):
        if not self._api_key:
            return {"error": "missing_api_key"}

        os.makedirs(state["output_dir"], exist_ok=True)

        builder = create_builder(self._api_key, self._base_url, state["output_dir"], self._batch_size)
        if state["resume"]:
            builder.load_progress()
        else:
            builder.processed_files.clear()
            builder.current_batch = 0
            builder.concept_id_counter = 201000000
            if os.path.exists(builder.progress_file):
                try:
                    os.remove(builder.progress_file)
                except Exception:
                    pass
        self._builder = builder
        self._parse_tool = RecipeParseTool(self._api_key, self._base_url, state["recipe_dir"])
        self._build_tool = GraphRecordBuilderTool(builder)
        self._export_tool = ExporterTool(builder)
        self._progress_tool = ProgressStoreTool(builder)
        self._rag_config = GraphRAGConfig(
            **{
                **DEFAULT_CONFIG.to_dict(),
                "neo4j_uri": Config.NEO4J_URI,
                "neo4j_user": Config.NEO4J_USER,
                "neo4j_password": Config.NEO4J_PASSWORD,
                "neo4j_database": Config.NEO4J_DATABASE,
                "milvus_host": Config.MILVUS_HOST,
                "milvus_port": Config.MILVUS_PORT,
                "milvus_collection_name": Config.MILVUS_COLLECTION_NAME,
            }
        )
        self._neo4j_write_node = make_neo4j_write_node(self._rag_config, builder=self._builder)
        self._milvus_index_node = make_milvus_index_node(self._rag_config)
        return {}

    def _scan(self, state: OfflineIngestionState):
        if state.get("error") or not self._builder:
            return {}

        all_files = scan_recipe_files(state["recipe_dir"], self._builder.ai_agent.excluded_directories)
        remaining = []
        for abs_path, rel_path in all_files:
            if rel_path not in self._builder.processed_files:
                remaining.append((abs_path, rel_path))

        return {
            "total_files": len(all_files),
            "file_list": [abs_path for abs_path, _ in remaining],
            "next_index": 0,
            "processed": len(self._builder.processed_files),
            "failed": 0,
            "current_batch_count": 0,
            "metrics": {**state["metrics"], "scan_total": len(all_files), "scan_remaining": len(remaining)},
        }

    def _after_parse(self, state: OfflineIngestionState) -> str:
        if state.get("error"):
            return "finalize"
        if state["current_batch_count"] >= self._batch_size:
            return "flush"
        if state["next_index"] >= len(state["file_list"]):
            return "finalize"
        return "continue"

    def _after_flush(self, state: OfflineIngestionState) -> str:
        if state.get("error"):
            return "finalize"
        if state["next_index"] >= len(state["file_list"]):
            return "finalize"
        return "continue"

    def _parse_batch(self, state: OfflineIngestionState):
        if state.get("error") or not self._builder:
            return {}
        if state["next_index"] >= len(state["file_list"]):
            return {}

        start = time.time()
        batch_paths = state["file_list"][state["next_index"] : state["next_index"] + self._parse_concurrency]
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._parse_concurrency) as executor:
            futures = [executor.submit(self._parse_one, p, state["recipe_dir"]) for p in batch_paths]
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append({"ok": False, "rel_path": "", "recipe_info": None, "error": str(e)})

        processed = state["processed"]
        failed = state["failed"]
        current_batch_count = state["current_batch_count"]

        for parsed in results:
            if parsed.get("ok") and parsed.get("recipe_info"):
                self._build_tool.build(parsed)
                processed += 1
                current_batch_count += 1
                if processed % self._save_every == 0:
                    progress_state = {**state, "processed": processed}
                    self._progress_tool.save(progress_state, parsed.get("rel_path", ""))
            else:
                failed += 1

        next_index = state["next_index"] + len(batch_paths)
        elapsed = time.time() - start
        return {
            "next_index": next_index,
            "processed": processed,
            "failed": failed,
            "current_batch_count": current_batch_count,
            "metrics": {**state["metrics"], "last_parse_seconds": elapsed},
        }

    def _parse_one(self, abs_path: str, recipe_root: str) -> dict:
        for attempt in range(3):
            try:
                return self._parse_tool.parse(abs_path)
            except Exception as e:
                if attempt == 2:
                    rel_path = os.path.relpath(abs_path, recipe_root)
                    return {"ok": False, "rel_path": rel_path, "recipe_info": None, "error": str(e)}
                time.sleep(min(2 ** attempt, 4))
        rel_path = os.path.relpath(abs_path, recipe_root)
        return {"ok": False, "rel_path": rel_path, "recipe_info": None, "error": "unknown"}

    def _flush(self, state: OfflineIngestionState):
        if state.get("error") or not self._builder:
            return {}
        if state["current_batch_count"] <= 0:
            return {"current_batch_count": 0}

        start = time.time()
        self._builder.save_batch_data()
        self._builder.concepts.clear()
        self._builder.relationships.clear()
        self._builder.current_batch += 1
        self._progress_tool.save(state, "BATCH_FLUSH")
        return {"current_batch_count": 0, "metrics": {**state["metrics"], "last_flush_seconds": time.time() - start}}

    def _finalize(self, state: OfflineIngestionState):
        if state.get("error") or not self._builder:
            return {}
        if state["current_batch_count"] > 0:
            self._builder.save_batch_data()
            self._builder.concepts.clear()
            self._builder.relationships.clear()
            self._builder.current_batch += 1
        self._progress_tool.save(state, "COMPLETED")
        return {}

    def _export(self, state: OfflineIngestionState):
        if state.get("error") or not self._builder:
            return {}
        start = time.time()
        self._export_tool.export_csv([], state["output_dir"], state["output_format"])
        return {"metrics": {**state["metrics"], "export_seconds": time.time() - start, "exported": True}}

    def _neo4j_write(self, state: OfflineIngestionState):
        if self._neo4j_write_node is None:
            return {}
        return self._neo4j_write_node(state)

    def _milvus_index(self, state: OfflineIngestionState):
        if self._milvus_index_node is None:
            return {}
        return self._milvus_index_node(state)
