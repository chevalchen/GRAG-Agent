import asyncio
import os
import time
from typing import List, Optional, Tuple

from langgraph.graph import END, StateGraph

from src.legacy.agent.recipe_ai_agent import KimiRecipeAgent, RecipeInfo, RecipeKnowledgeGraphBuilder
from src.app.offline_ingestion.state import OfflineIngestionState
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
        self._builder: Optional[RecipeKnowledgeGraphBuilder] = None
        self._graph = self._build()

    def _build(self):
        g = StateGraph(OfflineIngestionState)
        g.add_node("init", self._init)
        g.add_node("scan", self._scan)
        g.add_node("parse_batch", self._parse_batch)
        g.add_node("flush", self._flush)
        g.add_node("finalize", self._finalize)
        g.add_node("export", self._export)

        g.set_entry_point("init")
        g.add_edge("init", "scan")
        g.add_edge("scan", "parse_batch")
        g.add_conditional_edges("parse_batch", self._after_parse, {"continue": "parse_batch", "flush": "flush", "finalize": "finalize"})
        g.add_conditional_edges("flush", self._after_flush, {"continue": "parse_batch", "finalize": "finalize"})
        g.add_edge("finalize", "export")
        g.add_edge("export", END)
        return g.compile()

    async def ainvoke(self, recipe_dir: str, output_dir: str, output_format: str = "neo4j", resume: bool = True):
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
            "metrics": {},
            "error": None,
        }
        return await self._graph.ainvoke(state)

    async def _init(self, state: OfflineIngestionState):
        if not self._api_key:
            return {"error": "missing_api_key"}

        os.makedirs(state["output_dir"], exist_ok=True)

        ai_agent = KimiRecipeAgent(self._api_key, self._base_url)
        builder = RecipeKnowledgeGraphBuilder(ai_agent, state["output_dir"], self._batch_size)
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
        return {}

    async def _scan(self, state: OfflineIngestionState):
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

    async def _parse_batch(self, state: OfflineIngestionState):
        if state.get("error") or not self._builder:
            return {}
        if state["next_index"] >= len(state["file_list"]):
            return {}

        start = time.time()
        semaphore = asyncio.Semaphore(self._parse_concurrency)

        batch_paths = state["file_list"][state["next_index"] : state["next_index"] + self._parse_concurrency]
        tasks = [asyncio.create_task(self._parse_one(p, state["recipe_dir"], semaphore)) for p in batch_paths]
        results = await asyncio.gather(*tasks)

        processed = state["processed"]
        failed = state["failed"]
        current_batch_count = state["current_batch_count"]

        for ok, rel_path, recipe_info, err in results:
            if ok and recipe_info:
                self._builder.process_recipe_info(recipe_info, rel_path)
                self._builder.processed_files.add(rel_path)
                processed += 1
                current_batch_count += 1
                if processed % self._save_every == 0:
                    self._builder.save_progress(rel_path, state["total_files"], processed)
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

    async def _parse_one(self, abs_path: str, recipe_root: str, semaphore: asyncio.Semaphore) -> Tuple[bool, str, Optional[RecipeInfo], Optional[str]]:
        rel_path = os.path.relpath(abs_path, recipe_root)
        async with semaphore:
            for attempt in range(3):
                try:
                    return await asyncio.to_thread(self._parse_one_sync, abs_path, rel_path)
                except Exception as e:
                    if attempt == 2:
                        return False, rel_path, None, str(e)
                    await asyncio.sleep(min(2 ** attempt, 4))
        return False, rel_path, None, "unknown"

    def _parse_one_sync(self, abs_path: str, rel_path: str) -> Tuple[bool, str, Optional[RecipeInfo], Optional[str]]:
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()
        agent = KimiRecipeAgent(self._api_key, self._base_url)
        recipe_info = agent.extract_recipe_info(content, rel_path)
        return True, rel_path, recipe_info, None

    async def _flush(self, state: OfflineIngestionState):
        if state.get("error") or not self._builder:
            return {}
        if state["current_batch_count"] <= 0:
            return {"current_batch_count": 0}

        start = time.time()
        await asyncio.to_thread(self._builder.save_batch_data)
        self._builder.concepts.clear()
        self._builder.relationships.clear()
        self._builder.current_batch += 1
        self._builder.save_progress("BATCH_FLUSH", state["total_files"], state["processed"])
        return {"current_batch_count": 0, "metrics": {**state["metrics"], "last_flush_seconds": time.time() - start}}

    async def _finalize(self, state: OfflineIngestionState):
        if state.get("error") or not self._builder:
            return {}
        if state["current_batch_count"] > 0:
            await asyncio.to_thread(self._builder.save_batch_data)
            self._builder.concepts.clear()
            self._builder.relationships.clear()
            self._builder.current_batch += 1
        self._builder.save_progress("COMPLETED", state["total_files"], state["processed"])
        return {}

    async def _export(self, state: OfflineIngestionState):
        if state.get("error") or not self._builder:
            return {}
        start = time.time()
        if state["output_format"] == "neo4j":
            await asyncio.to_thread(self._builder.export_to_neo4j_csv, state["output_dir"], True)
        else:
            await asyncio.to_thread(self._builder.merge_all_batches)
        return {"metrics": {**state["metrics"], "export_seconds": time.time() - start, "exported": True}}
