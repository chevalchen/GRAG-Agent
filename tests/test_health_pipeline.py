import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.app.config import DEFAULT_CONFIG, GraphRAGConfig
from src.app.offline_ingestion.graphs.ingestion_graph import OfflineIngestionGraph
from src.app.offline_ingestion.nodes.health_chunk_node import make_health_chunk_node
from src.app.offline_ingestion.nodes.health_milvus_index_node import make_health_milvus_index_node
from src.app.online_qa.graphs.online_qa_graph import build_graph
from src.app.online_qa.nodes.answer_node import _build_context_with_budget
from src.app.online_qa.nodes.health_retrieve_node import make_health_retrieve_node
from src.core.tools.graph.neo4j_tool import Neo4jGraphTool
from src.core.tools.llm.generation_tool import LLMGenerationTool
from src.core.tools.retrieval.bm25_tool import BM25Tool
from src.core.tools.vector.milvus_tool import MilvusVectorTool


class HealthChunkNodeTests(unittest.TestCase):
    def test_health_chunk_expands_questions_and_tracks_bad_lines(self):
        node = make_health_chunk_node()
        with tempfile.TemporaryDirectory() as recipe_dir:
            health_dir = os.path.join(recipe_dir, "health")
            os.makedirs(health_dir, exist_ok=True)
            file_path = os.path.join(health_dir, "train.jsonl")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write('{"questions":[["失眠怎么调理","失眠如何改善"]],"answers":["早睡早起","规律作息"]}\n')
                f.write('{"questions":["口干怎么办"],"answers":"注意补水"}\n')
                f.write("{bad json}\n")
            state = node({"recipe_dir": recipe_dir, "metrics": {}})

        records = state["health_chunk_records"]
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["answer"], "早睡早起")
        self.assertEqual(records[1]["answer"], "规律作息")
        self.assertEqual(records[2]["text"], "口干怎么办")
        self.assertEqual(records[2]["answer"], "注意补水")
        self.assertEqual(state["metrics"]["health_invalid_json_lines"], 1)
        self.assertEqual(state["metrics"]["health_questions_expanded"], 3)

    def test_health_chunk_without_files_returns_empty_records(self):
        node = make_health_chunk_node()
        with tempfile.TemporaryDirectory() as recipe_dir:
            state = node({"recipe_dir": recipe_dir, "metrics": {}})
        self.assertEqual(state["health_chunk_records"], [])
        self.assertEqual(state["metrics"]["health_files_found"], 0)


class HealthIngestionGraphScanTests(unittest.TestCase):
    def test_scan_allows_health_only_dataset(self):
        graph = OfflineIngestionGraph.__new__(OfflineIngestionGraph)
        with tempfile.TemporaryDirectory() as recipe_dir, tempfile.TemporaryDirectory() as output_dir:
            health_dir = os.path.join(recipe_dir, "health")
            os.makedirs(health_dir, exist_ok=True)
            with open(os.path.join(health_dir, "train.jsonl"), "w", encoding="utf-8") as f:
                f.write('{"questions":[["久坐腰酸怎么办"]],"answers":["注意活动"]}\n')
            state = graph._scan({"recipe_dir": recipe_dir, "output_dir": output_dir, "metrics": {}})
        self.assertNotIn("error", state)
        self.assertEqual(state["total_files"], 1)
        self.assertEqual(state["metrics"]["health_files_found"], 1)


class HealthMilvusIndexNodeTests(unittest.TestCase):
    @patch("src.app.offline_ingestion.nodes.health_milvus_index_node.MilvusVectorTool")
    def test_health_index_writes_documents_to_health_collection(self, milvus_cls):
        config = GraphRAGConfig(**DEFAULT_CONFIG.to_dict())
        milvus = MagicMock()
        milvus.ensure_collection.return_value = True
        milvus.upsert_documents.return_value = 1
        milvus.create_index.return_value = True
        milvus.load_collection.return_value = True
        milvus_cls.return_value = milvus

        node = make_health_milvus_index_node(config)
        state = node(
            {
                "health_chunk_records": [
                    {
                        "text": "感冒后咳嗽怎么办",
                        "answer": "多休息，多饮水",
                        "node_id": "health_0_0_0",
                        "chunk_type": "qa",
                        "source": "health_popularization",
                        "doc_type": "health_popularization",
                    }
                ],
                "metrics": {},
            }
        )

        self.assertEqual(state["health_indexed_count"], 1)
        self.assertTrue(state["metrics"]["health_indexed"])
        docs = milvus.upsert_documents.call_args.args[0]
        self.assertEqual(docs[0].page_content, "感冒后咳嗽怎么办")
        self.assertEqual(docs[0].metadata["answer"], "多休息，多饮水")


class HealthRetrieveNodeTests(unittest.TestCase):
    def test_health_retrieve_uses_dict_payload_and_reports_top_k(self):
        milvus = MagicMock(spec=MilvusVectorTool)
        milvus.has_collection.return_value = True
        milvus.invoke.return_value = [Document(page_content="Q", metadata={"chunk_id": "h1"})]
        node = make_health_retrieve_node(milvus, top_k=3, expand_factor=1.5, max_retrieval_top_k=12)
        state = node({"query": "最近总失眠怎么办", "routing": {"use_health_vec": True}, "metrics": {}})

        self.assertEqual(len(state["health_docs"]), 1)
        self.assertEqual(milvus.invoke.call_args.args[0], {"query": "最近总失眠怎么办", "top_k": 4})
        self.assertEqual(state["metrics"]["health_retrieve_top_k"], 4)
        self.assertTrue(state["metrics"]["health_collection_ready"])

    def test_health_retrieve_skips_when_collection_missing(self):
        milvus = MagicMock(spec=MilvusVectorTool)
        milvus.has_collection.return_value = False
        node = make_health_retrieve_node(milvus, top_k=3, expand_factor=1.5, max_retrieval_top_k=12)
        state = node({"query": "最近总失眠怎么办", "routing": {"use_health_vec": True}, "metrics": {}})

        self.assertEqual(state["health_docs"], [])
        self.assertFalse(state["metrics"]["health_collection_ready"])
        self.assertEqual(state["metrics"]["health_retrieved"], 0)
        milvus.invoke.assert_not_called()


class HealthAnswerContextTests(unittest.TestCase):
    def test_health_context_uses_answer_body_instead_of_only_question(self):
        docs = [
            Document(
                page_content="失眠怎么调理",
                metadata={
                    "doc_type": "health_popularization",
                    "chunk_type": "qa",
                    "source": "health_popularization",
                    "answer": "建议规律作息，减少熬夜。",
                },
            )
        ]
        context, _, _ = _build_context_with_budget(docs, {}, total_budget=400, per_doc_chars=200)
        self.assertIn("【健康科普】", context)
        self.assertIn("回答：建议规律作息，减少熬夜。", context)


class HealthGraphE2ETests(unittest.TestCase):
    def test_health_route_hits_health_collection_and_builds_answer_context(self):
        llm = MagicMock(spec=LLMGenerationTool)
        llm.invoke_text.side_effect = [
            '{"query_intent":"health_advice","use_graph":false,"use_drug_vec":false,"use_lit_vec":false,"use_health_vec":true,"source_hint":[],"keywords":["失眠"],"reason":"health"}',
            "保持规律作息。",
        ]
        bm25 = MagicMock(spec=BM25Tool)
        bm25.invoke.return_value = []
        milvus = MagicMock(spec=MilvusVectorTool)
        milvus.invoke.return_value = []
        health_milvus = MagicMock(spec=MilvusVectorTool)
        health_milvus.invoke.return_value = [
            Document(
                page_content="失眠怎么调理",
                metadata={
                    "node_id": "h1",
                    "chunk_id": "h1",
                    "doc_type": "health_popularization",
                    "chunk_type": "qa",
                    "source": "health_popularization",
                    "answer": "建议减少熬夜，保持规律作息。",
                },
            )
        ]
        neo4j = MagicMock(spec=Neo4jGraphTool)
        neo4j.invoke.return_value = []
        neo4j._graph = MagicMock()
        neo4j._graph.query.return_value = []

        config = GraphRAGConfig(**{**DEFAULT_CONFIG.to_dict(), "top_k": 3})
        graph = build_graph(
            config,
            bm25_tool=bm25,
            milvus_tool=milvus,
            health_milvus_tool=health_milvus,
            neo4j_tool=neo4j,
            llm_tool=llm,
        )

        state = graph.invoke(
            {"query": "最近总失眠怎么办", "metrics": {}},
            config={"configurable": {"thread_id": "t-health-e2e"}},
        )

        self.assertEqual(state["answer"], "保持规律作息。")
        self.assertTrue(health_milvus.invoke.called)
        answer_prompt = llm.invoke_text.call_args_list[-1].args[0]
        self.assertIn("【健康科普】", answer_prompt)
        self.assertIn("回答：建议减少熬夜，保持规律作息。", answer_prompt)


if __name__ == "__main__":
    unittest.main()
