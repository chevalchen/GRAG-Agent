import unittest
import time
from unittest.mock import MagicMock

from langchain_core.documents import Document

from src.app.online_qa.nodes.drug_retrieve_node import make_drug_retrieve_node
from src.core.tools.retrieval.bm25_tool import BM25Tool
from src.core.tools.vector.milvus_tool import MilvusVectorTool


class DrugRetrieveNodeTests(unittest.TestCase):
    def test_resolved_drug_adds_node_filter_and_query_boost(self):
        bm25 = MagicMock(spec=BM25Tool)
        milvus = MagicMock(spec=MilvusVectorTool)
        bm25.invoke.return_value = [Document(page_content="B", metadata={"chunk_id": "c1", "node_id": "drug_1"})]
        milvus.invoke.return_value = [Document(page_content="M", metadata={"chunk_id": "c2", "node_id": "drug_1"})]
        node = make_drug_retrieve_node(bm25, milvus, top_k=3)

        state = node(
            {
                "query": "乌鸡 白凤丸的适应症是什么？",
                "routing": {"use_drug_vec": True},
                "resolved_drug": {
                    "canonical_name": "乌鸡白凤丸",
                    "node_id": "drug_1",
                    "matched_alias": "乌鸡 白凤丸",
                    "match_type": "alias_norm",
                },
            }
        )

        self.assertEqual(len(state["drug_docs"]), 2)
        bm25_query = bm25.invoke.call_args.args[0]["query"]
        milvus_expr = milvus.invoke.call_args.args[0]["expr"]
        self.assertIn("乌鸡白凤丸", bm25_query)
        self.assertEqual(milvus_expr, 'doc_type=="drug" AND node_id=="drug_1"')

    def test_balanced_merge_interleaves_bm25_and_milvus(self):
        bm25 = MagicMock(spec=BM25Tool)
        milvus = MagicMock(spec=MilvusVectorTool)
        bm25.invoke.return_value = [
            Document(page_content="B1", metadata={"chunk_id": "b1"}),
            Document(page_content="B2", metadata={"chunk_id": "b2"}),
        ]
        milvus.invoke.return_value = [
            Document(page_content="M1", metadata={"chunk_id": "m1"}),
            Document(page_content="M2", metadata={"chunk_id": "m2"}),
        ]
        node = make_drug_retrieve_node(bm25, milvus, top_k=4, merge_strategy="balanced")
        state = node({"query": "测试", "routing": {"use_drug_vec": True}})
        contents = [d.page_content for d in state["drug_docs"]]
        self.assertEqual(contents, ["B1", "M1", "B2", "M2"])
        self.assertEqual(state["metrics"]["drug_merge_strategy"], "balanced")

    def test_timeout_returns_available_branch(self):
        bm25 = MagicMock(spec=BM25Tool)
        milvus = MagicMock(spec=MilvusVectorTool)

        def _slow_bm25(_):
            time.sleep(0.2)
            return [Document(page_content="B", metadata={"chunk_id": "b"})]

        bm25.invoke.side_effect = _slow_bm25
        milvus.invoke.return_value = [Document(page_content="M", metadata={"chunk_id": "m"})]
        node = make_drug_retrieve_node(bm25, milvus, top_k=3, timeout_seconds=0.05)
        state = node({"query": "测试", "routing": {"use_drug_vec": True}})
        contents = [d.page_content for d in state["drug_docs"]]
        self.assertEqual(contents, ["M"])
        self.assertTrue(state["metrics"]["drug_retrieve_timeout"])


if __name__ == "__main__":
    unittest.main()
