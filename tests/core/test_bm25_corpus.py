"""
Tests for Stage 3 addition to BM25Tool:
  - filter_expr=None → full corpus (empty Milvus filter)
  - filter_expr="..." → custom corpus (filter passed to Milvus)
  - BM25 retrieval works across all doc_types when built from full corpus
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from langchain_core.documents import Document

from src.core.tools.retrieval.bm25_tool import BM25Tool


def _milvus_stub(rows: list[dict]) -> MagicMock:
    """Build a minimal milvus_tool mock whose client.query returns `rows`."""
    client = MagicMock()
    client.query.return_value = rows
    milvus = MagicMock()
    milvus.client = client
    milvus.collection_name = "tcm_knowledge"
    return milvus


def _row(text: str, doc_type: str, idx: int = 0) -> dict:
    return {
        "text": text,
        "doc_type": doc_type,
        "chunk_type": "",
        "source": "",
        "drug_name": "",
        "canonical_name": "",
        "normalized_name": "",
        "aliases_text": "",
        "alias_norms_text": "",
        "answer": "",
        "node_id": f"n{idx}",
        "chunk_id": f"c{idx}",
    }


class TestBM25FullCorpus(unittest.TestCase):

    def test_default_filter_is_empty_string(self):
        """filter_expr=None (default) → _fetch_from_milvus calls client.query with filter=''."""
        milvus = _milvus_stub([_row("全量文档", "tcm_literature", 0)])
        bm25 = BM25Tool(milvus_tool=milvus, cache_path=".cache/_test_bm25_default.pkl")
        # Call _fetch_from_milvus directly to bypass the disk-cache lookup
        bm25._fetch_from_milvus()
        call_filter = milvus.client.query.call_args.kwargs.get("filter", "NOT_FOUND")
        self.assertEqual(call_filter, "")

    def test_custom_filter_expr_is_passed_through(self):
        """filter_expr set at construction → forwarded to client.query filter kwarg."""
        milvus = _milvus_stub([_row("药品文档", "drug", 0)])
        expr = 'doc_type == "drug"'
        bm25 = BM25Tool(
            milvus_tool=milvus,
            filter_expr=expr,
            cache_path=".cache/_test_bm25_drug.pkl",
        )
        # Call _fetch_from_milvus directly to bypass the disk-cache lookup
        bm25._fetch_from_milvus(bm25._filter_expr)
        call_filter = milvus.client.query.call_args.kwargs.get("filter", "NOT_FOUND")
        self.assertEqual(call_filter, expr)

    def test_full_corpus_includes_non_drug_docs(self):
        """BM25 built from full corpus indexes all doc_types (not just 'drug')."""
        milvus = _milvus_stub([
            _row("人参补气益血", "drug", 0),
            _row("失眠食疗健康科普", "health_popularization", 1),
        ])
        bm25 = BM25Tool(milvus_tool=milvus, cache_path=".cache/_test_bm25_full.pkl")
        bm25._ensure_loaded()
        # Corpus should contain both doc_types
        corpus_types = {d.metadata.get("doc_type") for d in (bm25._documents or [])}
        self.assertIn("drug", corpus_types)
        self.assertIn("health_popularization", corpus_types)

    def test_filtered_corpus_excludes_other_types(self):
        """BM25 built with doc_type filter only indexes matching docs."""
        milvus = _milvus_stub([
            _row("人参功效说明", "drug", 0),
            # health_popularization row is excluded by the filter (Milvus won't return it)
        ])
        bm25 = BM25Tool(
            milvus_tool=milvus,
            filter_expr='doc_type == "drug"',
            cache_path=".cache/_test_bm25_filtered.pkl",
        )
        bm25._ensure_loaded()
        # Only 1 doc in corpus — the drug one
        self.assertEqual(len(bm25._documents or []), 1)
        self.assertEqual((bm25._documents or [{}])[0].page_content, "人参功效说明")

    def test_documents_kwarg_bypasses_milvus(self):
        """Passing documents= directly skips Milvus entirely; corpus is set from docs."""
        milvus = _milvus_stub([])
        docs = [Document(page_content="直接传入文档", metadata={"doc_type": "drug"})]
        bm25 = BM25Tool(
            milvus_tool=milvus,
            documents=docs,
            cache_path=".cache/_test_bm25_direct.pkl",
        )
        bm25._ensure_loaded()
        # Milvus must not be touched when documents= is provided
        milvus.client.query.assert_not_called()
        # Corpus must contain the injected document
        self.assertEqual(len(bm25._documents or []), 1)
        self.assertEqual((bm25._documents or [{}])[0].page_content, "直接传入文档")


if __name__ == "__main__":
    unittest.main()
