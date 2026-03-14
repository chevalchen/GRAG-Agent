import unittest

from langchain_core.documents import Document

from src.app.online_qa.nodes.fuse_node import _dedup_key, _rrf


class RRFFusionTests(unittest.TestCase):
    def _doc(self, content: str, node_id: str = "", chunk_id: str = "") -> Document:
        return Document(page_content=content, metadata={"node_id": node_id, "chunk_id": chunk_id})

    def test_rrf_scores_shared_doc_higher(self):
        graph_docs = [self._doc("G1" * 10, "g1", "c1"), self._doc("SAME" * 10, "s", "cs")]
        hybrid_docs = [self._doc("T1" * 10, "t1", "ct"), self._doc("SAME" * 10, "s", "cs")]
        result = _rrf([graph_docs, hybrid_docs], k=60, top_k=10)
        self.assertEqual(_dedup_key(result[0]), "scs")

    def test_rrf_deduplicates_by_node_chunk_key(self):
        docs_a = [self._doc("A" * 10, "n1", "c1"), self._doc("B" * 10, "n2", "c2")]
        docs_b = [self._doc("A" * 10, "n1", "c1"), self._doc("C" * 10, "n3", "c3")]
        result = _rrf([docs_a, docs_b], k=60, top_k=10)
        keys = [_dedup_key(d) for d in result]
        self.assertEqual(len(keys), len(set(keys)))

    def test_rrf_respects_top_k(self):
        docs = [self._doc(f"D{i}" * 5, f"n{i}", f"c{i}") for i in range(20)]
        result = _rrf([docs], k=60, top_k=5)
        self.assertEqual(len(result), 5)

    def test_rrf_attaches_final_score(self):
        docs = [self._doc("X" * 5, "nx", "cx")]
        result = _rrf([docs], k=60, top_k=5)
        self.assertIn("final_score", result[0].metadata)
        self.assertGreater(result[0].metadata["final_score"], 0.0)

    def test_empty_inputs_return_empty(self):
        self.assertEqual(_rrf([], k=60, top_k=5), [])
        self.assertEqual(_rrf([[]], k=60, top_k=5), [])


if __name__ == "__main__":
    unittest.main()
