import unittest

from langchain_core.documents import Document

from src.app.online_qa.agents.fusion import FusionAgent


class FusionAgentTests(unittest.TestCase):
    def test_fuse_round_robin_graph_first_and_dedup(self):
        agent = FusionAgent()
        graph_docs = [
            Document(page_content="G1" * 100, metadata={}),
            Document(page_content="SAME" * 30, metadata={}),
        ]
        traditional_docs = [
            Document(page_content="T1" * 100, metadata={}),
            Document(page_content="SAME" * 30, metadata={}),
        ]

        fused = agent.fuse_round_robin(graph_docs, traditional_docs, top_k=10)
        self.assertEqual(fused[0].page_content[:2], "G1")
        self.assertEqual(fused[1].page_content[:2], "T1")
        self.assertEqual(len([d for d in fused if d.page_content.startswith("SAME")]), 1)

