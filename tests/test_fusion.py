import unittest

from src.app.online_qa.tools.fusion import FusionTool
from src.core.schemas.document import Document


class FusionAgentTests(unittest.TestCase):
    def test_fuse_round_robin_graph_first_and_dedup(self):
        agent = FusionTool()
        graph_docs = [
            Document(content="G1" * 100, metadata={}),
            Document(content="SAME" * 30, metadata={}),
        ]
        traditional_docs = [
            Document(content="T1" * 100, metadata={}),
            Document(content="SAME" * 30, metadata={}),
        ]

        fused = agent.fuse(graph_docs, traditional_docs, top_k=10)
        self.assertEqual(fused[0].content[:2], "G1")
        self.assertEqual(fused[1].content[:2], "T1")
        self.assertEqual(len([d for d in fused if d.content.startswith("SAME")]), 1)
