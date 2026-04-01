import unittest
from unittest.mock import MagicMock, patch

import src.core.tools.db.milvus_client as milvus_module
import src.core.tools.db.neo4j_client as neo4j_module
from src.core.tools.db.milvus_client import MilvusClient, get_milvus_client
from src.core.tools.db.neo4j_client import Neo4jClient, get_neo4j_client
from src.core.tools.llm.embedding_client import EmbeddingClient


class CoreToolsTests(unittest.TestCase):
    def test_neo4j_get_session_raises_connection_error_when_connect_fail(self):
        client = Neo4jClient("bolt://x", "u", "p", "neo4j")
        with patch.object(client, "_get_driver", side_effect=ConnectionError("boom")):
            with self.assertRaises(ConnectionError):
                client.get_session()

    def test_milvus_get_client_raises_connection_error_when_connect_fail(self):
        client = MilvusClient("localhost", 19530, "c")
        with patch.object(client, "get_client", side_effect=ConnectionError("boom")):
            with self.assertRaises(ConnectionError):
                client.get_client()

    def test_embedding_embed_returns_vectors(self):
        c = EmbeddingClient("k", "https://api.example.com", "m")
        fake = MagicMock()
        fake.data = [MagicMock(embedding=[0.1, 0.2]), MagicMock(embedding=[0.3, 0.4])]
        c._client.embeddings.create = MagicMock(return_value=fake)
        vectors = c.embed(["a", "b"])
        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), 2)

    def test_llm_generation_tool_invoke_text(self):
        from langchain_core.messages import HumanMessage

        from src.core.tools.llm.generation_tool import LLMGenerationTool

        tool = LLMGenerationTool.__new__(LLMGenerationTool)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="test_answer")
        tool._llm = mock_llm
        msg = tool._llm.invoke([HumanMessage(content="test prompt")])
        self.assertEqual(msg.content, "test_answer")

    def test_neo4j_singleton_factory(self):
        neo4j_module._instance = None
        c1 = get_neo4j_client()
        c2 = get_neo4j_client()
        self.assertIs(c1, c2)

    def test_milvus_singleton_factory(self):
        milvus_module._instance = None
        c1 = get_milvus_client()
        c2 = get_milvus_client()
        self.assertIs(c1, c2)
