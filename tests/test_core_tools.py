import unittest
from unittest.mock import MagicMock, patch

from src.app.online_qa.tools.registry import assert_tool_allowed
from src.core.schemas.document import Document
import src.core.tools.db.milvus_client as milvus_module
import src.core.tools.db.neo4j_client as neo4j_module
from src.core.tools.db.milvus_client import MilvusClient, get_milvus_client
from src.core.tools.db.neo4j_client import Neo4jClient, get_neo4j_client
from src.core.tools.llm.embedding_client import EmbeddingClient
from src.core.tools.llm.generation_tool import GenerationTool


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

    def test_generation_tool_calls_llm_chat_only(self):
        llm = MagicMock()
        llm.chat = MagicMock(return_value="ok")
        tool = GenerationTool(llm)
        out = tool.generate("q", [Document(content="ctx")], stream=False)
        self.assertEqual(out, "ok")
        llm.chat.assert_called_once()

    def test_registry_permission_error(self):
        from src.app.online_qa.tools.hybrid_search import HybridSearchTool

        with self.assertRaises(PermissionError):
            assert_tool_allowed("query_analysis_node", HybridSearchTool)

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
