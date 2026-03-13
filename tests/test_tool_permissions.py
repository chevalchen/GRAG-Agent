import unittest

from src.app.online_qa.tools.hybrid_search import HybridSearchTool
from src.app.online_qa.tools.query_analysis import QueryAnalysisTool
from src.app.online_qa.tools.registry import assert_tool_allowed


class ToolPermissionTests(unittest.TestCase):
    def test_registry_denies_unlisted_tool(self):
        assert_tool_allowed("query_analysis_node", QueryAnalysisTool)
        with self.assertRaises(PermissionError):
            assert_tool_allowed("query_analysis_node", HybridSearchTool)
