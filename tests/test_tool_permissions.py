import asyncio
import unittest

from src.app.online_qa.tools.registry import make_tool_context


class ToolPermissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_context_denies_unlisted_tool(self):
        ctx = make_tool_context({"a": object(), "b": object()}, allowed=["a"])
        self.assertIsNotNone(ctx.get("a"))
        with self.assertRaises(PermissionError):
            ctx.get("b")

