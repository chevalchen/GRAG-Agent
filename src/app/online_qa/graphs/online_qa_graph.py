import asyncio
import time
from typing import Dict, List, Optional

from langgraph.graph import END, StateGraph

from src.legacy.rag_modules.intelligent_query_router import IntelligentQueryRouter, QueryAnalysis
from src.app.online_qa.agents.answer_agent import AnswerAgent
from src.app.online_qa.agents.fusion import FusionAgent
from src.app.online_qa.agents.graph_agent import GraphRetrievalAgent
from src.app.online_qa.agents.hybrid_agent import HybridRetrievalAgent
from src.app.online_qa.agents.router_agent import RouterAgent
from src.app.online_qa.agents.supervisor import SupervisorAgent
from src.app.online_qa.state import OnlineQAState
from src.app.online_qa.tools.answer_generation import AnswerGenerationTool
from src.app.online_qa.tools.graph_rag_search import GraphRAGSearchTool
from src.app.online_qa.tools.hybrid_search import HybridSearchTool
from src.app.online_qa.tools.registry import make_tool_context


class OnlineQAGraph:
    def __init__(
        self,
        router: IntelligentQueryRouter,
        hybrid_tool: HybridSearchTool,
        graph_tool: GraphRAGSearchTool,
        answer_tool: AnswerGenerationTool,
        top_k: int,
        llm_concurrency: int = 2,
        retrieve_concurrency: int = 4,
    ):
        self._router = router
        self._hybrid_tool = hybrid_tool
        self._graph_tool = graph_tool
        self._answer_tool = answer_tool
        self._fusion = FusionAgent()
        self._top_k = top_k
        self._llm_sem = asyncio.Semaphore(llm_concurrency)
        self._retrieve_sem = asyncio.Semaphore(retrieve_concurrency)

        tools = {
            "router": router,
            "hybrid_search": hybrid_tool,
            "graph_rag_search": graph_tool,
            "answer_generation": answer_tool,
        }
        self._router_ctx = make_tool_context(tools, ["router"])
        self._hybrid_ctx = make_tool_context(tools, ["hybrid_search"])
        self._graph_ctx = make_tool_context(tools, ["graph_rag_search"])
        self._answer_ctx = make_tool_context(tools, ["answer_generation"])

        self._router_agent = RouterAgent(self._router_ctx, self._llm_sem)
        self._hybrid_agent = HybridRetrievalAgent(self._hybrid_ctx, self._retrieve_sem)
        self._graph_agent = GraphRetrievalAgent(self._graph_ctx, self._retrieve_sem)
        self._answer_agent = AnswerAgent(self._answer_ctx, self._llm_sem)
        self._supervisor = SupervisorAgent()
        self._graph = self._build()

    def _build(self):
        g = StateGraph(OnlineQAState)
        g.add_node("supervisor", self._supervisor_node)
        g.add_node("hybrid_retrieve", self._hybrid_retrieve)
        g.add_node("graph_retrieve", self._graph_retrieve)
        g.add_node("combined_retrieve", self._combined_retrieve)
        g.add_node("fuse", self._fuse)
        g.add_node("generate", self._generate)

        g.set_entry_point("supervisor")
        g.add_conditional_edges("supervisor", self._route, {
            "hybrid_traditional": "hybrid_retrieve",
            "graph_rag": "graph_retrieve",
            "combined": "combined_retrieve",
        })

        g.add_edge("hybrid_retrieve", "generate")
        g.add_edge("graph_retrieve", "generate")
        g.add_edge("combined_retrieve", "fuse")
        g.add_edge("fuse", "generate")
        g.add_edge("generate", END)
        return g.compile()

    async def ainvoke(self, query: str, stream: bool = False) -> OnlineQAState:
        init_state: OnlineQAState = {
            "query": query,
            "analysis": None,
            "route": None,
            "docs_hybrid": [],
            "docs_graph": [],
            "docs_final": [],
            "answer": None,
            "metrics": {"stream": stream},
            "errors": [],
        }
        return await self._graph.ainvoke(init_state)

    async def _supervisor_node(self, state: OnlineQAState):
        return await self._supervisor.run(state, self._router_agent)

    def _route(self, state: OnlineQAState) -> str:
        route = state.get("route") or "hybrid_traditional"
        if route not in {"hybrid_traditional", "graph_rag", "combined"}:
            return "hybrid_traditional"
        return route

    async def _hybrid_retrieve(self, state: OnlineQAState):
        start = time.perf_counter()
        docs = await self._hybrid_agent.retrieve(state["query"], self._top_k)
        router = self._router_ctx.get("router")
        docs = router.post_process_results(docs, state["analysis"]) if state["analysis"] else docs
        metrics = {
            **state["metrics"],
            "retrieve_seconds": time.perf_counter() - start,
            "docs_hybrid": len(docs),
        }
        return {"docs_final": docs, "docs_hybrid": docs, "metrics": metrics}

    async def _graph_retrieve(self, state: OnlineQAState):
        start = time.perf_counter()
        docs = await self._graph_agent.retrieve(state["query"], self._top_k)
        router = self._router_ctx.get("router")
        docs = router.post_process_results(docs, state["analysis"]) if state["analysis"] else docs
        metrics = {
            **state["metrics"],
            "retrieve_seconds": time.perf_counter() - start,
            "docs_graph": len(docs),
        }
        return {"docs_final": docs, "docs_graph": docs, "metrics": metrics}

    async def _combined_retrieve(self, state: OnlineQAState):
        start = time.perf_counter()
        traditional_k = max(1, self._top_k // 2)
        graph_k = self._top_k - traditional_k
        docs_hybrid, docs_graph = await asyncio.gather(
            asyncio.create_task(self._hybrid_agent.retrieve(state["query"], traditional_k)),
            asyncio.create_task(self._graph_agent.retrieve(state["query"], graph_k)),
        )
        metrics = {
            **state["metrics"],
            "retrieve_seconds": time.perf_counter() - start,
            "docs_hybrid": len(docs_hybrid),
            "docs_graph": len(docs_graph),
        }
        return {"docs_hybrid": docs_hybrid, "docs_graph": docs_graph, "metrics": metrics}

    async def _fuse(self, state: OnlineQAState):
        start = time.perf_counter()
        fused = self._fusion.fuse_round_robin(state["docs_graph"], state["docs_hybrid"], self._top_k)
        if state["analysis"]:
            router = self._router_ctx.get("router")
            fused = router.post_process_results(fused, state["analysis"])
        metrics = {**state["metrics"], "fuse_seconds": time.perf_counter() - start, "docs_final": len(fused)}
        return {"docs_final": fused, "metrics": metrics}

    async def _generate(self, state: OnlineQAState):
        if not state["docs_final"]:
            return {"answer": "抱歉，没有找到相关的烹饪信息。请尝试其他问题。"}
        if state["metrics"].get("stream"):
            return {"answer": None}
        start = time.perf_counter()
        answer = await self._answer_agent.generate(state["query"], state["docs_final"], stream=False)
        metrics = {**state["metrics"], "generate_seconds": time.perf_counter() - start}
        return {"answer": answer, "metrics": metrics}
