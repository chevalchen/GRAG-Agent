from __future__ import annotations

from dataclasses import dataclass

from src.app.online_qa.graphs.online_qa_graph import OnlineQAGraph
from src.app.online_qa.tools.answer_generation import AnswerGenerationTool
from src.app.online_qa.tools.graph_rag_search import GraphRAGSearchTool
from src.app.online_qa.tools.hybrid_search import HybridSearchTool


@dataclass
class OnlineQASystem:
    data_module: object
    index_module: object
    generation_module: object
    online_graph: OnlineQAGraph
    query_router: object
    traditional_retrieval: object
    graph_rag_retrieval: object


def build_online_qa_system(config) -> OnlineQASystem:
    from src.legacy.rag_modules.generation_integration import GenerationIntegrationModule
    from src.legacy.rag_modules.graph_data_preparation import GraphDataPreparationModule
    from src.legacy.rag_modules.graph_rag_retrieval import GraphRAGRetrieval
    from src.legacy.rag_modules.hybrid_retrieval import HybridRetrievalModule
    from src.legacy.rag_modules.intelligent_query_router import IntelligentQueryRouter
    from src.legacy.rag_modules.milvus_index_construction import MilvusIndexConstructionModule

    data_module = GraphDataPreparationModule(
        uri=config.neo4j_uri,
        user=config.neo4j_user,
        password=config.neo4j_password,
        database=config.neo4j_database,
    )
    index_module = MilvusIndexConstructionModule(
        host=config.milvus_host,
        port=config.milvus_port,
        collection_name=config.milvus_collection_name,
        dimension=config.milvus_dimension,
        model_name=config.embedding_model,
    )
    generation_module = GenerationIntegrationModule(
        model_name=config.llm_model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    traditional_retrieval = HybridRetrievalModule(
        config=config,
        milvus_module=index_module,
        data_module=data_module,
        llm_client=generation_module.client,
    )
    graph_rag_retrieval = GraphRAGRetrieval(config=config, llm_client=generation_module.client)
    query_router = IntelligentQueryRouter(
        traditional_retrieval=traditional_retrieval,
        graph_rag_retrieval=graph_rag_retrieval,
        llm_client=generation_module.client,
        config=config,
    )
    online_graph = OnlineQAGraph(
        router=query_router,
        hybrid_tool=HybridSearchTool(traditional_retrieval),
        graph_tool=GraphRAGSearchTool(graph_rag_retrieval),
        answer_tool=AnswerGenerationTool(generation_module),
        top_k=config.top_k,
    )
    return OnlineQASystem(
        data_module=data_module,
        index_module=index_module,
        generation_module=generation_module,
        online_graph=online_graph,
        query_router=query_router,
        traditional_retrieval=traditional_retrieval,
        graph_rag_retrieval=graph_rag_retrieval,
    )

