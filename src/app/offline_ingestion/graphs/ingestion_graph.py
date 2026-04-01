import os

from langgraph.graph import END, StateGraph

from src.app.config import Config, DEFAULT_CONFIG, GraphRAGConfig
from src.app.offline_ingestion.nodes.chunk_node import make_chunk_node
from src.app.offline_ingestion.nodes.milvus_index_node import make_milvus_index_node
from src.app.offline_ingestion.nodes.neo4j_write_node import make_neo4j_write_node
from src.app.offline_ingestion.nodes.normalize_node import make_normalize_node
from src.app.offline_ingestion.nodes.parse_node import make_parse_node
from src.app.offline_ingestion.state import OfflineIngestionState


def _find_data_file(root_dir: str, file_name: str) -> str:
    if os.path.isfile(root_dir) and os.path.basename(root_dir) == file_name:
        return root_dir
    direct_path = os.path.join(root_dir, file_name)
    if os.path.exists(direct_path):
        return direct_path
    for root, _, files in os.walk(root_dir):
        if file_name in files:
            return os.path.join(root, file_name)
    return ""


class OfflineIngestionGraph:
    def __init__(self, batch_size: int = 20, parse_concurrency: int = 3, save_every: int = 5, base_url: str = ""):
        self._batch_size = batch_size
        self._parse_concurrency = parse_concurrency
        self._save_every = save_every
        self._base_url = base_url

        self._rag_config = GraphRAGConfig(
            **{
                **DEFAULT_CONFIG.to_dict(),
                "neo4j_uri": Config.NEO4J_URI,
                "neo4j_user": Config.NEO4J_USER,
                "neo4j_password": Config.NEO4J_PASSWORD,
                "neo4j_database": Config.NEO4J_DATABASE,
                "milvus_host": Config.MILVUS_HOST,
                "milvus_port": Config.MILVUS_PORT,
                "milvus_uri": Config.MILVUS_URI,
                "milvus_collection_name": Config.MILVUS_COLLECTION_NAME,
            }
        )

        self._parse_node = make_parse_node()
        self._chunk_node = make_chunk_node(chunk_size=self._rag_config.chunk_size, overlap=self._rag_config.chunk_overlap)
        self._normalize_node = make_normalize_node()
        self._neo4j_write_node = make_neo4j_write_node(self._rag_config)
        self._milvus_index_node = make_milvus_index_node(self._rag_config)
        self._graph = self._build()

    def _build(self):
        g = StateGraph(OfflineIngestionState)
        g.add_node("scan", self._scan)
        g.add_node("parse", self._parse_node)
        g.add_node("chunk", self._chunk_node)
        g.add_node("normalize", self._normalize_node)
        g.add_node("neo4j_write", self._neo4j_write)
        g.add_node("milvus_index", self._milvus_index)

        g.set_entry_point("scan")
        g.add_edge("scan", "parse")
        g.add_edge("scan", "chunk")
        g.add_edge("parse", "normalize")
        g.add_edge("normalize", "neo4j_write")
        g.add_edge("neo4j_write", "milvus_index")
        g.add_edge("chunk", "milvus_index")
        g.add_edge("milvus_index", END)
        return g.compile()

    def invoke(self, recipe_dir: str, output_dir: str, output_format: str = "neo4j", resume: bool = True):
        state: OfflineIngestionState = {
            "recipe_dir": recipe_dir,
            "output_dir": output_dir,
            "output_format": output_format,
            "resume": resume,
            "total_files": 2,
            "file_list": [],
            "next_index": 0,
            "processed": 0,
            "failed": 0,
            "current_batch_count": 0,
            "indexed_count": 0,
            "failed_writes": [],
            "metrics": {},
            "error": None,
            "parsed_records": [],
            "chunk_records": [],
        }
        return self._graph.invoke(state)

    def _scan(self, state: OfflineIngestionState):
        os.makedirs(state["output_dir"], exist_ok=True)
        medical_entities_path = _find_data_file(state["recipe_dir"], "medical_ner_entities.json")
        literature_path = _find_data_file(state["recipe_dir"], "Traditional_Chinese_Medical_Literature_QA.json")
        if not medical_entities_path or not literature_path:
            missing = []
            if not medical_entities_path:
                missing.append("medical_ner_entities.json")
            if not literature_path:
                missing.append("Traditional_Chinese_Medical_Literature_QA.json")
            return {"error": f"missing_data_files: {', '.join(missing)}"}
        return {
            "medical_entities_path": medical_entities_path,
            "literature_path": literature_path,
            "processed": 2,
            "metrics": {**(state.get("metrics") or {}), "scan_total": 2},
        }

    def _neo4j_write(self, state: OfflineIngestionState):
        return self._neo4j_write_node(state)

    def _milvus_index(self, state: OfflineIngestionState):
        return self._milvus_index_node(state)
