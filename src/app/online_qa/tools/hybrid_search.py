from typing import List

from langchain_core.documents import Document


class HybridSearchTool:
    def __init__(self, hybrid_retrieval_module):
        self._hybrid = hybrid_retrieval_module

    def search(self, query: str, top_k: int) -> List[Document]:
        return self._hybrid.hybrid_search(query, top_k)
