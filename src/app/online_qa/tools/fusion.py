from __future__ import annotations

from typing import List

from src.core.schemas.document import Document


class FusionTool:
    def fuse(self, graph_docs: List[Document], hybrid_docs: List[Document], top_k: int) -> List[Document]:
        combined_docs: List[Document] = []
        seen_contents = set()
        max_len = max(len(hybrid_docs), len(graph_docs))
        for i in range(max_len):
            if i < len(graph_docs):
                doc = graph_docs[i]
                content_hash = hash(doc.content[:100])
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    doc.metadata["search_source"] = "graph_rag"
                    combined_docs.append(doc)
            if i < len(hybrid_docs):
                doc = hybrid_docs[i]
                content_hash = hash(doc.content[:100])
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    doc.metadata["search_source"] = "traditional"
                    combined_docs.append(doc)
        return combined_docs[:top_k]

