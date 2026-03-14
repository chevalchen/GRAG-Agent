from __future__ import annotations


class MilvusClient:
    """
    Milvus 客户端
    
    Attributes:
        host: Milvus 主机
        port: Milvus 端口
        collection_name: 集合名称
        client: Milvus 客户端
    """
    def __init__(self, host: str, port: int, collection_name: str):
        self._host = host
        self._port = port
        self._collection_name = collection_name
        self._client = None

    def get_client(self):
        if self._client is not None:
            return self._client
        try:
            from pymilvus import MilvusClient as _MilvusClient

            self._client = _MilvusClient(uri=f"http://{self._host}:{self._port}")
            return self._client
        except Exception as e:
            raise ConnectionError(f"failed to connect milvus: {e}") from e

    def close(self) -> None:
        self._client = None

    def has_collection(self) -> bool:
        client = self.get_client()
        try:
            return client.has_collection(self._collection_name)
        except Exception as e:
            raise ConnectionError(f"failed to query milvus collection: {e}") from e


_instance: MilvusClient | None = None


def get_milvus_client() -> MilvusClient:
    global _instance
    if _instance is None:
        from src.app.config import Config

        _instance = MilvusClient(
            host=Config.MILVUS_HOST,
            port=Config.MILVUS_PORT,
            collection_name=Config.MILVUS_COLLECTION_NAME,
        )
    return _instance
