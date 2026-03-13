from __future__ import annotations

from typing import Any


class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str, database: str):
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._driver = None

    def _get_driver(self):
        if self._driver is not None:
            return self._driver
        try:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password), database=self._database)
            return self._driver
        except Exception as e:
            raise ConnectionError(f"failed to connect neo4j: {e}") from e

    def get_session(self):
        driver = self._get_driver()
        try:
            return driver.session(database=self._database)
        except Exception as e:
            raise ConnectionError(f"failed to create neo4j session: {e}") from e

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def __enter__(self):
        self._get_driver()
        return self

    def __exit__(self, *args: Any):
        self.close()


_instance: Neo4jClient | None = None


def get_neo4j_client() -> Neo4jClient:
    global _instance
    if _instance is None:
        from src.app.config import Config

        _instance = Neo4jClient(
            uri=Config.NEO4J_URI,
            user=Config.NEO4J_USER,
            password=Config.NEO4J_PASSWORD,
            database=Config.NEO4J_DATABASE,
        )
    return _instance
