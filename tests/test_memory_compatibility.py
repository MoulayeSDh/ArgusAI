from argusai.config import Config
from argusai.memory import MemoryManager


class NewQdrantClient:
    def query_points(self):
        return None


class LegacyQdrantVectorStore:
    def query(self):
        return self._client.search(collection_name="memory")


class ModernQdrantVectorStore:
    def query(self):
        return self._client.query_points(collection_name="memory")


def make_memory_manager():
    manager = MemoryManager.__new__(MemoryManager)
    manager.config = Config()
    return manager


def test_detects_legacy_qdrant_vector_store_with_new_client(monkeypatch):
    monkeypatch.setattr("argusai.memory.QdrantClient", NewQdrantClient)
    monkeypatch.setattr("argusai.memory.QdrantVectorStore", LegacyQdrantVectorStore)

    manager = make_memory_manager()

    assert manager._qdrant_vector_store_uses_removed_search_api() is True


def test_accepts_modern_qdrant_vector_store_with_new_client(monkeypatch):
    monkeypatch.setattr("argusai.memory.QdrantClient", NewQdrantClient)
    monkeypatch.setattr("argusai.memory.QdrantVectorStore", ModernQdrantVectorStore)

    manager = make_memory_manager()

    assert manager._qdrant_vector_store_uses_removed_search_api() is False


def test_disables_memory_for_incompatible_qdrant_stack(monkeypatch):
    monkeypatch.setattr("argusai.memory.QdrantClient", NewQdrantClient)
    monkeypatch.setattr("argusai.memory.QdrantVectorStore", LegacyQdrantVectorStore)

    manager = make_memory_manager()
    manager._validate_required_dependencies()

    assert manager.config.use_memory is False
