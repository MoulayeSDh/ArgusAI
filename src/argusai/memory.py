"""
Semantic memory management for ArgusAI.

Memory stack:
- Ollama embeddings through LlamaIndex
- Qdrant vector store
- Short-term memory is handled by agent.py
- Long-term semantic memory is handled here

This module is optional by design. If dependencies or Qdrant are unavailable,
ArgusAI can continue without long-term memory.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Deque, List, Optional, cast

from .config import Config, Message
from .utils import document_signature, safe_float

if TYPE_CHECKING:
    from .ollama_client import OllamaClient


# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
except ImportError:
    QdrantClient = None  # type: ignore[assignment]
    qdrant_models = None  # type: ignore[assignment]

try:
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.schema import TextNode
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.vector_stores.qdrant import QdrantVectorStore
except ImportError:
    StorageContext = None  # type: ignore[assignment]
    VectorStoreIndex = None  # type: ignore[assignment]
    SentenceSplitter = None  # type: ignore[assignment]
    TextNode = None  # type: ignore[assignment]
    OllamaEmbedding = None  # type: ignore[assignment]
    QdrantVectorStore = None  # type: ignore[assignment]


class MemoryManager:
    """Optional Qdrant + LlamaIndex semantic memory layer."""

    def __init__(self, config: Config, ollama_client: "OllamaClient") -> None:
        self.config = config
        self.ollama_client = ollama_client

        self.qdrant_client: Optional[Any] = None
        self.qdrant_vector_store: Optional[Any] = None
        self.vector_index: Optional[Any] = None
        self.embed_model: Optional[Any] = None
        self.node_parser: Optional[Any] = None
        self.embedding_dim: Optional[int] = None

        self.last_retrieved_count: int = 0
        self._ingested_signatures: set[str] = set()

        self._validate_required_dependencies()
        self._init_embeddings()
        self._init_qdrant()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _validate_required_dependencies(self) -> None:
        """Disable semantic memory if required Python packages are missing."""

        if not self.config.use_memory:
            return

        missing: List[str] = []

        if StorageContext is None or VectorStoreIndex is None:
            missing.append("llama-index-core")

        if OllamaEmbedding is None:
            missing.append("llama-index-embeddings-ollama")

        if SentenceSplitter is None or TextNode is None:
            missing.append("llama-index-core node parser/schema")

        if QdrantClient is None or qdrant_models is None or QdrantVectorStore is None:
            missing.append("qdrant-client / llama-index-vector-stores-qdrant")

        if not missing and self._qdrant_vector_store_uses_removed_search_api():
            missing.append(
                "compatible llama-index-vector-stores-qdrant "
                "(upgrade to >=0.10 for qdrant-client>=1.16)"
            )

        if missing:
            logging.warning(
                "Semantic memory disabled. Missing dependencies: %s",
                ", ".join(missing),
            )
            self.config.use_memory = False

    def _qdrant_vector_store_uses_removed_search_api(self) -> bool:
        """
        Detect old LlamaIndex/Qdrant integrations with new qdrant-client.

        qdrant-client 1.16 removed deprecated search/search_batch methods.
        Older llama-index-vector-stores-qdrant releases still call them during
        retrieval, which otherwise fails only at runtime.
        """

        if QdrantClient is None or QdrantVectorStore is None:
            return False

        if hasattr(QdrantClient, "search") or hasattr(QdrantClient, "search_batch"):
            return False

        try:
            source = inspect.getsource(QdrantVectorStore)
        except (OSError, TypeError):
            return False

        removed_calls = (
            "._client.search(",
            "._client.search_batch(",
            "._aclient.search(",
            "._aclient.search_batch(",
        )

        return any(call in source for call in removed_calls)

    def _init_embeddings(self) -> None:
        """Initialize Ollama embeddings through LlamaIndex."""

        if not self.config.use_memory:
            return

        if OllamaEmbedding is None or SentenceSplitter is None:
            return

        try:
            self.embed_model = OllamaEmbedding(
                model_name=self.config.embedding_model,
                base_url=self.config.ollama_base_url,
            )

            self.node_parser = SentenceSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )

            probe = self._probe_text_embedding("ArgusAI embedding warmup check.")

            if probe is None:
                logging.warning("Embedding preflight failed. Disabling semantic memory.")
                self.config.use_memory = False
                self.embed_model = None
                self.node_parser = None
                self.embedding_dim = None
                return

            self.embedding_dim = len(probe)

            logging.info(
                "Embedding model ready: %s | detected_dim=%s",
                self.config.embedding_model,
                self.embedding_dim,
            )

        except Exception as exc:
            logging.warning("Embedding initialization failed: %s", exc)
            self.config.use_memory = False
            self.embed_model = None
            self.node_parser = None
            self.embedding_dim = None

    def _init_qdrant(self) -> None:
        """Initialize Qdrant collection and LlamaIndex vector index."""

        if not self.config.use_memory:
            return

        if QdrantClient is None or qdrant_models is None or QdrantVectorStore is None:
            return

        if self.embed_model is None:
            logging.warning("Embedding model unavailable; memory disabled.")
            self.config.use_memory = False
            return

        if self.embedding_dim is None:
            logging.warning("Embedding dimension is unknown; memory disabled.")
            self.config.use_memory = False
            return

        try:
            self.qdrant_client = QdrantClient(
                url=self.config.qdrant_url,
                timeout=10,
            )
            self.qdrant_client.get_collections()
        except Exception as exc:
            logging.warning("Qdrant unavailable: %s", exc)
            self._handle_missing_qdrant()
            return

        try:
            self._ensure_collection_exists()

            self.qdrant_vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.config.qdrant_collection,
            )

            assert StorageContext is not None
            assert VectorStoreIndex is not None

            storage_context = StorageContext.from_defaults(
                vector_store=self.qdrant_vector_store,
            )

            self.vector_index = VectorStoreIndex(
                nodes=[],
                storage_context=storage_context,
                embed_model=self.embed_model,
            )

        except Exception as exc:
            logging.warning("Qdrant initialization failed: %s", exc)
            self._handle_missing_qdrant()

    def _ensure_collection_exists(self) -> None:
        """Create or validate the configured Qdrant collection."""

        if self.qdrant_client is None:
            raise RuntimeError("Qdrant client is not initialized.")

        if qdrant_models is None:
            raise RuntimeError("Qdrant models are unavailable.")

        if self.embedding_dim is None:
            raise RuntimeError("Embedding dimension is unavailable.")

        embedding_dim = int(self.embedding_dim)

        collections = self.qdrant_client.get_collections()
        existing = {collection.name for collection in collections.collections}

        if self.config.qdrant_collection not in existing:
            qdrant_models_ns = cast(Any, qdrant_models)

            self.qdrant_client.create_collection(
                collection_name=self.config.qdrant_collection,
                vectors_config=qdrant_models_ns.VectorParams(
                    size=embedding_dim,
                    distance=qdrant_models_ns.Distance.COSINE,
                ),
            )

            logging.info(
                "Created Qdrant collection: %s | size=%s",
                self.config.qdrant_collection,
                embedding_dim,
            )

            return

        if not self._ensure_collection_matches_embedding_dim():
            self.qdrant_client = None
            self.qdrant_vector_store = None
            self.vector_index = None
            raise RuntimeError("Qdrant collection schema is incompatible.")

    def _handle_missing_qdrant(self) -> None:
        """
        Disable memory when Qdrant is unavailable.

        In interactive mode, optionally ask the user before continuing.
        In non-interactive/Docker mode, disable memory automatically unless
        Config.ask_before_disable_memory is False.
        """

        self.qdrant_client = None
        self.qdrant_vector_store = None
        self.vector_index = None

        if not self.config.ask_before_disable_memory or not sys.stdin.isatty():
            self.config.use_memory = False
            return

        answer = input("Long-term memory is unavailable. Continue without it? [y/n]: ")
        answer = answer.strip().lower()

        if answer not in {"y", "yes", "o", "oui"}:
            raise RuntimeError("Startup aborted because long-term memory is unavailable.")

        self.config.use_memory = False

    # ------------------------------------------------------------------
    # Embedding checks
    # ------------------------------------------------------------------

    def _candidate_embedding_methods(
        self,
        *,
        prefer_query: bool = False,
    ) -> List[Callable[..., Any]]:
        """Return callable embedding methods available on the embedding model."""

        if self.embed_model is None:
            return []

        if prefer_query:
            candidate_method_names = [
                "get_query_embedding",
                "_get_query_embedding",
                "get_text_embedding",
                "_get_text_embedding",
            ]
        else:
            candidate_method_names = [
                "get_text_embedding",
                "_get_text_embedding",
                "get_query_embedding",
                "_get_query_embedding",
            ]

        methods: List[Callable[..., Any]] = []

        for method_name in candidate_method_names:
            maybe = getattr(self.embed_model, method_name, None)

            if callable(maybe):
                methods.append(cast(Callable[..., Any], maybe))

        return methods

    def _probe_text_embedding(
        self,
        text: str,
        *,
        prefer_query: bool = False,
    ) -> Optional[List[float]]:
        """
        Probe embedding generation and reject invalid vectors.

        This prevents NaN/inf/all-zero vectors from entering Qdrant.
        """

        methods = self._candidate_embedding_methods(prefer_query=prefer_query)

        if not methods:
            logging.warning("No usable embedding method found on embed_model.")
            return None

        for method in methods:
            try:
                vector = method(text)
            except Exception as exc:
                logging.warning(
                    "Embedding probe failed via %s: %s",
                    getattr(method, "__name__", "method"),
                    exc,
                )
                continue

            if vector is None or not isinstance(vector, (list, tuple)) or len(vector) == 0:
                continue

            try:
                cleaned_vector = [safe_float(item, 0.0) for item in vector]
            except Exception as exc:
                logging.warning("Embedding sanitation failed: %s", exc)
                continue

            if not cleaned_vector:
                continue

            if all(value == 0.0 for value in cleaned_vector):
                logging.warning("Embedding vector collapsed to all zeros.")
                continue

            return cleaned_vector

        return None

    def embedding_looks_valid(self, text: str) -> bool:
        """Public embedding preflight helper."""

        return self._probe_text_embedding(text) is not None

    # ------------------------------------------------------------------
    # Qdrant collection validation
    # ------------------------------------------------------------------

    def _extract_collection_vector_size(self, collection_info: Any) -> Optional[int]:
        """Extract vector size from different Qdrant client response shapes."""

        vectors = None

        try:
            vectors = collection_info.config.params.vectors
        except Exception:
            pass

        if vectors is not None:
            vectors_any = cast(Any, vectors)

            if hasattr(vectors_any, "size"):
                try:
                    return int(vectors_any.size)
                except Exception:
                    return None

        if isinstance(vectors, dict) and vectors:
            first = next(iter(vectors.values()))

            if hasattr(first, "size"):
                try:
                    return int(first.size)
                except Exception:
                    return None

            if isinstance(first, dict) and "size" in first:
                try:
                    return int(first["size"])
                except Exception:
                    return None

        if isinstance(collection_info, dict):
            try:
                vectors_cfg = collection_info["config"]["params"]["vectors"]

                if isinstance(vectors_cfg, dict) and "size" in vectors_cfg:
                    return int(vectors_cfg["size"])

            except Exception:
                return None

        return None

    def _get_collection_points_count(
        self,
        collection_name: str,
        collection_info: Any,
    ) -> Optional[int]:
        """Return Qdrant collection point count if available."""

        points_count = getattr(collection_info, "points_count", None)

        if points_count is not None:
            try:
                return int(points_count)
            except Exception:
                pass

        if self.qdrant_client is None:
            return None

        try:
            count_result = cast(
                Any,
                self.qdrant_client.count(
                    collection_name=collection_name,
                    exact=True,
                ),
            )

            count_value = getattr(count_result, "count", None)

            if count_value is not None:
                return int(count_value)

        except Exception as exc:
            logging.warning(
                "Unable to count Qdrant points for %s: %s",
                collection_name,
                exc,
            )

        return None

    def _confirm_yes_no(self, question: str, *, default_no: bool = True) -> bool:
        """Ask a yes/no question in interactive terminals."""

        if not sys.stdin.isatty():
            return not default_no

        answer = input(question).strip().lower()

        return answer in {"y", "yes", "o", "oui"}

    def _recreate_collection(self, collection_name: str, expected_size: int) -> bool:
        """Delete and recreate a Qdrant collection."""

        if self.qdrant_client is None or qdrant_models is None:
            return False

        qdrant_models_ns = cast(Any, qdrant_models)

        self.qdrant_client.delete_collection(collection_name=collection_name)

        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=qdrant_models_ns.VectorParams(
                size=expected_size,
                distance=qdrant_models_ns.Distance.COSINE,
            ),
        )

        logging.info(
            "Recreated Qdrant collection %s with size=%s",
            collection_name,
            expected_size,
        )

        return True

    def _ensure_collection_matches_embedding_dim(self) -> bool:
        """Validate that Qdrant collection vector size matches embedding dim."""

        if self.qdrant_client is None or self.embedding_dim is None:
            return False

        collection_name = self.config.qdrant_collection
        expected_size = int(self.embedding_dim)

        collection_info = self.qdrant_client.get_collection(
            collection_name=collection_name,
        )

        existing_size = self._extract_collection_vector_size(collection_info)

        if existing_size is None:
            logging.warning(
                "Unable to inspect Qdrant collection vector size for %s.",
                collection_name,
            )
            return True

        if existing_size == expected_size:
            return True

        logging.warning(
            "Qdrant collection size mismatch for %s: existing=%s expected=%s",
            collection_name,
            existing_size,
            expected_size,
        )

        points_count = self._get_collection_points_count(
            collection_name,
            collection_info,
        )

        if points_count == 0:
            logging.info(
                "Collection %s is empty. Recreating it automatically.",
                collection_name,
            )
            return self._recreate_collection(collection_name, expected_size)

        if self._confirm_yes_no(
            "Memory schema mismatch. Recreate now? [y/n]: ",
            default_no=True,
        ):
            return self._recreate_collection(collection_name, expected_size)

        logging.warning(
            "Semantic memory disabled because the collection schema is incompatible."
        )
        self.config.use_memory = False

        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """Return True if semantic memory is available."""

        return (
            self.config.use_memory
            and self.vector_index is not None
            and self.embed_model is not None
        )

    def retrieve(self, query: str, short_memory: Deque[Message]) -> str:
        """
        Retrieve short-term and long-term memory context.

        Short-term memory is included when available.
        Long-term retrieval is used only when Qdrant/LlamaIndex are ready.
        """

        self.last_retrieved_count = 0

        blocks: List[str] = []

        if short_memory:
            recent = "\n".join(
                f"- [{message.role}] {message.content[:250]}"
                for message in short_memory
            )
            blocks.append("Recent conversation:\n" + recent)

        if self.is_ready() and query.strip() and self.vector_index is not None:
            if self._probe_text_embedding(query, prefer_query=True) is None:
                logging.warning(
                    "Skipping semantic retrieval because the query embedding probe failed."
                )
            else:
                try:
                    vector_index = cast(Any, self.vector_index)
                    retriever = vector_index.as_retriever(
                        similarity_top_k=self.config.memory_top_k,
                    )

                    retrieved_nodes = retriever.retrieve(query)
                    self.last_retrieved_count = len(retrieved_nodes)

                    if retrieved_nodes:
                        snippets: List[str] = []

                        for item in retrieved_nodes:
                            node_obj = getattr(item, "node", item)
                            node_text = getattr(node_obj, "text", str(node_obj))
                            snippets.append(f"- {str(node_text)[:450]}")

                        long_term = "\n".join(snippets)
                        blocks.append("Long-term memory:\n" + long_term)

                except Exception as exc:
                    logging.warning("Memory retrieval failed: %s", exc)

        return "\n\n".join(blocks)

    def write(
        self,
        user_text: str,
        answer: str,
        route: str,
        score: float,
        short_memory: Deque[Message],
    ) -> None:
        """
        Write current exchange into short-term memory and long-term memory.

        Long-term write is skipped if Qdrant/LlamaIndex are not ready.
        """

        if user_text:
            short_memory.append(
                Message(
                    role="user",
                    content=user_text[:600],
                )
            )

        if answer:
            short_memory.append(
                Message(
                    role="assistant",
                    content=answer[:900],
                )
            )

        if not self.is_ready() or TextNode is None or self.vector_index is None:
            return

        safe_score = safe_float(score, 0.0)

        payload = (
            f"User: {user_text}\n"
            f"Assistant: {answer}\n"
            f"Route: {route}\n"
            f"Score: {safe_score:.2f}"
        )[: self.config.max_memory_write_chars]

        if self._probe_text_embedding(payload) is None:
            logging.warning(
                "Skipping semantic memory write because embedding probe failed."
            )
            return

        node = TextNode(
            text=payload,
            metadata={
                "type": "conversation",
                "route": route,
                "score": safe_score,
                "created_at": time.time(),
            },
            id_=str(uuid.uuid4()),
        )

        try:
            vector_index = cast(Any, self.vector_index)
            vector_index.insert_nodes([node])

        except Exception as exc:
            logging.warning("Memory write failed for this payload only: %s", exc)

    def ingest_document(self, doc_path: str, text: str) -> None:
        """
        Ingest a document into semantic memory.

        Documents are deduplicated by file path, modification time and size.
        Absolute file paths are not stored in Qdrant metadata.
        """

        if not self.is_ready():
            return

        if self.node_parser is None or TextNode is None or self.vector_index is None:
            return

        path = Path(doc_path)

        if not path.exists():
            return

        signature = document_signature(path)

        if signature in self._ingested_signatures:
            return

        truncated_text = text[: self.config.max_doc_chars]
        chunks = self.node_parser.split_text(truncated_text)

        if not chunks:
            return

        source_hash = hashlib.sha256(
            str(path.resolve()).encode("utf-8")
        ).hexdigest()[:16]

        nodes = [
            TextNode(
                text=chunk,
                metadata={
                    "source_name": path.name,
                    "source_hash": source_hash,
                    "chunk_id": chunk_id,
                    "type": "document",
                    "ingested_at": time.time(),
                },
                id_=str(uuid.uuid4()),
            )
            for chunk_id, chunk in enumerate(chunks)
        ]

        inserted_any = False
        vector_index = cast(Any, self.vector_index)

        for node in nodes:
            try:
                vector_index.insert_nodes([node])
                inserted_any = True
            except Exception as exc:
                logging.warning(
                    "Skipping one document chunk during ingestion: %s",
                    exc,
                )

        if inserted_any:
            self._ingested_signatures.add(signature)
            logging.info("Ingested document into semantic memory: %s", path.name)

    def status(self) -> str:
        """Return human-readable memory status."""

        return "on" if self.is_ready() else "off"
