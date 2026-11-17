"""
Vector Database Manager using ChromaDB
Provides document embedding storage, similarity search, metadata filtering,
and chunk retrieval for Agentic RAG.
"""
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

from core.document_loader import DocumentChunk
import config

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


class ChromaManager:
    """
    Manages ChromaDB vector database instance, collection lifecycle,
    chunk ingestion, and similarity retrieval.
    """
    def __init__(self, persist_directory: Optional[Path] = None, collection_name: str = config.CHROMA_COLLECTION_NAME):
        self.persist_directory = Path(persist_directory or config.CHROMA_PERSIST_DIR)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedding_fn = None
        self._init_db()

    def _init_db(self):
        """Initializes ChromaDB client and default embedding function."""
        if not CHROMADB_AVAILABLE:
            return

        try:
            # Default embedding function uses all-MiniLM-L6-v2 via onnxruntime
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        except Exception:
            self.embedding_fn = None

        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False, allow_reset=True)
            )
            self._get_or_create_collection()
        except Exception as e:
            # Fallback to Ephemeral/in-memory client if persistent directory has lock
            try:
                self.client = chromadb.EphemeralClient(
                    settings=Settings(anonymized_telemetry=False, allow_reset=True)
                )
                self._get_or_create_collection()
            except Exception as e2:
                print(f"ChromaDB initialization error: {e2}")

    def _get_or_create_collection(self):
        if self.client:
            kwargs = {"name": self.collection_name}
            if self.embedding_fn:
                kwargs["embedding_function"] = self.embedding_fn
            self.collection = self.client.get_or_create_collection(**kwargs)

    def index_document_chunks(self, chunks: List[DocumentChunk], clear_existing: bool = True) -> int:
        """
        Stores document chunks and metadata into ChromaDB vector database.
        Returns the number of indexed chunks.
        """
        if not self.collection:
            self._init_db()
            if not self.collection:
                raise RuntimeError("ChromaDB collection is not available.")

        if clear_existing:
            self.clear()
            self._get_or_create_collection()

        if not chunks:
            return 0

        documents = []
        metadatas = []
        ids = []

        for chunk in chunks:
            documents.append(chunk.page_content)
            # Ensure metadata values are strings, ints, floats, or bools for ChromaDB
            meta = {
                "chunk_id": int(chunk.metadata.get("chunk_id", 0)),
                "source": str(chunk.metadata.get("source", "paper")),
                "section": str(chunk.metadata.get("section", "General")),
                "page_number": int(chunk.metadata.get("page_number", 1)),
                "char_length": int(chunk.metadata.get("char_length", len(chunk.page_content)))
            }
            metadatas.append(meta)
            ids.append(f"{meta['source']}_chunk_{meta['chunk_id']}")

        # ChromaDB batch upsert
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end = i + batch_size
            self.collection.upsert(
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end]
            )

        return len(documents)

    def similarity_search(
        self,
        query: str,
        k: int = config.DEFAULT_TOP_K,
        section_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes similarity search on ChromaDB, returning top-k grounded chunks.
        """
        if not self.collection:
            return []

        count = self.collection.count()
        if count == 0:
            return []

        actual_k = min(k, count)
        where_filter = None
        if section_filter and section_filter.lower() != "all":
            where_filter = {"section": section_filter}

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=actual_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
        except Exception:
            # Retry without filter if filter had no match
            results = self.collection.query(
                query_texts=[query],
                n_results=actual_k,
                include=["documents", "metadatas", "distances"]
            )

        formatted_results = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            dists = results["distances"][0] if "distances" in results else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, dists):
                # Convert cosine/L2 distance to a 0-1 similarity score estimate
                score = max(0.0, min(1.0, 1.0 - (dist / 2.0))) if dist is not None else 0.85
                formatted_results.append({
                    "content": doc,
                    "metadata": meta,
                    "chunk_id": meta.get("chunk_id"),
                    "section": meta.get("section", "General"),
                    "page_number": meta.get("page_number", 1),
                    "source": meta.get("source", "paper"),
                    "distance": dist,
                    "score": round(score, 3)
                })

        return formatted_results

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics on the current vector database state."""
        if not self.collection:
            return {"count": 0, "status": "Not Initialized"}

        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "count": count,
            "persist_dir": str(self.persist_directory),
            "status": "Ready" if count > 0 else "Empty"
        }

    def clear(self):
        """Clears the collection."""
        if self.client and self.collection:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.collection = None
            self._get_or_create_collection()
