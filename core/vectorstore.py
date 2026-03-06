"""Vector store for document chunk storage and semantic search using ChromaDB."""

from typing import List
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings

from core.document import DocumentChunk
from core.embeddings import EmbeddingProvider


@dataclass
class SearchResult:
    """A single search result from the vector store."""
    chunk: DocumentChunk
    score: float

    def __str__(self):
        return f"[Page {self.chunk.page_number}] (score: {self.score:.3f}) {self.chunk.text}"


class VectorStore:
    """Manages document embeddings in ChromaDB for semantic retrieval."""

    def __init__(self, embedding_provider: EmbeddingProvider,
                 collection_name: str = "pdf_documents"):
        self._embedding_provider = embedding_provider
        self._collection_name = collection_name
        self._client = chromadb.Client(Settings(anonymized_telemetry=False))
        self._collection = None
        self._chunks: List[DocumentChunk] = []
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def clear(self):
        """Remove all documents from the store."""
        if self._collection is not None:
            self._client.delete_collection(self._collection_name)
        self._collection = None
        self._chunks = []
        self._is_loaded = False

    def add_chunks(self, chunks: List[DocumentChunk]):
        """Index a list of document chunks into the vector store."""
        if not chunks:
            return

        self.clear()
        self._chunks = chunks
        self._collection = self._client.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        texts = [chunk.text for chunk in chunks]
        embeddings = self._embedding_provider.embed_texts(texts)

        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "source": chunk.source,
            }
            for chunk in chunks
        ]

        self._collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
        )
        self._is_loaded = True

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Perform semantic search and return ranked results."""
        if not self._is_loaded or self._collection is None:
            return []

        query_embedding = self._embedding_provider.embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.chunk_count),
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                idx = int(doc_id.split("_")[1])
                distance = results["distances"][0][i]
                similarity = 1.0 - distance
                search_results.append(SearchResult(
                    chunk=self._chunks[idx],
                    score=similarity,
                ))

        search_results.sort(key=lambda r: r.score, reverse=True)
        return search_results
