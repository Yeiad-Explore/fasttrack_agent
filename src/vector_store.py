"""
FAISS Vector Store with persistent storage
Handles document indexing and similarity search
"""

import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from pathlib import Path
from config import settings


class FAISSVectorStore:
    """FAISS-based vector store with persistence"""

    def __init__(self, dimension: int = 3072):
        """
        Initialize FAISS vector store

        Args:
            dimension: Embedding vector dimension
        """
        self.dimension = dimension
        self.index = None
        self.documents = []  # Store document metadata
        self.index_path = settings.VECTOR_STORE_DIR / "faiss_index.bin"
        self.metadata_path = settings.VECTOR_STORE_DIR / "metadata.pkl"

        # Initialize or load index
        self._initialize_index()

    def _initialize_index(self):
        """Initialize or load existing FAISS index"""
        if self.index_path.exists() and self.metadata_path.exists():
            print("Loading existing FAISS index...")
            self.load()
        else:
            print("Creating new FAISS index...")
            # Use HNSW index for better performance
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index.hnsw.efConstruction = 40
            self.index.hnsw.efSearch = 16
            self.documents = []

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[np.ndarray],
        metadatas: List[Dict[str, Any]]
    ):
        """
        Add documents to the vector store

        Args:
            texts: List of document texts
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
        """
        if len(texts) != len(embeddings) != len(metadatas):
            raise ValueError("texts, embeddings, and metadatas must have same length")

        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Add to FAISS index
        self.index.add(embeddings_array)

        # Store documents with metadata
        for text, metadata in zip(texts, metadatas):
            doc = {
                "text": text,
                "metadata": metadata
            }
            self.documents.append(doc)

        print(f"Added {len(texts)} documents. Total: {len(self.documents)}")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar documents

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            score_threshold: Minimum similarity score (optional)

        Returns:
            List of (document, score) tuples
        """
        if self.index is None or self.index.ntotal == 0:
            print("Index is empty")
            return []

        # Ensure query is 2D array
        query_array = np.array([query_embedding], dtype=np.float32)

        # Search
        distances, indices = self.index.search(query_array, k)

        # Convert distances to similarity scores (L2 distance -> similarity)
        # For normalized vectors: similarity = 1 / (1 + distance)
        similarities = 1 / (1 + distances[0])

        # Prepare results
        results = []
        for idx, score in zip(indices[0], similarities):
            if idx < len(self.documents):
                if score_threshold is None or score >= score_threshold:
                    results.append((self.documents[idx], float(score)))

        return results

    def save(self):
        """Save index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))

            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.documents, f)

            print(f"Saved index with {len(self.documents)} documents")
        except Exception as e:
            print(f"Error saving index: {e}")
            raise

    def load(self):
        """Load index and metadata from disk"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))

            # Load metadata
            with open(self.metadata_path, 'rb') as f:
                self.documents = pickle.load(f)

            print(f"Loaded index with {len(self.documents)} documents")
        except Exception as e:
            print(f"Error loading index: {e}")
            raise

    def clear(self):
        """Clear the index and documents"""
        self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        self.index.hnsw.efConstruction = 40
        self.index.hnsw.efSearch = 16
        self.documents = []
        print("Cleared vector store")

    def is_file_indexed(self, file_path: str) -> bool:
        """
        Check if a file is already indexed in the vector store

        Args:
            file_path: Path to the file to check

        Returns:
            True if file is already indexed, False otherwise
        """
        for doc in self.documents:
            if doc.get("metadata", {}).get("file_path") == file_path:
                return True
        return False

    def get_indexed_files(self) -> List[str]:
        """
        Get list of all indexed file paths

        Returns:
            List of file paths that are indexed
        """
        indexed_files = set()
        for doc in self.documents:
            file_path = doc.get("metadata", {}).get("file_path")
            if file_path:
                indexed_files.add(file_path)
        return sorted(list(indexed_files))

    def remove_file(self, file_path: str) -> int:
        """
        Remove all chunks from a specific file

        Args:
            file_path: Path to the file to remove

        Returns:
            Number of chunks removed
        """
        # Find indices to keep
        indices_to_keep = []
        new_documents = []

        for i, doc in enumerate(self.documents):
            if doc.get("metadata", {}).get("file_path") != file_path:
                indices_to_keep.append(i)
                new_documents.append(doc)

        if len(indices_to_keep) == len(self.documents):
            print(f"File not found in index: {file_path}")
            return 0

        # Rebuild index with remaining documents
        removed_count = len(self.documents) - len(new_documents)

        if len(new_documents) == 0:
            # Clear everything
            self.clear()
        else:
            # Rebuild index (FAISS doesn't support deletion, so we rebuild)
            print(f"Rebuilding index after removing {removed_count} chunks...")
            self.clear()
            # Note: This requires re-embedding, which we can't do here
            # This method is a placeholder for future implementation
            raise NotImplementedError(
                "Removing individual files requires re-embedding. "
                "Use clear_index() and re-ingest instead."
            )

        return removed_count

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        indexed_files = self.get_indexed_files()
        file_types = {}
        for doc in self.documents:
            file_type = doc.get("metadata", {}).get("file_type", "unknown")
            file_types[file_type] = file_types.get(file_type, 0) + 1

        return {
            "total_documents": len(self.documents),
            "total_files": len(indexed_files),
            "index_size": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "index_type": type(self.index).__name__ if self.index else None,
            "file_types": file_types,
            "indexed_files": indexed_files
        }


# Create singleton instance
vector_store = FAISSVectorStore()
