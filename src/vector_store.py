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

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "index_type": type(self.index).__name__ if self.index else None
        }


# Create singleton instance
vector_store = FAISSVectorStore()
