"""
Hybrid Retrieval System
Combines semantic search with BM25 and cross-encoder reranking
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import settings
from embeddings import embedding_service
from vector_store import vector_store


class HybridRetriever:
    """Hybrid retrieval combining semantic search, BM25, and reranking"""

    def __init__(self):
        """Initialize the hybrid retriever"""
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.bm25_index = None
        self.bm25_documents = []

    def _build_bm25_index(self, documents: List[Dict[str, Any]]):
        """
        Build BM25 index from documents

        Args:
            documents: List of document dictionaries
        """
        # Tokenize documents for BM25
        tokenized_docs = [doc["text"].lower().split() for doc in documents]
        self.bm25_index = BM25Okapi(tokenized_docs)
        self.bm25_documents = documents

    def _semantic_search(
        self,
        query: str,
        k: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform semantic search using vector similarity

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of (document, score) tuples
        """
        # Generate query embedding
        query_embedding = embedding_service.embed_text(query)

        # Search vector store
        results = vector_store.search(
            query_embedding,
            k=k,
            score_threshold=settings.SIMILARITY_THRESHOLD
        )

        return results

    def _bm25_search(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        k: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform BM25 keyword search

        Args:
            query: Search query
            documents: List of documents to search
            k: Number of results

        Returns:
            List of (document, score) tuples
        """
        # Build BM25 index if needed
        if self.bm25_index is None or self.bm25_documents != documents:
            self._build_bm25_index(documents)

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores
        scores = self.bm25_index.get_scores(tokenized_query)

        # Get top k results
        top_indices = np.argsort(scores)[::-1][:k]

        results = [
            (documents[idx], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0
        ]

        return results

    def _rerank(
        self,
        query: str,
        documents: List[Tuple[Dict[str, Any], float]],
        top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rerank documents using cross-encoder

        Args:
            query: Search query
            documents: List of (document, score) tuples
            top_k: Number of results to return

        Returns:
            Reranked list of (document, score) tuples
        """
        if not documents:
            return []

        # Prepare pairs for cross-encoder
        pairs = [[query, doc["text"]] for doc, _ in documents]

        # Get cross-encoder scores
        ce_scores = self.cross_encoder.predict(pairs)

        # Combine with original scores (weighted)
        combined_results = []
        for (doc, orig_score), ce_score in zip(documents, ce_scores):
            # Weight: 60% cross-encoder, 40% original
            combined_score = 0.6 * float(ce_score) + 0.4 * orig_score
            combined_results.append((doc, combined_score))

        # Sort by combined score
        combined_results.sort(key=lambda x: x[1], reverse=True)

        return combined_results[:top_k]

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        use_reranking: bool = True
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform hybrid retrieval

        Args:
            query: Search query
            top_k: Number of final results (defaults to TOP_K_RERANK)
            use_reranking: Whether to use cross-encoder reranking

        Returns:
            List of (document, score) tuples
        """
        if top_k is None:
            top_k = settings.TOP_K_RERANK

        # Step 1: Semantic search
        semantic_results = self._semantic_search(
            query,
            k=settings.TOP_K_RETRIEVAL
        )

        if not semantic_results:
            print("No semantic results found")
            return []

        # Step 2: BM25 search on semantic results
        documents_for_bm25 = [doc for doc, _ in semantic_results]
        bm25_results = self._bm25_search(
            query,
            documents_for_bm25,
            k=settings.TOP_K_RETRIEVAL
        )

        # Step 3: Merge results (deduplicate and combine scores)
        merged = self._merge_results(semantic_results, bm25_results)

        # Step 4: Rerank if enabled
        if use_reranking:
            final_results = self._rerank(query, merged, top_k)
        else:
            final_results = sorted(merged, key=lambda x: x[1], reverse=True)[:top_k]

        return final_results

    def _merge_results(
        self,
        semantic_results: List[Tuple[Dict[str, Any], float]],
        bm25_results: List[Tuple[Dict[str, Any], float]]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Merge semantic and BM25 results

        Args:
            semantic_results: Results from semantic search
            bm25_results: Results from BM25 search

        Returns:
            Merged list of (document, score) tuples
        """
        # Use document text as unique key
        merged_dict = {}

        # Add semantic results (weight: 0.7)
        for doc, score in semantic_results:
            key = doc["text"]
            merged_dict[key] = (doc, 0.7 * score)

        # Add/update with BM25 results (weight: 0.3)
        for doc, score in bm25_results:
            key = doc["text"]
            if key in merged_dict:
                # Combine scores
                existing_doc, existing_score = merged_dict[key]
                merged_dict[key] = (existing_doc, existing_score + 0.3 * score)
            else:
                merged_dict[key] = (doc, 0.3 * score)

        # Convert back to list
        merged_results = list(merged_dict.values())

        # Sort by score
        merged_results.sort(key=lambda x: x[1], reverse=True)

        return merged_results


# Create singleton instance
retriever = HybridRetriever()
