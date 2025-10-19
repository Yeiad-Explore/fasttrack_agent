"""
Embeddings service using Azure OpenAI
Handles text embedding generation for the RAG pipeline
"""

import os
from typing import List
from openai import AzureOpenAI
from config import settings
import numpy as np


class EmbeddingService:
    """Service for generating embeddings using Azure OpenAI"""

    def __init__(self):
        """Initialize Azure OpenAI client"""
        self.client = AzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
        )
        self.deployment = settings.EMBEDDING_MODEL_DEPLOYMENT
        self.dimension = 3072  # text-embedding-3-large dimension

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text

        Args:
            text: Input text to embed

        Returns:
            numpy array of embedding vector
        """
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.deployment
            )
            embedding = response.data[0].embedding
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise

    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts in batches

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process per batch

        Returns:
            List of numpy arrays containing embeddings
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.deployment
                )
                batch_embeddings = [
                    np.array(item.embedding, dtype=np.float32)
                    for item in response.data
                ]
                embeddings.extend(batch_embeddings)
                print(f"Embedded batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            except Exception as e:
                print(f"Error embedding batch {i//batch_size + 1}: {e}")
                raise

        return embeddings

    def get_dimension(self) -> int:
        """Get the dimension of the embedding vectors"""
        return self.dimension


# Create singleton instance
embedding_service = EmbeddingService()
