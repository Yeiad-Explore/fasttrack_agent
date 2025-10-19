# Reusable RAG Architecture - Framework-Agnostic Guide

## Table of Contents
1. [Overview](#overview)
2. [Architecture Layers](#architecture-layers)
3. [Layer Details](#layer-details)
4. [Technology Stack Options](#technology-stack-options)
5. [Implementation Patterns](#implementation-patterns)
6. [Configuration & Tuning](#configuration--tuning)
7. [Integration Examples](#integration-examples)

---

## Overview

This document describes a **production-grade RAG (Retrieval-Augmented Generation) architecture** that can be integrated into any application framework (Django, FastAPI, Flask, Express.js, etc.). The architecture is modular and framework-agnostic, focusing on the core RAG pipeline components.

### Core Principles

- **Modularity**: Each layer is independent and swappable
- **Framework-Agnostic**: Works with any web framework or standalone application
- **Production-Ready**: Includes error handling, caching, and monitoring hooks
- **Scalable**: Designed to handle 100K+ documents and concurrent users
- **Configurable**: All parameters externalized for easy tuning

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER                          │
│              (FastAPI / Django / Flask / Express)                 │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATION LAYER                          │
│         (Query Routing, Classification, Enhancement)              │
└────────────────────────────┬─────────────────────────────────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │   RETRIEVAL  │ │   SQL QUERY  │ │  WEB SEARCH  │
    │    LAYER     │ │    LAYER     │ │    LAYER     │
    └──────┬───────┘ └──────────────┘ └──────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│                         RETRIEVAL LAYER                           │
│        (Hybrid Search: Semantic + Keyword + Reranking)            │
└────────────────────────────┬─────────────────────────────────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │   SEMANTIC   │ │   KEYWORD    │ │   RERANKING  │
    │    SEARCH    │ │   SEARCH     │ │   (Cross-    │
    │  (Embeddings)│ │    (BM25)    │ │   Encoder)   │
    └──────┬───────┘ └──────┬───────┘ └──────────────┘
           │                │
           └────────┬───────┘
                    ▼
┌──────────────────────────────────────────────────────────────────┐
│                      VECTOR STORE LAYER                           │
│           (FAISS / Pinecone / Weaviate / Qdrant)                  │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                     EMBEDDING LAYER                               │
│      (Azure OpenAI / OpenAI / HuggingFace / Cohere)               │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                      CHUNKING LAYER                               │
│              (Hybrid Adaptive Chunking Strategy)                  │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                   DATA INGESTION LAYER                            │
│        (Document Loaders, Parsers, Preprocessors)                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## Architecture Layers

The RAG pipeline consists of **8 distinct layers**, each with specific responsibilities:

| Layer | Responsibility | Key Technologies |
|-------|---------------|------------------|
| **1. Data Ingestion** | Load & parse documents | PyPDF, Docling, Unstructured |
| **2. Chunking** | Split documents intelligently | SemanticChunker, RecursiveTextSplitter |
| **3. Embedding** | Convert text to vectors | OpenAI, Cohere, HuggingFace |
| **4. Vector Store** | Store & index embeddings | FAISS, Pinecone, Weaviate, Qdrant |
| **5. Retrieval** | Hybrid search (semantic + keyword) | FAISS, BM25, RRF |
| **6. Reranking** | Fine-tune relevance | Cross-Encoder, LLM-based reranking |
| **7. LLM Interface** | Generate responses | GPT-4, Claude, Llama |
| **8. Orchestration** | Route queries & combine results | LangGraph, Custom logic |

---

## Layer Details

### Layer 1: Data Ingestion

**Purpose**: Load documents from various sources and formats.

**Key Operations**:
- Parse PDFs, DOCX, TXT, HTML, Markdown
- Extract text while preserving structure
- Handle images, tables, charts (optional)
- Clean and normalize text

**Technology Options**:

| Technology | Best For | Pros | Cons |
|------------|----------|------|------|
| **PyPDF2** | Simple PDFs | Fast, lightweight | Limited formatting support |
| **PyMuPDF (fitz)** | Complex PDFs | OCR support, tables | Larger dependency |
| **Docling** | Academic papers | Excellent structure preservation | Slower processing |
| **Unstructured.io** | Multi-format | Handles many formats | Commercial licensing |
| **python-docx** | Word documents | Native DOCX support | DOCX only |

**Code Pattern** (Framework-Agnostic):

```python
from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Document:
    """Unified document representation"""
    content: str
    metadata: dict
    source: str

class DocumentLoader:
    """Abstract base for document loaders"""

    def load(self, file_path: Path) -> Document:
        raise NotImplementedError

class PDFLoader(DocumentLoader):
    """Load PDF documents"""

    def load(self, file_path: Path) -> Document:
        import PyPDF2

        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()

        return Document(
            content=text,
            metadata={
                "num_pages": len(reader.pages),
                "format": "pdf"
            },
            source=str(file_path)
        )

class IngestionPipeline:
    """Main ingestion orchestrator"""

    def __init__(self):
        self.loaders = {
            '.pdf': PDFLoader(),
            '.docx': DOCXLoader(),
            '.txt': TextLoader()
        }

    def ingest(self, file_path: Path) -> Document:
        """Load document based on file extension"""
        suffix = file_path.suffix.lower()
        loader = self.loaders.get(suffix)

        if not loader:
            raise ValueError(f"Unsupported file type: {suffix}")

        return loader.load(file_path)
```

**Best Practices**:
- Validate file types before processing
- Add error handling for corrupted files
- Extract metadata (author, date, page numbers)
- Support batch processing for multiple files
- Implement rate limiting for large uploads

---

### Layer 2: Chunking

**Purpose**: Split documents into semantically meaningful chunks for embedding and retrieval.

**Why Chunking Matters**:
- LLMs have token limits (context windows)
- Smaller chunks = more precise retrieval
- Larger chunks = more context but less precision
- Poor chunking = poor retrieval accuracy

**Chunking Strategies**:

| Strategy | How It Works | Best For | Pros | Cons |
|----------|--------------|----------|------|------|
| **Fixed-Size** | Split every N characters/tokens | Simple documents | Fast, predictable | Breaks semantic boundaries |
| **Recursive Character** | Split on separators (\\n\\n, \\n, . ) | General documents | Preserves structure | Still arbitrary |
| **Semantic Chunking** | Split at topic boundaries using embeddings | Long-form content | Preserves meaning | Slower, requires embeddings |
| **Sentence Window** | Overlapping sentence windows | All documents | Preserves context | Creates more chunks |
| **Hybrid Adaptive** | Semantic + Sentence Window | Production systems | Best accuracy | Most complex |

**Recommended: Hybrid Adaptive Chunking**

This combines **semantic chunking** (finds topic boundaries) with **sentence windowing** (preserves context):

```python
import re
import uuid
from typing import List

class HybridAdaptiveChunker:
    """
    Two-stage chunking:
    1. Semantic chunking finds major topic boundaries
    2. Sentence windowing creates overlapping context
    """

    def __init__(
        self,
        embedding_function,
        semantic_threshold: float = 0.7,
        window_size: int = 5,
        window_overlap: int = 1
    ):
        self.embedding_function = embedding_function
        self.semantic_threshold = semantic_threshold
        self.window_size = window_size
        self.window_overlap = window_overlap

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Handle common abbreviations
        text = re.sub(
            r'\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|etc|e\.g|i\.e)\.\s',
            r'\1<PERIOD> ',
            text
        )

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Restore periods
        sentences = [s.replace('<PERIOD>', '.') for s in sentences]

        return [s.strip() for s in sentences if s.strip()]

    def _semantic_chunking(self, text: str) -> List[str]:
        """Find topic boundaries using embeddings"""
        sentences = self._split_sentences(text)

        if len(sentences) <= 1:
            return [text]

        # Embed all sentences
        embeddings = self.embedding_function.embed_documents(sentences)

        # Calculate similarity between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i+1])
            similarities.append(sim)

        # Find breakpoints (low similarity = topic change)
        threshold = np.percentile(similarities, self.semantic_threshold * 100)
        breakpoints = [0]
        for i, sim in enumerate(similarities):
            if sim < threshold:
                breakpoints.append(i + 1)
        breakpoints.append(len(sentences))

        # Create chunks from breakpoints
        chunks = []
        for i in range(len(breakpoints) - 1):
            start = breakpoints[i]
            end = breakpoints[i + 1]
            chunk = ' '.join(sentences[start:end])
            chunks.append(chunk)

        return chunks

    def _create_sentence_windows(self, sentences: List[str]) -> List[str]:
        """Create overlapping sentence windows"""
        windows = []

        for i in range(0, len(sentences), self.window_size - self.window_overlap):
            window = sentences[i:i + self.window_size]
            if window:
                windows.append(' '.join(window))

        return windows

    def chunk_document(self, document: Document) -> List[Document]:
        """
        Main chunking method:
        1. Semantic chunking for topic boundaries
        2. Sentence windowing within each semantic chunk
        """
        # Step 1: Semantic chunking
        semantic_chunks = self._semantic_chunking(document.content)

        # Step 2: Sentence windowing
        final_chunks = []

        for i, semantic_chunk in enumerate(semantic_chunks):
            sentences = self._split_sentences(semantic_chunk)

            # If chunk is small, keep as-is
            if len(sentences) <= self.window_size:
                final_chunks.append(Document(
                    content=semantic_chunk,
                    metadata={
                        **document.metadata,
                        "chunk_id": str(uuid.uuid4()),
                        "semantic_chunk_index": i,
                        "chunking_strategy": "hybrid_adaptive"
                    },
                    source=document.source
                ))
            else:
                # Create sentence windows
                windows = self._create_sentence_windows(sentences)

                for j, window in enumerate(windows):
                    final_chunks.append(Document(
                        content=window,
                        metadata={
                            **document.metadata,
                            "chunk_id": str(uuid.uuid4()),
                            "semantic_chunk_index": i,
                            "window_index": j,
                            "total_windows": len(windows),
                            "chunking_strategy": "hybrid_adaptive"
                        },
                        source=document.source
                    ))

        return final_chunks

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity"""
        import numpy as np
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
```

**Configuration Guidelines**:

```python
# For technical documentation
SEMANTIC_THRESHOLD = 0.8  # Stricter topic boundaries
WINDOW_SIZE = 7           # More context
WINDOW_OVERLAP = 2        # More overlap

# For conversational content
SEMANTIC_THRESHOLD = 0.6  # Looser boundaries
WINDOW_SIZE = 5           # Moderate context
WINDOW_OVERLAP = 1        # Less overlap

# For short-form content (FAQs, Q&A)
SEMANTIC_THRESHOLD = 0.5  # Very loose
WINDOW_SIZE = 3           # Minimal context
WINDOW_OVERLAP = 1        # Minimal overlap
```

---

### Layer 3: Embedding Generation

**Purpose**: Convert text chunks into high-dimensional vectors that capture semantic meaning.

**How Embeddings Work**:

```
Text: "Student visa requirements for Australia"
  ↓ (Embedding Model)
Vector: [0.123, -0.456, 0.789, ..., 0.234]  (1536 or 3072 dimensions)

Similar Text: "Australian visa application process"
Vector: [0.119, -0.452, 0.791, ..., 0.229]  (Very close in vector space!)

Different Text: "Python programming tutorial"
Vector: [0.891, -0.112, 0.334, ..., 0.667]  (Far away in vector space)
```

**Embedding Model Options**:

| Model | Dimensions | Cost | Performance | Best For |
|-------|-----------|------|-------------|----------|
| **text-embedding-3-large** (OpenAI) | 3072 | $0.13/1M tokens | Excellent | Production systems |
| **text-embedding-3-small** (OpenAI) | 1536 | $0.02/1M tokens | Good | Cost-sensitive apps |
| **text-embedding-ada-002** (OpenAI) | 1536 | $0.10/1M tokens | Good (legacy) | Legacy systems |
| **Cohere embed-multilingual-v3** | 1024 | $0.10/1M tokens | Excellent | Multilingual content |
| **all-MiniLM-L6-v2** (HuggingFace) | 384 | Free (self-hosted) | Moderate | Local/offline systems |
| **BGE-large-en-v1.5** (HuggingFace) | 1024 | Free (self-hosted) | Very Good | Local/offline systems |

**Code Pattern** (Framework-Agnostic):

```python
from typing import List, Protocol
from abc import ABC, abstractmethod

class EmbeddingProvider(Protocol):
    """Protocol for embedding providers"""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        ...

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        ...

class OpenAIEmbeddings:
    """OpenAI embeddings implementation"""

    def __init__(self, api_key: str, model: str = "text-embedding-3-large"):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts"""
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        """Embed single text"""
        return self.embed_documents([text])[0]

class AzureOpenAIEmbeddings:
    """Azure OpenAI embeddings"""

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        deployment: str,
        api_version: str = "2024-02-01"
    ):
        import openai
        self.client = openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )
        self.deployment = deployment

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            input=texts,
            model=self.deployment
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

class HuggingFaceEmbeddings:
    """Local HuggingFace embeddings (free)"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

class EmbeddingService:
    """Unified embedding service with provider abstraction"""

    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider
        self._cache = {}  # Simple cache

    def embed_with_cache(self, text: str) -> List[float]:
        """Embed with caching to save API calls"""
        if text in self._cache:
            return self._cache[text]

        embedding = self.provider.embed_query(text)
        self._cache[text] = embedding
        return embedding
```

**Best Practices**:
- **Cache embeddings** to avoid redundant API calls
- **Batch embedding requests** (most APIs support 100+ texts per request)
- **Monitor costs** for cloud-based models
- **Use dimensionality reduction** if storage is a concern (e.g., PCA)
- **Normalize vectors** for cosine similarity searches

---

### Layer 4: Vector Store

**Purpose**: Store embeddings and enable fast similarity search.

**Vector Store Options**:

| Vector Store | Type | Best For | Pros | Cons |
|--------------|------|----------|------|------|
| **FAISS** | Local/In-memory | Small-medium datasets (<1M vectors) | Free, fast, no API costs | No persistence (must save/load), single machine |
| **Pinecone** | Cloud-managed | Production apps, serverless | Fully managed, scalable, filters | Paid, vendor lock-in |
| **Weaviate** | Self-hosted/Cloud | Large-scale deployments | Open-source, GraphQL API, hybrid search | Complex setup |
| **Qdrant** | Self-hosted/Cloud | High-performance needs | Fast, rich filters, open-source | Requires infrastructure |
| **Chroma** | Local/Self-hosted | Development, small apps | Simple API, free, local-first | Limited scalability |
| **Milvus** | Self-hosted/Cloud | Enterprise-scale | Highly scalable, feature-rich | Complex setup, resource-intensive |

**Recommended for Different Scenarios**:

```python
# Scenario 1: Prototype / Development
# Use: FAISS (local, free, simple)

# Scenario 2: Small Production App (<100K documents)
# Use: FAISS (with persistent storage) or Chroma

# Scenario 3: Medium Production App (100K-1M documents)
# Use: Pinecone (managed) or Qdrant (self-hosted)

# Scenario 4: Large Enterprise (1M+ documents)
# Use: Weaviate or Milvus (self-hosted cluster)
```

**Code Pattern** (FAISS Example):

```python
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

class FAISSVectorStore:
    """FAISS-based vector store with persistence"""

    def __init__(self, dimension: int, index_path: Optional[Path] = None):
        import faiss

        self.dimension = dimension
        self.index_path = index_path

        # Initialize FAISS index (L2 distance)
        self.index = faiss.IndexFlatL2(dimension)

        # Metadata storage (FAISS doesn't store metadata)
        self.metadata = []
        self.documents = []

        # Load existing index if available
        if index_path and index_path.exists():
            self.load()

    def add_documents(
        self,
        embeddings: List[List[float]],
        documents: List[Document]
    ):
        """Add documents with embeddings"""
        # Convert to numpy array
        vectors = np.array(embeddings, dtype='float32')

        # Add to FAISS index
        self.index.add(vectors)

        # Store metadata and documents
        self.metadata.extend([doc.metadata for doc in documents])
        self.documents.extend(documents)

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        # Convert query to numpy array
        query_vec = np.array([query_embedding], dtype='float32')

        # Search in FAISS (returns distances and indices)
        distances, indices = self.index.search(query_vec, k)

        # Convert L2 distance to similarity score
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            # Convert distance to similarity (0-1 scale)
            similarity = 1.0 / (1.0 + dist)

            # Apply threshold if specified
            if threshold and similarity < threshold:
                continue

            results.append((self.documents[idx], float(similarity)))

        return results

    def save(self):
        """Save index and metadata to disk"""
        import faiss

        if not self.index_path:
            raise ValueError("index_path not specified")

        # Create directory
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path))

        # Save metadata and documents
        with open(f"{self.index_path}.meta", 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'documents': self.documents
            }, f)

    def load(self):
        """Load index and metadata from disk"""
        import faiss

        if not self.index_path or not self.index_path.exists():
            raise ValueError("Index file not found")

        # Load FAISS index
        self.index = faiss.read_index(str(self.index_path))

        # Load metadata
        with open(f"{self.index_path}.meta", 'rb') as f:
            data = pickle.load(f)
            self.metadata = data['metadata']
            self.documents = data['documents']

    def get_count(self) -> int:
        """Get number of vectors in index"""
        return self.index.ntotal
```

**Code Pattern** (Pinecone Example):

```python
class PineconeVectorStore:
    """Pinecone cloud vector store"""

    def __init__(
        self,
        api_key: str,
        environment: str,
        index_name: str,
        dimension: int
    ):
        import pinecone

        # Initialize Pinecone
        pinecone.init(api_key=api_key, environment=environment)

        # Create or connect to index
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric='cosine'
            )

        self.index = pinecone.Index(index_name)

    def add_documents(
        self,
        embeddings: List[List[float]],
        documents: List[Document]
    ):
        """Add documents to Pinecone"""
        # Prepare vectors for upsert
        vectors = []
        for i, (embedding, doc) in enumerate(zip(embeddings, documents)):
            vectors.append({
                'id': doc.metadata.get('chunk_id', str(i)),
                'values': embedding,
                'metadata': {
                    'text': doc.content,
                    **doc.metadata
                }
            })

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        metadata_filter: Optional[dict] = None
    ) -> List[Tuple[Document, float]]:
        """Search in Pinecone"""
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
            filter=metadata_filter
        )

        # Convert to documents
        documents = []
        for match in results['matches']:
            doc = Document(
                content=match['metadata'].pop('text'),
                metadata=match['metadata'],
                source=match['metadata'].get('source', '')
            )
            documents.append((doc, match['score']))

        return documents
```

---

### Layer 5: Retrieval Layer (Hybrid Search)

**Purpose**: Combine multiple search strategies for optimal retrieval accuracy.

**Why Hybrid Search?**

Single search methods have limitations:
- **Semantic Search**: Misses exact keywords (e.g., product codes, names)
- **Keyword Search**: Misses synonyms and paraphrases
- **Hybrid Search**: Gets best of both worlds!

**Hybrid Search Architecture**:

```
User Query: "visa requirements"
       │
       ├─────────────────┬─────────────────┐
       │                 │                 │
       ▼                 ▼                 ▼
┌──────────────┐  ┌─────────────┐  ┌─────────────┐
│   SEMANTIC   │  │   KEYWORD   │  │  (Optional) │
│    SEARCH    │  │   SEARCH    │  │   FILTERS   │
│              │  │    (BM25)   │  │             │
│ Embedding    │  │             │  │ metadata:   │
│ + Vector DB  │  │ Tokenize +  │  │ date range, │
│              │  │ Term freq   │  │ source, etc │
└──────┬───────┘  └──────┬──────┘  └─────────────┘
       │                 │
       └────────┬────────┘
                ▼
     ┌────────────────────┐
     │ RECIPROCAL RANK    │
     │    FUSION (RRF)    │
     │                    │
     │ Combines rankings  │
     │ from both sources  │
     └─────────┬──────────┘
               ▼
     ┌────────────────────┐
     │  CROSS-ENCODER     │
     │    RERANKING       │
     │                    │
     │ Fine-grained       │
     │ relevance scoring  │
     └─────────┬──────────┘
               ▼
        Final Results
```

**Code Pattern**:

```python
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Tuple

class HybridRetriever:
    """Hybrid search combining semantic + keyword + reranking"""

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedding_service: EmbeddingService,
        use_reranking: bool = True
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.use_reranking = use_reranking

        # Initialize BM25 index
        self._init_bm25_index()

        # Initialize cross-encoder for reranking (optional)
        if use_reranking:
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def _init_bm25_index(self):
        """Initialize BM25 index from vector store documents"""
        documents = self.vector_store.documents

        # Tokenize all documents
        tokenized_docs = [doc.content.lower().split() for doc in documents]

        # Create BM25 index
        self.bm25_index = BM25Okapi(tokenized_docs)
        self.bm25_documents = documents

    def semantic_search(
        self,
        query: str,
        k: int = 10
    ) -> List[Tuple[Document, float]]:
        """Semantic search using embeddings"""
        # Embed query
        query_embedding = self.embedding_service.embed_query(query)

        # Search in vector store
        results = self.vector_store.similarity_search(
            query_embedding,
            k=k
        )

        return results

    def keyword_search(
        self,
        query: str,
        k: int = 10
    ) -> List[Tuple[Document, float]]:
        """Keyword search using BM25"""
        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores
        scores = self.bm25_index.get_scores(tokenized_query)

        # Get top k indices
        top_indices = np.argsort(scores)[-k:][::-1]

        # Return documents with scores
        results = [
            (self.bm25_documents[idx], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0
        ]

        return results

    def reciprocal_rank_fusion(
        self,
        semantic_results: List[Tuple[Document, float]],
        keyword_results: List[Tuple[Document, float]],
        k: int = 10,
        semantic_weight: float = 0.6
    ) -> List[Document]:
        """
        Combine results using Reciprocal Rank Fusion (RRF)

        RRF Formula:
        score(doc) = Σ weight / (constant + rank)

        This avoids having to normalize scores from different sources.
        """
        keyword_weight = 1.0 - semantic_weight
        rrf_constant = 60  # Standard RRF constant

        doc_scores = {}

        # Add semantic search scores
        for rank, (doc, score) in enumerate(semantic_results):
            doc_id = id(doc)
            rrf_score = semantic_weight / (rrf_constant + rank + 1)

            if doc_id not in doc_scores:
                doc_scores[doc_id] = {'doc': doc, 'score': 0, 'sources': []}

            doc_scores[doc_id]['score'] += rrf_score
            doc_scores[doc_id]['sources'].append('semantic')
            doc_scores[doc_id]['doc'].metadata['semantic_score'] = score

        # Add keyword search scores
        for rank, (doc, score) in enumerate(keyword_results):
            doc_id = id(doc)
            rrf_score = keyword_weight / (rrf_constant + rank + 1)

            if doc_id not in doc_scores:
                doc_scores[doc_id] = {'doc': doc, 'score': 0, 'sources': []}

            doc_scores[doc_id]['score'] += rrf_score
            doc_scores[doc_id]['sources'].append('keyword')
            doc_scores[doc_id]['doc'].metadata['bm25_score'] = score

        # Sort by combined score
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )

        # Add RRF score to metadata
        for item in sorted_docs:
            item['doc'].metadata['rrf_score'] = item['score']
            item['doc'].metadata['retrieval_sources'] = item['sources']

        return [item['doc'] for item in sorted_docs[:k]]

    def cross_encoder_rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int = 5
    ) -> List[Document]:
        """
        Rerank documents using cross-encoder model

        Cross-encoders process query+document together,
        providing more accurate relevance scores.
        """
        if not documents or len(documents) <= top_n:
            return documents[:top_n]

        # Prepare query-document pairs
        pairs = [[query, doc.content[:512]] for doc in documents]

        # Get cross-encoder scores
        scores = self.cross_encoder.predict(pairs)

        # Sort by score
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Add scores to metadata and return top N
        reranked_docs = []
        for doc, score in doc_score_pairs[:top_n]:
            doc.metadata['cross_encoder_score'] = float(score)
            reranked_docs.append(doc)

        return reranked_docs

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        semantic_weight: float = 0.6,
        use_reranking: bool = None
    ) -> List[Document]:
        """
        Main hybrid search method

        Args:
            query: Search query
            k: Number of results to return
            semantic_weight: Weight for semantic search (0-1)
            use_reranking: Whether to use cross-encoder reranking

        Returns:
            List of documents ordered by relevance
        """
        # Use instance setting if not specified
        if use_reranking is None:
            use_reranking = self.use_reranking

        # Perform semantic search
        semantic_results = self.semantic_search(query, k=k*2)

        # Perform keyword search
        keyword_results = self.keyword_search(query, k=k*2)

        # Combine using RRF
        fused_results = self.reciprocal_rank_fusion(
            semantic_results,
            keyword_results,
            k=k*2 if use_reranking else k,
            semantic_weight=semantic_weight
        )

        # Optionally rerank with cross-encoder
        if use_reranking and len(fused_results) > k:
            final_results = self.cross_encoder_rerank(
                query,
                fused_results,
                top_n=k
            )
        else:
            final_results = fused_results[:k]

        return final_results
```

**Configuration Tuning**:

```python
# Balanced (recommended default)
semantic_weight = 0.6  # 60% semantic, 40% keyword
use_reranking = True
k = 5

# Semantic-heavy (for conceptual queries)
semantic_weight = 0.8
use_reranking = True
k = 3

# Keyword-heavy (for specific terms, codes, names)
semantic_weight = 0.4
use_reranking = False  # Faster
k = 10
```

---

### Layer 6: Reranking (Optional but Recommended)

**Purpose**: Fine-tune the final ranking for maximum relevance.

**Why Reranking?**

Initial retrieval methods (semantic + keyword) are fast but approximate. Reranking provides:
- More accurate relevance scoring
- Query-document interaction modeling
- Better handling of nuanced queries

**Reranking Options**:

| Method | How It Works | Pros | Cons |
|--------|--------------|------|------|
| **Cross-Encoder** | Process query+doc together through transformer | Most accurate | Slower, must process each pair |
| **LLM-based** | Use GPT-4/Claude to rank | Very accurate, flexible | Expensive, slow |
| **ColBERT** | Late interaction model | Good balance | Complex setup |

**Cross-Encoder Implementation** (shown above in HybridRetriever)

**LLM-based Reranking**:

```python
class LLMReranker:
    """Rerank using LLM (GPT-4, Claude, etc.)"""

    def __init__(self, llm_client):
        self.llm = llm_client

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int = 5
    ) -> List[Document]:
        """Rerank documents using LLM"""
        if len(documents) <= top_n:
            return documents

        # Prepare document list for LLM
        doc_list = "\n\n".join([
            f"[{i}] {doc.content[:300]}..."
            for i, doc in enumerate(documents[:10])
        ])

        prompt = f"""Given the following query and documents, rank the documents by relevance.
        Return only the indices of the top {top_n} documents in order, separated by commas.

        Query: {query}

        Documents:
        {doc_list}

        Top {top_n} indices (comma-separated):"""

        response = self.llm.generate(prompt)

        # Parse indices
        indices = [
            int(idx.strip())
            for idx in response.split(',')
            if idx.strip().isdigit()
        ]

        # Return reranked documents
        return [documents[idx] for idx in indices[:top_n] if idx < len(documents)]
```

---

### Layer 7: LLM Interface

**Purpose**: Generate natural language responses using retrieved context.

**LLM Options**:

| Model | Provider | Best For | Cost (per 1M tokens) | Context Window |
|-------|----------|----------|---------------------|----------------|
| **GPT-4 Turbo** | OpenAI | Production apps | $10 input / $30 output | 128K |
| **GPT-4o** | OpenAI | Fast production apps | $2.50 input / $10 output | 128K |
| **GPT-3.5 Turbo** | OpenAI | Cost-sensitive apps | $0.50 input / $1.50 output | 16K |
| **Claude 3.5 Sonnet** | Anthropic | High-quality responses | $3 input / $15 output | 200K |
| **Claude 3 Haiku** | Anthropic | Fast, cheap | $0.25 input / $1.25 output | 200K |
| **Llama 3 70B** | Meta (self-hosted) | Open-source | Free (infra costs) | 8K |
| **Mixtral 8x7B** | Mistral (self-hosted) | Open-source | Free (infra costs) | 32K |

**Code Pattern**:

```python
from typing import List, Optional

class LLMInterface:
    """Unified LLM interface for RAG"""

    def __init__(self, provider: str, **kwargs):
        self.provider = provider

        if provider == "openai":
            import openai
            self.client = openai.OpenAI(api_key=kwargs['api_key'])
            self.model = kwargs.get('model', 'gpt-4-turbo')

        elif provider == "azure_openai":
            import openai
            self.client = openai.AzureOpenAI(
                api_key=kwargs['api_key'],
                azure_endpoint=kwargs['endpoint'],
                api_version=kwargs.get('api_version', '2024-02-01')
            )
            self.model = kwargs['deployment']

        elif provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=kwargs['api_key'])
            self.model = kwargs.get('model', 'claude-3-5-sonnet-20241022')

    def generate(
        self,
        query: str,
        context: List[Document],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000
    ) -> str:
        """Generate response using retrieved context"""

        # Build context string
        context_str = self._format_context(context)

        # Build system prompt
        if not system_prompt:
            system_prompt = self._default_system_prompt()

        # Build user message
        user_message = f"""Context:
{context_str}

User Query: {query}

Please provide a comprehensive answer based on the context above."""

        # Generate response based on provider
        if self.provider in ["openai", "azure_openai"]:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content

        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.content[0].text

    def _format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents as context"""
        formatted = []

        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', '')
            page_str = f" (page {page})" if page else ""

            formatted.append(f"""[{i}] Source: {source}{page_str}
{doc.content}
""")

        return "\n---\n".join(formatted)

    def _default_system_prompt(self) -> str:
        """Default system prompt for RAG"""
        return """You are a helpful AI assistant. Answer the user's question based on the provided context.

Guidelines:
- Use only information from the provided context
- If the context doesn't contain enough information, say so
- Cite sources by referencing [1], [2], etc.
- Be concise but comprehensive
- If the context is contradictory, acknowledge the contradiction"""
```

---

### Layer 8: Orchestration

**Purpose**: Route queries to appropriate pipelines and combine results.

**Query Types**:
- **Semantic RAG**: "What are the visa requirements?" (needs document retrieval)
- **SQL Query**: "How many students enrolled in 2023?" (needs database query)
- **Hybrid**: "Show me universities in Sydney with high acceptance rates" (needs both)
- **Web Search**: "Latest Australia visa policy changes 2025" (needs real-time info)

**Code Pattern**:

```python
from enum import Enum
from typing import Optional, List

class QueryType(Enum):
    SEMANTIC_RAG = "semantic_rag"
    SQL_QUERY = "sql_query"
    HYBRID = "hybrid"
    WEB_SEARCH = "web_search"

class QueryClassifier:
    """Classify queries to route to appropriate pipeline"""

    def __init__(self, llm_interface: LLMInterface):
        self.llm = llm_interface

    def classify(self, query: str) -> QueryType:
        """Classify query type"""

        # Simple keyword-based classification first
        query_lower = query.lower()

        # SQL indicators
        sql_keywords = ['how many', 'count', 'total', 'average', 'list all', 'show all']
        if any(kw in query_lower for kw in sql_keywords):
            return QueryType.SQL_QUERY

        # Web search indicators
        web_keywords = ['latest', 'recent', 'news', 'current', '2025', 'today']
        if any(kw in query_lower for kw in web_keywords):
            return QueryType.WEB_SEARCH

        # Default to semantic RAG
        return QueryType.SEMANTIC_RAG

class Orchestrator:
    """Main orchestrator for RAG pipeline"""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_interface: LLMInterface,
        sql_engine: Optional[Any] = None,
        web_search: Optional[Any] = None
    ):
        self.retriever = retriever
        self.llm = llm_interface
        self.sql_engine = sql_engine
        self.web_search = web_search
        self.classifier = QueryClassifier(llm_interface)

    def process_query(
        self,
        query: str,
        conversation_history: Optional[List[dict]] = None
    ) -> dict:
        """
        Main query processing method

        Returns:
            {
                'answer': str,
                'sources': List[Document],
                'query_type': str,
                'metadata': dict
            }
        """
        # Classify query
        query_type = self.classifier.classify(query)

        # Route to appropriate pipeline
        if query_type == QueryType.SEMANTIC_RAG:
            return self._process_rag_query(query, conversation_history)

        elif query_type == QueryType.SQL_QUERY:
            return self._process_sql_query(query)

        elif query_type == QueryType.WEB_SEARCH:
            return self._process_web_search_query(query)

        elif query_type == QueryType.HYBRID:
            return self._process_hybrid_query(query, conversation_history)

    def _process_rag_query(
        self,
        query: str,
        conversation_history: Optional[List[dict]] = None
    ) -> dict:
        """Process semantic RAG query"""

        # Retrieve relevant documents
        documents = self.retriever.hybrid_search(query, k=5)

        # Generate response
        answer = self.llm.generate(
            query=query,
            context=documents,
            temperature=0.3
        )

        return {
            'answer': answer,
            'sources': documents,
            'query_type': 'semantic_rag',
            'metadata': {
                'num_sources': len(documents),
                'retrieval_method': 'hybrid_search'
            }
        }

    def _process_sql_query(self, query: str) -> dict:
        """Process SQL query"""
        if not self.sql_engine:
            return {
                'answer': "SQL querying not configured",
                'sources': [],
                'query_type': 'sql_query',
                'metadata': {}
            }

        # Generate and execute SQL
        sql_result = self.sql_engine.execute(query)

        # Format result with LLM
        answer = self.llm.generate(
            query=f"Format this SQL result as a natural language answer to: {query}\n\nResult: {sql_result}",
            context=[],
            temperature=0.1
        )

        return {
            'answer': answer,
            'sources': [],
            'query_type': 'sql_query',
            'metadata': {
                'sql_result': sql_result
            }
        }

    def _process_web_search_query(self, query: str) -> dict:
        """Process web search query"""
        if not self.web_search:
            return {
                'answer': "Web search not configured",
                'sources': [],
                'query_type': 'web_search',
                'metadata': {}
            }

        # Perform web search
        web_results = self.web_search.search(query, k=5)

        # Generate answer from web results
        answer = self.llm.generate(
            query=query,
            context=web_results,
            temperature=0.3
        )

        return {
            'answer': answer,
            'sources': web_results,
            'query_type': 'web_search',
            'metadata': {
                'num_sources': len(web_results)
            }
        }
```

---

## Technology Stack Options

### Complete Stack Recommendations

**Stack 1: Simple & Free (Development/Prototype)**
```yaml
Framework: FastAPI
Ingestion: PyPDF2, python-docx
Chunking: RecursiveCharacterTextSplitter
Embeddings: HuggingFace (all-MiniLM-L6-v2)
Vector Store: FAISS (local)
Retrieval: Semantic only
LLM: Llama 3 (local) or GPT-3.5-turbo
Total Cost: ~$0 (except GPT API calls)
```

**Stack 2: Production-Ready (Small-Medium Scale)**
```yaml
Framework: FastAPI or Django
Ingestion: PyMuPDF, Docling
Chunking: Hybrid Adaptive (Semantic + Sentence Window)
Embeddings: text-embedding-3-large (OpenAI/Azure)
Vector Store: FAISS (persistent) or Chroma
Retrieval: Hybrid (Semantic + BM25 + RRF)
Reranking: Cross-Encoder (ms-marco-MiniLM-L-6-v2)
LLM: GPT-4-turbo or Claude 3.5 Sonnet
Total Cost: ~$50-200/month (depending on usage)
```

**Stack 3: Enterprise Scale (Large Production)**
```yaml
Framework: FastAPI (async) with Redis caching
Ingestion: Unstructured.io (multi-format)
Chunking: Hybrid Adaptive with custom tuning
Embeddings: text-embedding-3-large (Azure OpenAI)
Vector Store: Pinecone (managed) or Qdrant (self-hosted cluster)
Retrieval: Hybrid + Metadata Filtering + Reranking
Reranking: ColBERT or Cross-Encoder
LLM: GPT-4-turbo with fallback to GPT-4o
Monitoring: Langfuse or LangSmith
Total Cost: $500-5000/month (depending on scale)
```

---

## Configuration & Tuning

### Key Parameters

```python
# config.py - Framework-agnostic configuration

import os
from dataclasses import dataclass

@dataclass
class RAGConfig:
    """Centralized RAG configuration"""

    # ========== EMBEDDING SETTINGS ==========
    EMBEDDING_PROVIDER: str = "azure_openai"  # openai, azure_openai, huggingface, cohere
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_DIMENSIONS: int = 3072
    EMBEDDING_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY")
    EMBEDDING_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT")

    # ========== CHUNKING SETTINGS ==========
    CHUNKING_STRATEGY: str = "hybrid_adaptive"  # fixed, recursive, semantic, hybrid_adaptive
    SEMANTIC_BREAKPOINT_TYPE: str = "percentile"  # percentile, standard_deviation, interquartile
    SEMANTIC_BREAKPOINT_THRESHOLD: float = 0.7  # 0-1 (higher = stricter topic boundaries)
    SENTENCE_WINDOW_SIZE: int = 5  # Number of sentences per window
    SENTENCE_OVERLAP: int = 1  # Sentence overlap between windows

    # ========== VECTOR STORE SETTINGS ==========
    VECTOR_STORE_TYPE: str = "faiss"  # faiss, pinecone, weaviate, qdrant, chroma
    VECTOR_STORE_PATH: str = "./vector_store"  # For local stores
    VECTOR_STORE_PERSIST: bool = True  # Save to disk

    # ========== RETRIEVAL SETTINGS ==========
    RETRIEVAL_METHOD: str = "hybrid"  # semantic, keyword, hybrid
    SEMANTIC_WEIGHT: float = 0.6  # 0-1 (weight for semantic vs keyword)
    RETRIEVER_K: int = 5  # Number of documents to retrieve
    SIMILARITY_THRESHOLD: float = 0.7  # Minimum similarity score (0-1)

    # ========== RERANKING SETTINGS ==========
    USE_RERANKING: bool = True  # Enable cross-encoder reranking
    RERANKING_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKING_TOP_N: int = 5  # Final number of documents after reranking

    # ========== LLM SETTINGS ==========
    LLM_PROVIDER: str = "azure_openai"  # openai, azure_openai, anthropic, local
    LLM_MODEL: str = "gpt-4-turbo"
    LLM_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY")
    LLM_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT")
    LLM_TEMPERATURE: float = 0.3  # 0-1 (higher = more creative)
    LLM_MAX_TOKENS: int = 1000  # Maximum response length

    # ========== QUERY PROCESSING ==========
    ENABLE_QUERY_CLASSIFICATION: bool = True  # Classify query type
    ENABLE_QUERY_ENHANCEMENT: bool = True  # Expand queries with variations
    ENABLE_WEB_SEARCH_FALLBACK: bool = False  # Fall back to web search if no results

    # ========== PERFORMANCE SETTINGS ==========
    BATCH_SIZE: int = 100  # Embedding batch size
    MAX_CONCURRENT_REQUESTS: int = 10  # Async request limit
    CACHE_EMBEDDINGS: bool = True  # Cache embeddings
    CACHE_TTL: int = 3600  # Cache time-to-live (seconds)

    # ========== MONITORING & LOGGING ==========
    ENABLE_MONITORING: bool = True  # Enable performance monitoring
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    LOG_QUERIES: bool = True  # Log all queries
    LOG_RETRIEVAL_SCORES: bool = False  # Log detailed retrieval scores

# Singleton config
config = RAGConfig()
```

### Tuning Guidelines

**For Different Content Types**:

```python
# Technical Documentation
config.SEMANTIC_BREAKPOINT_THRESHOLD = 0.8  # Stricter boundaries
config.SENTENCE_WINDOW_SIZE = 7  # More context
config.SEMANTIC_WEIGHT = 0.7  # Favor semantic search
config.USE_RERANKING = True

# Conversational/Chat Logs
config.SEMANTIC_BREAKPOINT_THRESHOLD = 0.6  # Looser boundaries
config.SENTENCE_WINDOW_SIZE = 5
config.SEMANTIC_WEIGHT = 0.5  # Balanced
config.USE_RERANKING = False  # Faster

# FAQs/Q&A
config.SEMANTIC_BREAKPOINT_THRESHOLD = 0.5  # Very loose
config.SENTENCE_WINDOW_SIZE = 3  # Minimal context
config.SEMANTIC_WEIGHT = 0.6
config.USE_RERANKING = True  # Important for accuracy

# Legal/Financial Documents
config.SEMANTIC_BREAKPOINT_THRESHOLD = 0.9  # Very strict
config.SENTENCE_WINDOW_SIZE = 10  # Maximum context
config.SEMANTIC_WEIGHT = 0.4  # Favor exact keyword matches
config.USE_RERANKING = True
config.SIMILARITY_THRESHOLD = 0.8  # Higher threshold
```

**For Different Query Types**:

```python
# Factual Queries ("What is X?")
config.RETRIEVER_K = 3
config.LLM_TEMPERATURE = 0.1  # More deterministic
config.USE_RERANKING = True

# Exploratory Queries ("Tell me about X")
config.RETRIEVER_K = 10
config.LLM_TEMPERATURE = 0.5  # More creative
config.USE_RERANKING = True

# Comparison Queries ("Compare X and Y")
config.RETRIEVER_K = 8
config.LLM_TEMPERATURE = 0.3  # Balanced
config.SEMANTIC_WEIGHT = 0.7  # Favor semantic understanding
```

---

## Integration Examples

### Example 1: FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="RAG API")

# Initialize RAG components
embedding_service = EmbeddingService(
    provider=OpenAIEmbeddings(api_key=config.EMBEDDING_API_KEY)
)

vector_store = FAISSVectorStore(
    dimension=config.EMBEDDING_DIMENSIONS,
    index_path=Path(config.VECTOR_STORE_PATH)
)

retriever = HybridRetriever(
    vector_store=vector_store,
    embedding_service=embedding_service,
    use_reranking=config.USE_RERANKING
)

llm_interface = LLMInterface(
    provider=config.LLM_PROVIDER,
    api_key=config.LLM_API_KEY,
    model=config.LLM_MODEL
)

orchestrator = Orchestrator(
    retriever=retriever,
    llm_interface=llm_interface
)

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    conversation_history: Optional[List[dict]] = None
    k: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    query_type: str
    metadata: dict

# API endpoints
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a query using RAG"""
    try:
        result = orchestrator.process_query(
            query=request.query,
            conversation_history=request.conversation_history
        )

        return QueryResponse(
            answer=result['answer'],
            sources=[{
                'content': doc.content,
                'metadata': doc.metadata,
                'source': doc.source
            } for doc in result['sources']],
            query_type=result['query_type'],
            metadata=result['metadata']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(file: UploadFile):
    """Upload and index a document"""
    try:
        # Save uploaded file
        file_path = Path(f"./uploads/{file.filename}")
        file_path.parent.mkdir(exist_ok=True)

        with open(file_path, 'wb') as f:
            f.write(await file.read())

        # Ingest document
        ingestion_pipeline = IngestionPipeline()
        document = ingestion_pipeline.ingest(file_path)

        # Chunk document
        chunker = HybridAdaptiveChunker(
            embedding_function=embedding_service.provider,
            semantic_threshold=config.SEMANTIC_BREAKPOINT_THRESHOLD,
            window_size=config.SENTENCE_WINDOW_SIZE,
            window_overlap=config.SENTENCE_OVERLAP
        )
        chunks = chunker.chunk_document(document)

        # Embed and add to vector store
        embeddings = embedding_service.embed_documents([c.content for c in chunks])
        vector_store.add_documents(embeddings, chunks)

        return {
            "message": "Document uploaded and indexed successfully",
            "chunks_created": len(chunks),
            "filename": file.filename
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get vector store statistics"""
    return {
        "total_chunks": vector_store.get_count(),
        "embedding_dimensions": config.EMBEDDING_DIMENSIONS,
        "retrieval_method": config.RETRIEVAL_METHOD,
        "reranking_enabled": config.USE_RERANKING
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Example 2: Django Integration

```python
# views.py
from django.views import View
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import json

@method_decorator(csrf_exempt, name='dispatch')
class QueryView(View):
    """RAG query endpoint for Django"""

    def __init__(self):
        super().__init__()
        # Initialize RAG components (same as FastAPI example)
        self.orchestrator = self._init_orchestrator()

    def _init_orchestrator(self):
        """Initialize RAG orchestrator"""
        embedding_service = EmbeddingService(
            provider=OpenAIEmbeddings(api_key=config.EMBEDDING_API_KEY)
        )

        vector_store = FAISSVectorStore(
            dimension=config.EMBEDDING_DIMENSIONS,
            index_path=Path(config.VECTOR_STORE_PATH)
        )

        retriever = HybridRetriever(
            vector_store=vector_store,
            embedding_service=embedding_service,
            use_reranking=config.USE_RERANKING
        )

        llm_interface = LLMInterface(
            provider=config.LLM_PROVIDER,
            api_key=config.LLM_API_KEY,
            model=config.LLM_MODEL
        )

        return Orchestrator(
            retriever=retriever,
            llm_interface=llm_interface
        )

    def post(self, request):
        """Handle POST requests for queries"""
        try:
            data = json.loads(request.body)
            query = data.get('query')

            if not query:
                return JsonResponse({'error': 'Query is required'}, status=400)

            # Process query
            result = self.orchestrator.process_query(
                query=query,
                conversation_history=data.get('conversation_history')
            )

            return JsonResponse({
                'answer': result['answer'],
                'sources': [{
                    'content': doc.content,
                    'metadata': doc.metadata,
                    'source': doc.source
                } for doc in result['sources']],
                'query_type': result['query_type'],
                'metadata': result['metadata']
            })

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

# urls.py
from django.urls import path
from .views import QueryView

urlpatterns = [
    path('api/query/', QueryView.as_view(), name='query'),
]
```

### Example 3: Standalone CLI Application

```python
# cli.py
import click
from pathlib import Path

@click.group()
def cli():
    """RAG CLI Application"""
    pass

@cli.command()
@click.argument('query')
@click.option('--k', default=5, help='Number of results')
def query(query, k):
    """Query the RAG system"""
    # Initialize orchestrator (same as above)
    orchestrator = init_orchestrator()

    result = orchestrator.process_query(query)

    click.echo(f"\nAnswer:\n{result['answer']}\n")
    click.echo(f"Sources ({len(result['sources'])}):")
    for i, doc in enumerate(result['sources'], 1):
        click.echo(f"  [{i}] {doc.source}")

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
def index(file_path):
    """Index a document"""
    # Initialize components
    ingestion_pipeline = IngestionPipeline()
    chunker = HybridAdaptiveChunker(...)
    embedding_service = EmbeddingService(...)
    vector_store = FAISSVectorStore(...)

    # Process document
    document = ingestion_pipeline.ingest(Path(file_path))
    chunks = chunker.chunk_document(document)
    embeddings = embedding_service.embed_documents([c.content for c in chunks])
    vector_store.add_documents(embeddings, chunks)

    click.echo(f"Indexed {len(chunks)} chunks from {file_path}")

@cli.command()
def stats():
    """Show vector store statistics"""
    vector_store = FAISSVectorStore(...)
    click.echo(f"Total chunks: {vector_store.get_count()}")

if __name__ == '__main__':
    cli()
```

---

## Summary

This RAG architecture is:

- **Framework-Agnostic**: Works with FastAPI, Django, Flask, Express.js, or standalone
- **Production-Ready**: Includes all necessary layers for real-world applications
- **Scalable**: Designed to handle 100K+ documents
- **Configurable**: All parameters externalized for easy tuning
- **Hybrid**: Combines semantic and keyword search for best accuracy
- **Modular**: Each layer is independent and swappable

### Quick Start Checklist

1. **Choose your stack** (see Technology Stack Options)
2. **Set up configuration** (copy config.py template)
3. **Initialize components**:
   - Document loader
   - Chunker
   - Embedding service
   - Vector store
   - Retriever
   - LLM interface
   - Orchestrator
4. **Index your documents**
5. **Test queries**
6. **Tune parameters** based on your use case
7. **Deploy** to your framework (FastAPI/Django/etc.)

### Key Files to Implement

```
your-project/
├── config.py                 # Configuration
├── ingestion.py             # Document ingestion layer
├── chunking.py              # Chunking strategies
├── embeddings.py            # Embedding service
├── vector_store.py          # Vector store interface
├── retrieval.py             # Hybrid retrieval
├── llm_interface.py         # LLM interface
├── orchestration.py         # Query orchestration
└── main.py                  # Framework integration (FastAPI/Django/etc.)
```

---

**Version**: 1.0
**Last Updated**: January 2025
**Maintained By**: [Your Team]
**License**: MIT
