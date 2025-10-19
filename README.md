# Fast Track Communication & Support - AI Agent

A production-ready AI customer service agent for Fast Track courier service, built with advanced RAG (Retrieval-Augmented Generation) architecture.

## Features

- **Intelligent Customer Support**: Answers queries about shipping, tracking, rates, and more
- **Hybrid RAG Pipeline**: Combines semantic search, BM25, and cross-encoder reranking
- **Document Processing**: Automatic PDF ingestion using Docling
- **Azure OpenAI Integration**: Powered by GPT-4.1 for accurate responses
- **Conversation Memory**: Maintains context across chat sessions
- **Modern UI**: Clean, responsive chat interface
- **REST API**: Complete FastAPI backend with OpenAPI documentation

## Architecture

### RAG Pipeline
1. **Document Ingestion**: Docling extracts text from PDFs
2. **Chunking**: Hybrid adaptive chunking (sentence-aware)
3. **Embeddings**: Azure OpenAI text-embedding-3-large
4. **Vector Store**: FAISS with HNSW indexing
5. **Retrieval**: Hybrid search (Semantic + BM25 + Cross-Encoder)
6. **Generation**: Azure OpenAI GPT-4.1

### Components
- **Backend**: FastAPI (Python)
- **Document Parser**: Docling
- **Vector Database**: FAISS (persistent)
- **LLM**: Azure OpenAI GPT-4.1
- **Frontend**: HTML/CSS/JavaScript

## Project Structure

```
fasttrack_agent/
├── kb/                          # Knowledge base PDFs
├── vector_store/                # FAISS index (auto-created)
├── uploads/                     # Temporary uploads (auto-created)
├── static/                      # Frontend files
│   ├── index.html              # Chat interface
│   ├── style.css               # Styles
│   └── app.js                  # JavaScript logic
├── src/                         # Backend source code
│   ├── config.py               # Configuration
│   ├── embeddings.py           # Embedding service
│   ├── chunking.py             # Document chunking
│   ├── vector_store.py         # FAISS vector store
│   ├── ingestion.py            # PDF ingestion pipeline
│   ├── retrieval.py            # Hybrid retrieval
│   ├── llm_interface.py        # Azure OpenAI interface
│   ├── orchestration.py        # Query orchestration
│   └── main.py                 # FastAPI application
├── requirements.txt             # Python dependencies
├── .env                        # Environment variables
└── README.md                   # This file
```

## Setup Instructions

### Prerequisites
- Python 3.9 or higher
- Azure OpenAI API access
- Git (optional)

### Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd f:\Work\FastTrack\fasttrack_agent
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Your `.env` file is already configured with Azure OpenAI credentials:
   ```env
   AZURE_OPENAI_API_KEY=your_key_here
   AZURE_OPENAI_ENDPOINT=your_endpoint_here
   AZURE_OPENAI_API_VERSION=2025-01-01-preview
   CHAT_MODEL_DEPLOYMENT=chat-heavy
   EMBEDDING_MODEL_DEPLOYMENT=embed-large
   ```

5. **Add PDF documents to the knowledge base**

   Place your PDF files (shipping policies, FAQs, service guides, etc.) in the `kb/` folder.

## Usage

### Starting the Server

```bash
cd src
python main.py
```

The server will start on `http://localhost:8000`

### Accessing the Application

- **Chat Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### Indexing Knowledge Base

**Option 1: Using the Web Interface**
1. Open http://localhost:8000
2. Click "Index KB Folder" in the sidebar
3. Wait for indexing to complete

**Option 2: Using the API**
```bash
curl -X POST http://localhost:8000/api/index-kb
```

**Option 3: Upload individual PDFs**
- Use the "Upload PDF" button in the web interface
- Or via API: POST to `/api/upload` with file

### Asking Questions

**Via Web Interface:**
1. Open http://localhost:8000
2. Type your question in the chat input
3. Press Enter or click Send

**Via API:**
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are your shipping rates?", "session_id": "test-session"}'
```

## API Endpoints

### Query Processing
- `POST /api/query` - Process customer query
  ```json
  {
    "query": "How do I track my package?",
    "session_id": "optional-session-id",
    "stream": false,
    "top_k": 5
  }
  ```

### Knowledge Base Management
- `POST /api/index-kb` - Index all PDFs in kb/ folder
- `POST /api/upload` - Upload and index a single PDF
- `GET /api/stats` - Get vector store statistics

### Conversation Management
- `DELETE /api/clear-history/{session_id}` - Clear conversation history

### Health Check
- `GET /api/health` - Check service status

## Configuration

Edit [src/config.py](src/config.py) to customize:

- `CHUNK_SIZE`: Maximum tokens per chunk (default: 512)
- `CHUNK_OVERLAP`: Overlapping tokens (default: 128)
- `TOP_K_RETRIEVAL`: Documents to retrieve (default: 10)
- `TOP_K_RERANK`: Final documents after reranking (default: 5)
- `SIMILARITY_THRESHOLD`: Minimum similarity score (default: 0.7)
- `MAX_TOKENS`: Maximum response tokens (default: 2000)
- `TEMPERATURE`: LLM temperature (default: 0.7)

## Features Breakdown

### Document Ingestion
- Automatic PDF parsing with Docling
- Preserves document structure
- Batch processing support
- Progress tracking

### Chunking Strategy
- Sentence-aware splitting
- Configurable chunk size and overlap
- Handles large documents
- Preserves context

### Retrieval System
- **Semantic Search**: Vector similarity using embeddings
- **BM25**: Traditional keyword-based search
- **Cross-Encoder Reranking**: Advanced relevance scoring
- **Hybrid Scoring**: Weighted combination of all methods

### LLM Generation
- Context-aware responses
- Source citation
- Conversation history support
- Customizable system prompts

## Troubleshooting

### Common Issues

**Issue: "Module not found" errors**
- Solution: Ensure you're in the virtual environment and all dependencies are installed
  ```bash
  pip install -r requirements.txt
  ```

**Issue: "No PDF files found in kb/"**
- Solution: Add PDF files to the `kb/` folder before indexing

**Issue: "Azure OpenAI authentication failed"**
- Solution: Verify your `.env` file has correct credentials

**Issue: Port 8000 already in use**
- Solution: Change the port in [src/config.py](src/config.py) or stop the other process

**Issue: Slow indexing**
- Solution: This is normal for large PDFs. Docling processing is thorough but can take time.

**Issue: Vector store not persisting**
- Solution: Check that `vector_store/` directory exists and has write permissions

## Development

### Running Tests
```bash
# Add tests in tests/ directory
pytest
```

### Code Style
```bash
# Format code
black src/

# Lint code
flake8 src/
```

### Adding New Features

1. **Custom prompts**: Edit the system prompt in [src/llm_interface.py](src/llm_interface.py)
2. **New endpoints**: Add routes in [src/main.py](src/main.py)
3. **Different chunking**: Modify [src/chunking.py](src/chunking.py)
4. **Alternative retrieval**: Update [src/retrieval.py](src/retrieval.py)

## Performance Optimization

### For Large Knowledge Bases
- Increase `TOP_K_RETRIEVAL` for better recall
- Adjust `SIMILARITY_THRESHOLD` to filter low-quality results
- Use GPU-accelerated FAISS for faster search

### For Faster Responses
- Reduce `TOP_K_RERANK` for quicker reranking
- Lower `MAX_TOKENS` for shorter responses
- Cache frequently asked questions

## Deployment

### Production Checklist
- [ ] Set `reload=False` in [src/main.py](src/main.py)
- [ ] Use production ASGI server (e.g., Gunicorn)
- [ ] Enable HTTPS
- [ ] Set up monitoring and logging
- [ ] Configure backup for vector store
- [ ] Set rate limiting
- [ ] Use environment-specific configurations

### Docker Deployment
```dockerfile
# Create Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## License

This project is proprietary software for Fast Track Communication & Support.

## Support

For issues or questions:
- Email: support@fasttrack.com
- Phone: Your support line
- Documentation: This README

---

**Built with Azure OpenAI and FastAPI**

Version 1.0.0 - 2025
