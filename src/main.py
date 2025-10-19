"""
FastAPI Main Application
Provides REST API endpoints for the Fast Track AI Agent
"""

import os
import sys
from pathlib import Path
from typing import Optional
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from orchestration import orchestrator
from ingestion import ingestion_pipeline
from vector_store import vector_store


# Initialize FastAPI app
app = FastAPI(
    title="Fast Track AI Agent",
    description="AI-powered customer service agent for Fast Track Communication & Support",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str
    session_id: Optional[str] = "default"
    stream: Optional[bool] = False
    top_k: Optional[int] = 5


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    answer: str
    sources: list
    context_used: int
    session_id: str


class IndexResponse(BaseModel):
    """Response model for index endpoint"""
    message: str
    total_files: int
    total_chunks: int
    files_processed: list


class StatsResponse(BaseModel):
    """Response model for stats endpoint"""
    total_documents: int
    index_size: int
    dimension: int
    index_type: Optional[str]


# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend HTML page"""
    index_path = settings.STATIC_DIR / "index.html"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Fast Track AI Agent</h1><p>Frontend not found. Please ensure static files are present.</p>"


@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a user query and return AI-generated response

    Args:
        request: Query request containing question and session info

    Returns:
        Response with answer, sources, and metadata
    """
    try:
        # Process the query
        result = orchestrator.process_query(
            query=request.query,
            session_id=request.session_id,
            stream=request.stream,
            top_k=request.top_k
        )

        # Handle streaming responses
        if request.stream and "stream" in result:
            async def generate():
                for chunk in result["stream"]:
                    yield chunk

            return StreamingResponse(generate(), media_type="text/plain")

        # Return regular response
        return QueryResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            context_used=result.get("context_used", 0),
            session_id=request.session_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/index-kb", response_model=IndexResponse)
async def index_knowledge_base():
    """
    Index all PDF documents in the kb/ folder

    Returns:
        Statistics about the indexing process
    """
    try:
        result = ingestion_pipeline.ingest_directory()

        return IndexResponse(
            message="Knowledge base indexed successfully",
            total_files=result["total_files"],
            total_chunks=result["total_chunks"],
            files_processed=result["files_processed"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and index a single PDF document

    Args:
        file: PDF file to upload

    Returns:
        Upload status and statistics
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        # Save file to kb directory
        file_path = settings.KB_DIR / file.filename

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Ingest the document
        num_chunks = ingestion_pipeline.ingest_document(file_path)

        return {
            "message": "Document uploaded and indexed successfully",
            "filename": file.filename,
            "chunks_created": num_chunks
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get statistics about the vector store

    Returns:
        Vector store statistics
    """
    try:
        stats = vector_store.get_stats()

        return StatsResponse(
            total_documents=stats["total_documents"],
            index_size=stats["index_size"],
            dimension=stats["dimension"],
            index_type=stats.get("index_type")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/clear-history/{session_id}")
async def clear_history(session_id: str = "default"):
    """
    Clear conversation history for a session

    Args:
        session_id: Session identifier

    Returns:
        Success message
    """
    try:
        orchestrator.clear_conversation_history(session_id)
        return {"message": f"Conversation history cleared for session: {session_id}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """
    Health check endpoint

    Returns:
        Service health status
    """
    return {
        "status": "healthy",
        "service": "Fast Track AI Agent",
        "version": "1.0.0"
    }


# Run the application
if __name__ == "__main__":
    import uvicorn

    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║        Fast Track Communication & Support AI Agent       ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝

    Server starting on http://{settings.HOST}:{settings.PORT}

    API Documentation: http://{settings.HOST}:{settings.PORT}/docs
    """)

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
