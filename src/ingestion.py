"""
Document ingestion pipeline using Docling
Processes PDF documents and prepares them for indexing
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from docling.document_converter import DocumentConverter
from tqdm import tqdm

from config import settings
from chunking import chunker
from embeddings import embedding_service
from vector_store import vector_store


class DocumentIngestionPipeline:
    """Pipeline for ingesting and indexing PDF documents"""

    def __init__(self):
        """Initialize the ingestion pipeline"""
        self.converter = DocumentConverter()

    def process_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from PDF using Docling

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text content
        """
        try:
            print(f"Processing PDF: {pdf_path.name}")
            result = self.converter.convert(str(pdf_path))

            # Extract markdown format (preserves structure)
            text = result.document.export_to_markdown()

            return text
        except Exception as e:
            print(f"Error processing PDF {pdf_path.name}: {e}")
            raise

    def ingest_document(self, pdf_path: Path) -> int:
        """
        Ingest a single PDF document

        Args:
            pdf_path: Path to PDF file

        Returns:
            Number of chunks created
        """
        # Extract text
        text = self.process_pdf(pdf_path)

        # Create metadata
        metadata = {
            "source": pdf_path.name,
            "file_path": str(pdf_path),
            "file_type": "pdf"
        }

        # Chunk the document
        chunks = chunker.create_chunks(text, metadata)
        print(f"Created {len(chunks)} chunks from {pdf_path.name}")

        # Generate embeddings
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = embedding_service.embed_batch(chunk_texts)

        # Prepare metadatas
        metadatas = [chunk["metadata"] for chunk in chunks]

        # Add to vector store
        vector_store.add_documents(chunk_texts, embeddings, metadatas)

        return len(chunks)

    def ingest_directory(self, directory: Path = None) -> Dict[str, Any]:
        """
        Ingest all PDF files from a directory

        Args:
            directory: Directory containing PDFs (defaults to KB_DIR)

        Returns:
            Statistics about ingestion
        """
        if directory is None:
            directory = settings.KB_DIR

        # Find all PDF files
        pdf_files = list(directory.glob("*.pdf"))

        if not pdf_files:
            print(f"No PDF files found in {directory}")
            return {
                "total_files": 0,
                "total_chunks": 0,
                "files_processed": []
            }

        print(f"Found {len(pdf_files)} PDF files to process")

        # Process each PDF
        total_chunks = 0
        files_processed = []

        for pdf_path in tqdm(pdf_files, desc="Ingesting PDFs"):
            try:
                num_chunks = self.ingest_document(pdf_path)
                total_chunks += num_chunks
                files_processed.append({
                    "filename": pdf_path.name,
                    "chunks": num_chunks,
                    "status": "success"
                })
            except Exception as e:
                print(f"Failed to process {pdf_path.name}: {e}")
                files_processed.append({
                    "filename": pdf_path.name,
                    "chunks": 0,
                    "status": "failed",
                    "error": str(e)
                })

        # Save the vector store
        vector_store.save()

        return {
            "total_files": len(pdf_files),
            "total_chunks": total_chunks,
            "files_processed": files_processed
        }

    def clear_index(self):
        """Clear the vector store index"""
        vector_store.clear()
        vector_store.save()
        print("Index cleared successfully")


# Create singleton instance
ingestion_pipeline = DocumentIngestionPipeline()
