"""
Document ingestion pipeline using Docling
Processes PDF and Markdown documents and prepares them for indexing
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
    """Pipeline for ingesting and indexing PDF and Markdown documents"""

    def __init__(self):
        """Initialize the ingestion pipeline"""
        self.converter = DocumentConverter()
        self.supported_extensions = {'.pdf', '.md', '.markdown'}

    def process_markdown(self, md_path: Path) -> str:
        """
        Read markdown file directly

        Args:
            md_path: Path to markdown file

        Returns:
            Markdown content
        """
        try:
            print(f"Processing Markdown: {md_path.name}")
            with open(md_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return text
        except Exception as e:
            print(f"Error processing Markdown {md_path.name}: {e}")
            raise

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

    def detect_and_process_file(self, file_path: Path) -> tuple[str, str]:
        """
        Auto-detect file type and process accordingly

        Args:
            file_path: Path to file

        Returns:
            Tuple of (text_content, file_type)
        """
        file_ext = file_path.suffix.lower()

        if file_ext == '.pdf':
            text = self.process_pdf(file_path)
            return text, 'pdf'
        elif file_ext in {'.md', '.markdown'}:
            text = self.process_markdown(file_path)
            return text, 'markdown'
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

    def ingest_document(self, file_path: Path, check_duplicate: bool = True) -> int:
        """
        Ingest a single document (PDF or Markdown)

        Args:
            file_path: Path to file
            check_duplicate: Whether to check for existing file in vector store

        Returns:
            Number of chunks created
        """
        # Check for duplicates if enabled
        if check_duplicate and vector_store.is_file_indexed(str(file_path)):
            print(f"⚠️  File already indexed: {file_path.name} (skipping)")
            return 0

        # Auto-detect and extract text
        text, file_type = self.detect_and_process_file(file_path)

        # Create metadata
        metadata = {
            "source": file_path.name,
            "file_path": str(file_path),
            "file_type": file_type
        }

        # Chunk the document
        chunks = chunker.create_chunks(text, metadata)
        print(f"Created {len(chunks)} chunks from {file_path.name}")

        # Generate embeddings
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = embedding_service.embed_batch(chunk_texts)

        # Prepare metadatas
        metadatas = [chunk["metadata"] for chunk in chunks]

        # Add to vector store
        vector_store.add_documents(chunk_texts, embeddings, metadatas)

        return len(chunks)

    def ingest_directory(self, directory: Path = None, check_duplicate: bool = True) -> Dict[str, Any]:
        """
        Ingest all supported files (PDF, Markdown) from a directory

        Args:
            directory: Directory containing files (defaults to KB_DIR)
            check_duplicate: Whether to skip already indexed files

        Returns:
            Statistics about ingestion
        """
        if directory is None:
            directory = settings.KB_DIR

        # Find all supported files
        all_files = []
        for ext in self.supported_extensions:
            all_files.extend(directory.glob(f"*{ext}"))

        if not all_files:
            print(f"No supported files found in {directory}")
            print(f"Supported extensions: {', '.join(self.supported_extensions)}")
            return {
                "total_files": 0,
                "total_chunks": 0,
                "files_processed": [],
                "skipped_duplicates": 0
            }

        print(f"Found {len(all_files)} files to process")
        print(f"File types: {', '.join([f.suffix for f in all_files])}")

        # Process each file
        total_chunks = 0
        files_processed = []
        skipped_count = 0

        for file_path in tqdm(all_files, desc="Ingesting files"):
            try:
                num_chunks = self.ingest_document(file_path, check_duplicate=check_duplicate)

                if num_chunks == 0 and check_duplicate:
                    # File was skipped due to duplicate
                    skipped_count += 1
                    files_processed.append({
                        "filename": file_path.name,
                        "chunks": 0,
                        "status": "skipped",
                        "reason": "already_indexed"
                    })
                else:
                    total_chunks += num_chunks
                    files_processed.append({
                        "filename": file_path.name,
                        "chunks": num_chunks,
                        "file_type": file_path.suffix,
                        "status": "success"
                    })
            except Exception as e:
                print(f"Failed to process {file_path.name}: {e}")
                files_processed.append({
                    "filename": file_path.name,
                    "chunks": 0,
                    "status": "failed",
                    "error": str(e)
                })

        # Save the vector store
        vector_store.save()

        return {
            "total_files": len(all_files),
            "total_chunks": total_chunks,
            "files_processed": files_processed,
            "skipped_duplicates": skipped_count
        }

    def clear_index(self):
        """Clear the vector store index"""
        vector_store.clear()
        vector_store.save()
        print("Index cleared successfully")


# Create singleton instance
ingestion_pipeline = DocumentIngestionPipeline()
