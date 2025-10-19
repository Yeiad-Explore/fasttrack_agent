"""
Hybrid Adaptive Chunking Strategy
Combines semantic chunking with sentence window context
"""

from typing import List, Dict, Any
import re
import tiktoken
from config import settings


class HybridChunker:
    """Hybrid adaptive chunking with semantic awareness and sentence windows"""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        model: str = "gpt-4"
    ):
        """
        Initialize the chunker

        Args:
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Number of overlapping tokens
            model: Model name for tokenizer
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.encoding = tiktoken.encoding_for_model(model)

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Split on sentence boundaries
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))

    def create_chunks(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Create chunks from text with sentence-level awareness

        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to chunks

        Returns:
            List of chunks with metadata
        """
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_tokens = 0

        for i, sentence in enumerate(sentences):
            sentence_tokens = self.count_tokens(sentence)

            # If single sentence exceeds chunk size, split it further
            if sentence_tokens > self.chunk_size:
                # Add current chunk if it has content
                if current_chunk:
                    chunks.append(self._create_chunk_dict(
                        " ".join(current_chunk),
                        metadata,
                        len(chunks)
                    ))
                    current_chunk = []
                    current_tokens = 0

                # Split long sentence into smaller pieces
                words = sentence.split()
                temp_chunk = []
                temp_tokens = 0

                for word in words:
                    word_tokens = self.count_tokens(word + " ")
                    if temp_tokens + word_tokens > self.chunk_size:
                        if temp_chunk:
                            chunks.append(self._create_chunk_dict(
                                " ".join(temp_chunk),
                                metadata,
                                len(chunks)
                            ))
                        temp_chunk = [word]
                        temp_tokens = word_tokens
                    else:
                        temp_chunk.append(word)
                        temp_tokens += word_tokens

                if temp_chunk:
                    chunks.append(self._create_chunk_dict(
                        " ".join(temp_chunk),
                        metadata,
                        len(chunks)
                    ))

            # If adding sentence would exceed chunk size, save current chunk
            elif current_tokens + sentence_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk_dict(
                        " ".join(current_chunk),
                        metadata,
                        len(chunks)
                    ))

                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk,
                    self.chunk_overlap
                )
                current_chunk = overlap_sentences + [sentence]
                current_tokens = sum(self.count_tokens(s) for s in current_chunk)

            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        # Add remaining chunk
        if current_chunk:
            chunks.append(self._create_chunk_dict(
                " ".join(current_chunk),
                metadata,
                len(chunks)
            ))

        return chunks

    def _get_overlap_sentences(self, sentences: List[str], overlap_tokens: int) -> List[str]:
        """
        Get sentences for overlap based on token count

        Args:
            sentences: List of sentences
            overlap_tokens: Target number of overlap tokens

        Returns:
            List of sentences for overlap
        """
        if not sentences:
            return []

        overlap = []
        token_count = 0

        for sentence in reversed(sentences):
            sentence_tokens = self.count_tokens(sentence)
            if token_count + sentence_tokens > overlap_tokens:
                break
            overlap.insert(0, sentence)
            token_count += sentence_tokens

        return overlap

    def _create_chunk_dict(
        self,
        text: str,
        metadata: Dict[str, Any],
        chunk_index: int
    ) -> Dict[str, Any]:
        """
        Create a chunk dictionary with metadata

        Args:
            text: Chunk text
            metadata: Source metadata
            chunk_index: Index of this chunk

        Returns:
            Chunk dictionary
        """
        chunk_metadata = metadata.copy() if metadata else {}
        chunk_metadata["chunk_index"] = chunk_index
        chunk_metadata["token_count"] = self.count_tokens(text)

        return {
            "text": text,
            "metadata": chunk_metadata
        }


# Create singleton instance
chunker = HybridChunker()
