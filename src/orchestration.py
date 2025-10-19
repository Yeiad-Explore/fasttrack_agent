"""
Query Orchestration
Coordinates the entire RAG pipeline from query to response
"""

from typing import Dict, Any, List, Optional
from retrieval import retriever
from llm_interface import llm_interface


class QueryOrchestrator:
    """Orchestrates the end-to-end query processing pipeline"""

    def __init__(self):
        """Initialize the orchestrator"""
        self.conversation_histories = {}  # Store conversation histories by session

    def process_query(
        self,
        query: str,
        session_id: str = "default",
        stream: bool = False,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline

        Args:
            query: User query
            session_id: Session identifier for conversation tracking
            stream: Whether to stream the response
            top_k: Number of context documents to retrieve

        Returns:
            Response dictionary containing answer, sources, and metadata
        """
        try:
            # Step 1: Retrieve relevant context
            print(f"Processing query: {query}")
            retrieved_docs = retriever.retrieve(query, top_k=top_k)

            if not retrieved_docs:
                return {
                    "answer": "I apologize, but I couldn't find relevant information in our knowledge base to answer your question. Please contact our customer service team for assistance.",
                    "sources": [],
                    "context_used": 0
                }

            # Extract documents and scores
            documents = [doc for doc, score in retrieved_docs]
            scores = [score for doc, score in retrieved_docs]

            print(f"Retrieved {len(documents)} relevant documents")

            # Step 2: Get conversation history
            conversation_history = self.conversation_histories.get(session_id, [])

            # Step 3: Generate response
            if stream:
                return {
                    "stream": llm_interface.generate_streaming_response(
                        query,
                        documents,
                        conversation_history
                    ),
                    "sources": self._format_sources(documents, scores),
                    "context_used": len(documents)
                }
            else:
                answer = llm_interface.generate_response(
                    query,
                    documents,
                    conversation_history
                )

                # Step 4: Update conversation history
                self._update_conversation_history(session_id, query, answer)

                # Step 5: Format response
                return {
                    "answer": answer,
                    "sources": self._format_sources(documents, scores),
                    "context_used": len(documents)
                }

        except Exception as e:
            print(f"Error processing query: {e}")
            return {
                "answer": "I apologize, but an error occurred while processing your request. Please try again.",
                "sources": [],
                "context_used": 0,
                "error": str(e)
            }

    def _format_sources(
        self,
        documents: List[Dict[str, Any]],
        scores: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Format source documents for response

        Args:
            documents: Retrieved documents
            scores: Relevance scores

        Returns:
            List of formatted source dictionaries
        """
        sources = []
        seen_sources = set()

        for doc, score in zip(documents, scores):
            metadata = doc.get("metadata", {})
            source_name = metadata.get("source", "Unknown")

            # Avoid duplicate sources
            if source_name not in seen_sources:
                sources.append({
                    "source": source_name,
                    "relevance_score": round(score, 3),
                    "chunk_index": metadata.get("chunk_index", 0)
                })
                seen_sources.add(source_name)

        return sources

    def _update_conversation_history(
        self,
        session_id: str,
        query: str,
        answer: str
    ):
        """
        Update conversation history for a session

        Args:
            session_id: Session identifier
            query: User query
            answer: Assistant answer
        """
        if session_id not in self.conversation_histories:
            self.conversation_histories[session_id] = []

        self.conversation_histories[session_id].extend([
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer}
        ])

        # Keep only last 10 messages (5 exchanges)
        if len(self.conversation_histories[session_id]) > 10:
            self.conversation_histories[session_id] = \
                self.conversation_histories[session_id][-10:]

    def clear_conversation_history(self, session_id: str = "default"):
        """
        Clear conversation history for a session

        Args:
            session_id: Session identifier
        """
        if session_id in self.conversation_histories:
            del self.conversation_histories[session_id]
            print(f"Cleared conversation history for session: {session_id}")

    def get_conversation_history(self, session_id: str = "default") -> List[Dict[str, str]]:
        """
        Get conversation history for a session

        Args:
            session_id: Session identifier

        Returns:
            List of conversation messages
        """
        return self.conversation_histories.get(session_id, [])


# Create singleton instance
orchestrator = QueryOrchestrator()
