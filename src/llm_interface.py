"""
LLM Interface for Azure OpenAI
Handles response generation with context from retrieval
"""

from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
from config import settings


class LLMInterface:
    """Interface for Azure OpenAI LLM"""

    def __init__(self):
        """Initialize Azure OpenAI client"""
        self.client = AzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
        )
        self.deployment = settings.CHAT_MODEL_DEPLOYMENT

        # System prompt for Fast Track courier service
        self.system_prompt = """You are an AI customer service assistant for Fast Track Communication & Support, a professional courier and delivery service company.

Your role is to help customers with:
- Shipping rates and delivery times
- Package tracking and status updates
- Service areas and delivery restrictions
- Customs documentation and international shipping
- Pickup scheduling and arrangements
- Claims, complaints, and issue resolution
- General courier service inquiries

Guidelines:
1. Be professional, friendly, and helpful
2. Provide accurate information based on the knowledge base
3. If you don't have specific information, acknowledge it and suggest contacting customer service
4. Always cite your sources when providing specific information
5. For tracking numbers, delivery times, or customer-specific queries, direct them to the appropriate channel
6. Maintain a customer-first approach in all responses

Use the provided context to answer questions accurately. If the context doesn't contain relevant information, say so clearly."""

    def generate_response(
        self,
        query: str,
        context_documents: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False
    ) -> str:
        """
        Generate response using Azure OpenAI

        Args:
            query: User query
            context_documents: Retrieved context documents
            conversation_history: Previous conversation messages
            stream: Whether to stream the response

        Returns:
            Generated response text
        """
        # Prepare context from retrieved documents
        context = self._format_context(context_documents)

        # Build messages
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history[-6:])  # Last 3 exchanges

        # Add current query with context
        user_message = self._format_user_message(query, context)
        messages.append({"role": "user", "content": user_message})

        # Generate response
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                max_tokens=settings.MAX_TOKENS,
                temperature=settings.TEMPERATURE,
                stream=stream
            )

            if stream:
                return response  # Return generator for streaming
            else:
                return response.choices[0].message.content

        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again or contact our customer service team directly."

    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context string

        Args:
            documents: List of retrieved documents with metadata

        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant context found."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "Unknown")

            context_parts.append(
                f"[Source {i}: {source}]\n{text}\n"
            )

        return "\n".join(context_parts)

    def _format_user_message(self, query: str, context: str) -> str:
        """
        Format user message with context

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Formatted message
        """
        return f"""Context information from our knowledge base:
{context}

Customer Question: {query}

Please provide a helpful, accurate response based on the context above. If the context doesn't fully answer the question, acknowledge what you can answer and what additional information the customer should seek from our customer service team."""

    def generate_streaming_response(
        self,
        query: str,
        context_documents: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ):
        """
        Generate streaming response

        Args:
            query: User query
            context_documents: Retrieved context documents
            conversation_history: Previous conversation messages

        Yields:
            Response chunks
        """
        response = self.generate_response(
            query,
            context_documents,
            conversation_history,
            stream=True
        )

        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content


# Create singleton instance
llm_interface = LLMInterface()
