"""
Core RAG Chat Pipeline
Orchestrates the multi-agent RAG workflow with context retrieval and source tracking.
"""
from typing import List, Dict, Any, Generator, Tuple
from backend.modules.vector_db import VectorStore
from backend.agents.query_parser import parse_queries
from backend.agents.rag_query_agent import refine_search_query
from backend.agents.response_agent import generate_response, generate_response_stream
from backend.config.settings import (
    SIMILARITY_THRESHOLD,
    TOP_K_CHUNKS,
    CONTEXT_TURNS
)


class RAGChat:
    """
    Orchestrates the complete RAG pipeline:
    1. Query parsing (resolve pronouns via structured output)
    2. Query refinement (optimize for search)
    3. Vector search (retrieve relevant chunks)
    4. Response generation (with context)
    5. Source attribution (track sources for UI & DB)
    """

    def __init__(self):
        """Initialize RAG Chat with vector store"""
        self.vector_store = VectorStore()
        self.conversation_history: List[Dict[str, str]] = []
        self.context_turns = CONTEXT_TURNS

    def add_to_history(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append({"role": role, "content": content})

    def get_recent_context(self, turns: int = None) -> str:
        """Get recent conversation context for agent input"""
        turns = turns or self.context_turns
        filtered = [m for m in self.conversation_history if m['role'] in ('user', 'assistant')]

        if not filtered or turns <= 0:
            return ""

        recent = filtered[-(turns * 2):]
        return "\n".join([f"{m['role']}: {m['content']}" for m in recent])

    def search_knowledge_base(self, query: str) -> List[Dict[str, Any]]:
        """Search vector database for relevant chunks"""
        return self.vector_store.search(
            query=query,
            k=TOP_K_CHUNKS,
            score_threshold=SIMILARITY_THRESHOLD
        )

    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into context string"""
        if not chunks:
            return "No relevant information found in knowledge base."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get('source', 'Unknown')
            page = chunk.get('page', 'N/A')
            score = chunk.get('score', 0.0)
            content = chunk.get('content', '')

            context_parts.append(
                f"[Source {i}]: {source} (Page {page}, Score: {score:.2f})\n{content}"
            )

        return "\n\n".join(context_parts)

    def retrieve_context(self, user_query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Execute multi-agent retrieval pipeline:
        1. Resolve pronouns & compound queries
        2. Refine query keywords
        3. Search vector DB and deduplicate chunks
        """
        recent_context = self.get_recent_context()
        parsed_queries = parse_queries(user_query, recent_context)

        # Search vector database for each parsed query
        seen_ids = set()
        deduped_chunks = []

        for query in parsed_queries:
            refined_query = refine_search_query(query, recent_context)
            chunks = self.search_knowledge_base(refined_query)
            for c in chunks:
                cid = c.get("chunk_id", c.get("content")[:50])
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    # Format chunk source for UI
                    deduped_chunks.append({
                        "id": cid,
                        "chunk_id": cid,
                        "source": c.get("source", "Unknown"),
                        "page": c.get("page", 0),
                        "score": c.get("score", 0.0),
                        "content": c.get("content", "")
                    })

        formatted_context = self.format_context(deduped_chunks)
        return formatted_context, deduped_chunks

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Process a user query through the complete RAG pipeline.
        Returns dict containing both response text and retrieved sources.
        """
        self.add_to_history("user", user_query)

        formatted_context, sources = self.retrieve_context(user_query)

        response = generate_response(
            user_query=user_query,
            rag_context=formatted_context,
            conversation_history=self.conversation_history[:-1]
        )

        self.add_to_history("assistant", response)

        return {
            "response": response,
            "sources": sources
        }

    def process_query_stream(self, user_query: str) -> Generator[Dict[str, Any], None, None]:
        """
        Process user query and yield both sources and streaming chunks.
        """
        self.add_to_history("user", user_query)

        formatted_context, sources = self.retrieve_context(user_query)

        # Emit sources first
        yield {"type": "sources", "sources": sources}

        # Stream response tokens
        full_response = ""
        for chunk in generate_response_stream(
            user_query=user_query,
            rag_context=formatted_context,
            conversation_history=self.conversation_history[:-1]
        ):
            full_response += chunk
            yield {"type": "content", "content": chunk}

        self.add_to_history("assistant", full_response)
