"""
RAG Query Agent - Refines user queries for vector database search
Generates optimal search queries based on context
Supports OpenAI-compatible APIs and Ollama.
"""
import warnings
from colorama import init, Fore, Style
from backend.core.llm_provider import llm_client

init()
warnings.filterwarnings("ignore", category=DeprecationWarning)


def refine_search_query(query: str, conversation_context: str = "") -> str:
    """
    Refine user query for optimal vector database search
    
    Args:
        query (str): Original user query
        conversation_context (str): Previous conversation history
        
    Returns:
        str: Refined search query optimized for vector search
    """
    system_prompt = (
        "You are an AI agent that generates optimal search queries for RAG (Retrieval Augmented Generation) systems. "
        "Given a user query, generate a single, well-crafted search query that will retrieve the most relevant documents "
        "from a vector database. The query should be specific, clear, and include relevant keywords. "
        "Output ONLY the search query, with no explanations or punctuation around it."
    )

    messages = [
        {"role": "system", "content": system_prompt},
    ]

    # Include conversation context if provided
    if conversation_context:
        messages.append({
            "role": "user",
            "content": f"Conversation context:\n{conversation_context}"
        })

    messages.append({
        "role": "user",
        "content": f"Query to refine: {query}"
    })

    try:
        refined_query = llm_client.chat_completion(
            messages=messages,
            temperature=0.3
        ).strip().strip('"').strip("'")
        
        print(
            f"{Fore.LIGHTMAGENTA_EX}[RAG Query Agent] Refined Search Query:{Style.RESET_ALL} {refined_query}\n"
        )
        return refined_query or query
    except Exception as e:
        print(f"{Fore.RED}[RAG Query Agent] Error refining query: {e}{Style.RESET_ALL}")
        return query
