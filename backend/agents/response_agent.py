"""
Response Agent - Generates final responses based on RAG context
Uses retrieved documents to provide accurate, context-aware answers
Supports OpenAI-compatible APIs and Ollama (standard and streaming).
"""
import warnings
from backend.core.llm_provider import llm_client

warnings.filterwarnings("ignore", category=DeprecationWarning)

SYSTEM_PROMPT = (
    "You are a helpful AI Assistant powered by a Retrieval Augmented Generation (RAG) system. "
    "Your role is to provide accurate, helpful responses based strictly on the provided context. "
    "If the context does not contain sufficient information to answer the question, "
    "clearly state 'I don't have enough information to answer this question.' "
    "Always cite the sources or relevant context when possible."
)


def generate_response(user_query: str, rag_context: str, conversation_history: list = None) -> str:
    """
    Generate a response using retrieved RAG context
    
    Args:
        user_query (str): The user's question
        rag_context (str): Retrieved context from vector database
        conversation_history (list): Previous conversation messages
        
    Returns:
        str: Generated response
    """
    messages = []

    # Add conversation history if provided
    if conversation_history:
        messages.extend(conversation_history)

    # Add context and query
    messages.append({
        "role": "system",
        "content": SYSTEM_PROMPT
    })

    messages.append({
        "role": "user",
        "content": (
            f"Context from knowledge base:\n{rag_context}\n\n"
            f"User question: {user_query}"
        )
    })

    try:
        response = llm_client.chat_completion(messages=messages)
        return response
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I encountered an error while generating a response. Please try again."


def generate_response_stream(user_query: str, rag_context: str, conversation_history: list = None):
    """
    Generate a streaming response using retrieved RAG context
    
    Args:
        user_query (str): The user's question
        rag_context (str): Retrieved context from vector database
        conversation_history (list): Previous conversation messages
        
    Yields:
        str: Response chunks as they are generated
    """
    messages = []

    # Add conversation history if provided
    if conversation_history:
        messages.extend(conversation_history)

    # Add context and query
    messages.append({
        "role": "system",
        "content": SYSTEM_PROMPT
    })

    messages.append({
        "role": "user",
        "content": (
            f"Context from knowledge base:\n{rag_context}\n\n"
            f"User question: {user_query}"
        )
    })

    try:
        for chunk in llm_client.chat_stream(messages=messages):
            yield chunk
    except Exception as e:
        print(f"Error generating streaming response: {e}")
        yield "I encountered an error while generating a response. Please try again."
