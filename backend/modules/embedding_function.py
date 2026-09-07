"""
Embedding function module for generating embeddings using Universal Provider
Supports OpenAI-compatible endpoints, Ollama, and local ONNX embeddings.
"""
from backend.core.llm_provider import embeddings_client


def get_embedding_function():
    """
    Get universal embeddings function instance.
    
    Returns:
        Embeddings: LangChain-compatible embeddings instance
    """
    return embeddings_client
