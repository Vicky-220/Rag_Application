"""
Configuration settings for the Local MultiAgentic RAG System
Supports OpenAI-compatible servers (llama.cpp, Ollama, Cloud, vLLM, LM Studio) and native Ollama
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load environment variables from .env file if present
load_dotenv(PROJECT_ROOT / ".env")

# Database paths
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"
CHAT_DB_PATH = PROJECT_ROOT / "chat_history.db"

# LLM Configuration (OpenAI-compatible or native Ollama)
# LLM_PROVIDER options: "openai" (works with llama.cpp, Ollama /v1, vLLM, Cloud) or "ollama"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", "not-needed"))
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5-4b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# Embedding Configuration
# EMBEDDING_PROVIDER options:
# - "openai": calls /v1/embeddings on any OpenAI-compatible server (llama-server with --embeddings, Ollama via /v1, OpenAI, etc.)
# - "ollama": calls native Ollama embeddings API
# - "local": built-in ONNX all-MiniLM-L6-v2 (100% offline, zero server requirement)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434/v1")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", "not-needed"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# RAG Configuration
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "5"))
CONTEXT_TURNS = int(os.getenv("CONTEXT_TURNS", "3"))  # Number of previous conversation turns to include as context

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_RELOAD = os.getenv("API_RELOAD", "false").lower() == "true"

# File Upload
ALLOWED_EXTENSIONS = {"pdf"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# Create directories if they don't exist
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
