# Project Setup & Universal Compatibility Guide

## ✅ What Has Been Completed & Modernized

This project is fully ready to run with universal LLM and Embedding support across **llama.cpp / llama-server**, **Ollama**, **Cloud models (OpenAI, Groq, OpenRouter)**, and **offline local embeddings**.

### 1. Universal LLM & Embedding Architecture (`backend/core/llm_provider.py`)
- ✅ **OpenAI-Compatible Standard**: Connect to any OpenAI-compatible `/v1` endpoint (llama-server, Ollama via `/v1`, vLLM, LM Studio, or Cloud APIs).
- ✅ **Structured Outputs**: Fully compatible structured JSON schema parsing with multi-tier fallback (OpenAI Beta parse, JSON schema object mode, and robust markdown fence extraction).
- ✅ **Reasoning Model Support**: Safely cleans and parses thinking tags (`<think>...</think>`) and `reasoning_content` from reasoning models such as Qwen 3.5 and DeepSeek R1.
- ✅ **Streaming Support**: Real-time token streaming with separate UI source citation emission.
- ✅ **Flexible Embeddings**:
  - `local`: Embedded ONNX `all-MiniLM-L6-v2` (Zero external server or API key required).
  - `openai`: Calls any `/v1/embeddings` endpoint (llama-server `--embeddings`, Ollama `/v1`, OpenAI, etc.).
  - `ollama`: Native Ollama embedding API.

### 2. Database Management & Vector Visualization (Fully Restored & Enhanced)
- ✅ **Interactive CLI (`manage_database.py`)**:
  - Interactive exploration (Files -> Pages -> Chunks)
  - View collection statistics
  - Vector similarity search
  - Delete file from database and disk
  - Delete specific page chunks
  - Reset / clear collection
  - Generate & launch interactive 3D Vector Space Visualization
- ✅ **Vector Space 3D Visualizer (`backend/modules/visualizer.py`)**:
  - Computes 3D and 2D PCA projections of high-dimensional embeddings
  - Generates semantic cosine similarity network connections between chunks
  - Visualizes file clusters with interactive Plotly 3D scatter plots
  - Served live at `http://localhost:8000/api/knowledge/visualization` or via CLI `python manage_database.py --visualize`
- ✅ **SQLite Chat Database (`backend/database/chat_db.py`)**:
  - Foreign key cascade delete enabled
  - `chunk_references` properly populated for source attribution

### 3. Frontend & Build Rectifications
- ✅ Fixed 10 TypeScript compilation errors in frontend (`@/types` aliases, debounce types, unused variables).
- ✅ Built production assets (`npm run build`) in `frontend/dist`.
- ✅ Single-server mode: `main.py` directly serves the React SPA, backend API, WebSocket, and 3D visualizer on `http://localhost:8000`!

---

## 🚀 Quick Start

### 1. Activate Python 3.11 Virtual Environment
```bash
# If .venv is already created:
source .venv/bin/activate

# Or to recreate with Python 3.11:
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements_backend.txt
```

### 2. Configuration (`.env`)
Copy `.env.example` to `.env` or customize environment variables:

```bash
cp .env.example .env
```

#### Preset A: llama.cpp (llama-server)
If running `llama-server` on port 8080:
```env
LLM_PROVIDER=openai
LLM_BASE_URL=http://localhost:8080/v1
LLM_API_KEY=not-needed
LLM_MODEL=qwen3.5-4b
EMBEDDING_PROVIDER=local
```

#### Preset B: Ollama
If running Ollama on port 11434:
```env
LLM_PROVIDER=openai
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=not-needed
LLM_MODEL=qwen2.5:3b
EMBEDDING_PROVIDER=local
# Or for Ollama embeddings (after running 'ollama pull bge-m3:latest'):
# EMBEDDING_PROVIDER=ollama
# EMBEDDING_MODEL=bge-m3:latest
```

#### Preset C: Cloud Models (OpenAI / OpenRouter / Groq)
```env
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-openai-api-key
LLM_MODEL=gpt-4o-mini
EMBEDDING_PROVIDER=openai
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=sk-your-openai-api-key
EMBEDDING_MODEL=text-embedding-3-small
```

---

## 🖥️ Running the Application

### Option 1: Full-Stack Mode (Recommended)
FastAPI will host both the backend API and the React frontend on port 8000:
```bash
python main.py
```
- **Web App**: [http://localhost:8000](http://localhost:8000)
- **API Docs (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Interactive 3D Vector Space**: [http://localhost:8000/api/knowledge/visualization](http://localhost:8000/api/knowledge/visualization)

### Option 2: Frontend Development Server
If developing the React frontend with hot reloading:
```bash
cd frontend
npm run dev
# Accessible at http://localhost:3000 (proxies /api to http://localhost:8000)
```

---

## 🗄️ Database Management & Vector Visualization CLI

Run the database management utility at any time:
```bash
python manage_database.py
```

### CLI Shortcut Flags:
- **Launch 3D Visualization in browser**:
  ```bash
  python manage_database.py --visualize
  ```
- **View collection statistics**:
  ```bash
  python manage_database.py --stats
  ```
- **Ingest / refresh knowledge base from PDFs in `knowledge_base/`**:
  ```bash
  python manage_database.py --refresh
  ```
