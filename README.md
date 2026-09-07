<div align="center">

# 🧠 Local Multi-Agentic RAG System

**An enterprise-grade, fully local Retrieval-Augmented Generation system powered by a multi-agent verification pipeline, 3D PCA vector space visualization, and universal OpenAI-compatible inference.**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange?style=for-the-badge)](https://www.trychroma.com)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-OpenAI_Compatible-green?style=for-the-badge)](https://github.com/ggerganov/llama.cpp)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue?style=for-the-badge)](LICENSE)

<br/>

<!-- ================================================================ -->
<!--                      VIDEO DEMO PLACEHOLDER                      -->
<!-- ================================================================ -->

### 📺 Video Demonstration

<!-- 
REPLACE THIS SECTION WITH YOUR VIDEO DEMO
Options:
1. Embed video via HTML (for web rendering):
   <video src="https://your-domain.com/demo.mp4" width="100%" controls></video>
2. Embed an animated GIF or YouTube thumbnail link:
   [![Watch the Demo](https://img.youtube.com/vi/YOUR_VIDEO_ID/maxresdefault.jpg)](https://youtu.be/YOUR_VIDEO_ID)
-->

> [!TIP]
> ### 🎥 [Watch Video Demo](#) *(Click to view video walk-through)*
> ```
> ┌────────────────────────────────────────────────────────────────────────┐
> │                                                                        │
> │                    ▶️  ADD YOUR VIDEO DEMO HERE                        │
> │                                                                        │
> │   (Insert your YouTube / Loom / MP4 link or animated GIF preview)      │
> │                                                                        │
> └────────────────────────────────────────────────────────────────────────┘
> ```

---

</div>

## 🌟 Key Highlights

- **🤖 Multi-Agent Pipeline**: Specialized agents orchestrate query parsing, pronoun resolution, compound query splitting, vector refinement, and grounded answer synthesis.
- **🌐 Universal Model Compatibility**: Connect out-of-the-box to **llama.cpp (`llama-server`)**, **Ollama**, or **Cloud providers (OpenAI, Groq, OpenRouter)** using the standard OpenAI `/v1` protocol.
- **🧩 Structured Output Guarantees**: Multi-tier structured JSON extraction conforming to strict Pydantic schemas with automatic filtering of thinking/reasoning tokens (`<think>...</think>`).
- **📊 3D PCA Vector Space Visualization**: Interactive WebGL 3D scatter plots of chunk embeddings with nearest-neighbor semantic similarity network edges powered by Plotly.
- **⚡ Dual Embedding Strategies**: Run completely offline with embedded ONNX `all-MiniLM-L6-v2` (zero server required) or connect to high-dimensional embedding servers (`bge-m3`, `text-embedding-3-small`).
- **🗄️ Full Database Management Suite**: Interactive terminal CLI (`manage_database.py`) and REST endpoints for collection exploration, chunk inspection, document deletion, and index rebuilding.
- **🎨 Modern Cyberpunk UI**: Production-ready React + TypeScript frontend with live token streaming, multi-session management, and real-time source citation cards.

---

## 🏗️ Architecture & Pipeline Flow

```
User Message ───────────────────────────────────────────────────────────────────┐
                                                                                │
┌────────────────────────────────────────────────────────────────────────────┐  │
│                       MULTI-AGENT VERIFICATION PIPELINE                    │  │
│                                                                            │  │
│   ┌───────────────────────────┐                                            │  │
│   │   Agent 1: Query Parser   │ ◄── [Prior Conversation Context]          │  │
│   │  • Resolves pronouns      │                                            │  │
│   │  • Splits compound queries│ ──► Structured Output (Pydantic QueryList) │  │
│   └─────────────┬─────────────┘                                            │  │
│                 ▼ [Standalone Queries]                                     │  │
│   ┌───────────────────────────┐                                            │  │
│   │ Agent 2: RAG Query Agent  │                                            │  │
│   │  • Semantic expansion     │ ──► Dense Vector Search (ChromaDB)         │  │
│   └─────────────┬─────────────┘                                            │  │
│                 ▼ [Retrieved Chunks]                                       │  │
│   ┌───────────────────────────┐                                            │  │
│   │ Deduplication & Ranking   │ ──► Normalized Similarity: 1 / (1 + dist)  │  │
│   └─────────────┬─────────────┘                                            │  │
│                 ▼ [Context + System Prompt]                                │  │
│   ┌───────────────────────────┐                                            │  │
│   │  Agent 3: Response Agent  │ ──► Real-time Token Streaming & Citations  │  │
│   └───────────────────────────┘                                            │  │
└────────────────────────────────────────────────────────────────────────────┘  │
                                                                                ▼
UI / WebSocket ◄────────────────────────────────────────────────────────────────┘
```

---

## 📊 Interactive 3D Vector Space Visualization

The system features an integrated WebGL visualizer that projects your knowledge base embeddings into 3D and 2D space:

* **Principal Component Analysis (PCA)** projects high-dimensional embedding vectors (e.g. 384D or 1024D) into $(x, y, z)$ coordinates.
* **Semantic Network Edges**: Calculates pairwise cosine similarities and renders connecting lines between top-$k$ nearest neighbors to visualize semantic document bridges.
* **Interactive Tooltips**: Hover over any point to inspect the source file, page number, chunk ID, and a preview of the text content.

> **Accessing Visualization**:
> - Web browser: [http://localhost:8000/api/knowledge/visualization](http://localhost:8000/api/knowledge/visualization)
> - CLI shortcut: `python manage_database.py --visualize`

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.11+
- Node.js 18+ and npm
- Any local or remote LLM (e.g., `llama-server`, `ollama`, or an OpenAI API key)

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/vivekananda-2201/Local_MultiAgentic_RAG_System.git
cd Local_MultiAgentic_RAG_System

# Create and activate a Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install backend dependencies
pip install -r requirements_backend.txt
```

### 3. Configure Inference Provider
Copy the example environment file:
```bash
cp .env.example .env
```

Choose your preferred configuration:

#### Option A: llama.cpp / llama-server (Recommended for Local Speed)
```env
LLM_PROVIDER=openai
LLM_BASE_URL=http://localhost:8080/v1
LLM_API_KEY=not-needed
LLM_MODEL=qwen3.5-4b
EMBEDDING_PROVIDER=local
```

#### Option B: Ollama
```env
LLM_PROVIDER=openai
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=not-needed
LLM_MODEL=qwen2.5:3b
EMBEDDING_PROVIDER=local
```

#### Option C: Cloud Models (OpenAI / Groq / OpenRouter)
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

### 4. Build Frontend & Launch Full-Stack Server
```bash
# Build frontend assets (single distribution)
cd frontend
npm install
npm run build
cd ..

# Launch application
python main.py
```

* **Web UI**: [http://localhost:8000](http://localhost:8000)
* **Interactive API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **3D Vector Visualizer**: [http://localhost:8000/api/knowledge/visualization](http://localhost:8000/api/knowledge/visualization)

---

## 🗄️ Database Management CLI

A dedicated interactive terminal tool is included to manage the vector database:

```bash
python manage_database.py
```

```text
============================================================
      LOCAL MULTI-AGENTIC RAG - DATABASE MANAGER      
============================================================

Database Management Menu:
  1. View Collection Statistics
  2. Interactive Exploration (Files -> Pages -> Chunks)
  3. Search Vector Database
  4. Populate / Refresh Database from knowledge_base/
  5. Delete a File from Database
  6. Delete a Specific Page from Database
  7. Reset / Clear Vector Database
  8. Generate & Open 3D Vector Space Visualization
  0. Exit
```

**CLI Flags:**
```bash
python manage_database.py --stats        # View chunk counts & sources
python manage_database.py --visualize    # Generate & open 3D vector space in browser
python manage_database.py --refresh      # Ingest PDFs from knowledge_base/
```

---

## 📂 Project Structure

```
Local_MultiAgentic_RAG_System/
├── backend/
│   ├── agents/                  # Specialized Multi-Agent logic
│   │   ├── models.py            # Global model definitions
│   │   ├── query_parser.py      # Pronoun resolution & structured output
│   │   ├── rag_query_agent.py   # Vector keyword optimization
│   │   └── response_agent.py    # Grounded answer synthesis & streaming
│   ├── api/
│   │   └── routes/
│   │       ├── chat.py          # Chat REST & WebSocket endpoints
│   │       └── knowledge.py     # Knowledge management & visualization
│   ├── config/
│   │   └── settings.py          # Unified environment & model configuration
│   ├── core/
│   │   ├── llm_provider.py      # Universal OpenAI/Ollama/Local client
│   │   └── rag_chat.py          # Multi-agent pipeline orchestrator
│   ├── database/
│   │   └── chat_db.py           # SQLite persistence with foreign key cascades
│   └── modules/
│       ├── embedding_function.py# Embeddings factory
│       ├── pdf_loader.py        # PDF document directory loader
│       ├── text_splitter.py     # Recursive character text splitting
│       ├── vector_db.py         # ChromaDB operations & distance scoring
│       └── visualizer.py        # 3D PCA projection & Plotly engine
├── docs/                        # In-depth technical documentation
│   ├── ARCHITECTURE.md          # Complete system architecture
│   ├── MULTI_AGENT_ENGINEERING.md # Multi-agent & structured output details
│   ├── VECTOR_DATABASE_AND_PCA_VISUALIZATION.md # PCA math & vector store
│   ├── UNIVERSAL_COMPATIBILITY_GUIDE.md # llama.cpp, Ollama, & Cloud recipes
│   └── STORAGE_AND_DATABASE.md  # Relational schema & foreign key cascades
├── frontend/                    # React 18 + TypeScript + Vite SPA
│   ├── src/
│   │   ├── components/          # Chat, Sidebar, and SourcesPanel
│   │   ├── hooks/               # Custom state hooks
│   │   └── services/            # API client layer
│   └── vite.config.ts
├── knowledge_base/              # Target directory for PDF knowledge documents
├── main.py                      # FastAPI entry point & SPA server
├── manage_database.py           # Database management CLI tool
├── requirements_backend.txt     # Python backend dependencies
└── .env.example                 # Example configuration presets
```

---

## 📖 In-Depth Engineering Documentation

For deep dives into the mathematics, architecture, and design decisions, consult the **[`docs/`](docs/)** directory:

- [System Architecture Deep Dive](docs/ARCHITECTURE.md)
- [Multi-Agent Engineering & Structured Outputs](docs/MULTI_AGENT_ENGINEERING.md)
- [Vector Store & 3D PCA Visualization](docs/VECTOR_DATABASE_AND_PCA_VISUALIZATION.md)
- [Universal Compatibility & Server Setup Recipes](docs/UNIVERSAL_COMPATIBILITY_GUIDE.md)
- [Relational Storage & Schema Design](docs/STORAGE_AND_DATABASE.md)

---

## 📄 License

This project is licensed under the [Apache 2.0 License](LICENSE).