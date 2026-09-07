# System Architecture & Engineering Deep Dive

## 1. Architectural Philosophy

The **Local MultiAgentic RAG System** is designed from first principles to decouple **agent orchestration**, **inference backends**, **embedding providers**, and **user interface**. It avoids hard-coding any specific local engine (such as Ollama or llama.cpp) or remote cloud provider, allowing the entire pipeline to operate identically regardless of whether the model runs locally or in the cloud.

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                           PRESENTATION LAYER                                   │
│  React 18 + TypeScript + Vite SPA (Embedded /assets or standalone on :3000)   │
│  ├─ Sidebar (Sessions & Navigation)                                           │
│  ├─ Chat Interface (Markdown streaming & autoscroll)                          │
│  ├─ Sources Panel (Chunk preview, relevance scores, and metadata)              │
│  └─ 3D Vector Space Visualizer (Plotly WebGL Canvas)                           │
└──────────────────────────────────────┬─────────────────────────────────────────┘
                                       │ HTTP REST & WebSocket (ws://)
┌──────────────────────────────────────▼─────────────────────────────────────────┐
│                             API GATEWAY (FastAPI)                              │
│  ├─ POST /api/chat/message          ├─ GET  /api/knowledge/stats              │
│  ├─ WS   /api/chat/ws/{session_id}  ├─ GET  /api/knowledge/visualization     │
│  ├─ GET  /api/chat/sessions         ├─ GET  /api/knowledge/visualization/data│
│  └─ POST /api/chat/upload           └─ POST /api/knowledge/refresh            │
└──────────────────────────────────────┬─────────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼─────────────────────────────────────────┐
│                     MULTI-AGENT RAG PIPELINE (Core)                           │
│                                                                               │
│   User Query ──► [Agent 1: Query Parser]                                      │
│                  ├─ Resolves contextual pronouns using prior conversation     │
│                  ├─ Splits compound queries into discrete sub-queries         │
│                  └─ Guarantees structured output via QueryList JSON schema    │
│                         │                                                     │
│                         ▼ [List of explicit search queries]                   │
│                  [Agent 2: RAG Query Agent]                                   │
│                  ├─ Semantic keyword expansion & vector search optimization   │
│                  └─ Dispatches queries to VectorStore                         │
│                         │                                                     │
│                         ▼ [Retrieved & Deduplicated Chunks]                   │
│                  [Context Formatter]                                          │
│                  └─ Ranks by normalized similarity score (1 / (1 + distance)) │
│                         │                                                     │
│                         ▼ [RAG Context + User Query + Chat History]           │
│                  [Agent 3: Response Generation Agent]                         │
│                  ├─ Context-bound answer synthesis with strict grounding      │
│                  └─ Streams tokens & emits real-time source citations         │
└───────────────────────┬───────────────────────────────┬───────────────────────┘
                        │                               │
┌───────────────────────▼─────────────┐ ┌───────────────▼───────────────────────┐
│     STORAGE & PERSISTENCE LAYER     │ │     UNIVERSAL INFERENCE LAYER         │
│                                     │ │                                       │
│  SQLite (chat_history.db):          │ │  UniversalLLMClient:                  │
│  ├─ sessions                        │ │  ├─ llama-server (llama.cpp on :8080) │
│  ├─ messages                        │ │  ├─ Ollama (via /v1 on :11434)        │
│  ├─ chunk_references (FK cascade)   │ │  └─ Cloud OpenAI / Groq / OpenRouter  │
│  └─ conversation_metadata           │ │                                       │
│                                     │ │  UniversalEmbeddings:                 │
│  ChromaDB (chroma_db/):             │ │  ├─ local: ONNX all-MiniLM-L6-v2      │
│  ├─ HNSW Index                      │ │  ├─ openai: /v1/embeddings            │
│  └─ Document & Vector Storage       │ │  └─ ollama: native embeddings API     │
└─────────────────────────────────────┘ └───────────────────────────────────────┘
```

---

## 2. End-to-End Pipeline Execution Flow

### Step 1: User Request Ingestion
When a user submits a query via REST (`POST /api/chat/message`) or WebSocket (`/api/chat/ws/{session_id}`):
1. The session is resolved or automatically created in SQLite.
2. The `RAGChat` instance loads previous conversation context from the database to retain multi-turn awareness.

### Step 2: Query Resolution & Disambiguation (Agent 1)
Human conversation is often filled with ambiguous follow-ups, e.g.:
> *"Tell me more about them"* or *"What are its main exceptions?"*

The **Query Parser Agent** inspects the conversation turns:
- Resolves all pronouns (`them`, `it`, `those`) into explicit entities based on previous context.
- Breaks multi-intent questions into standalone atomic search queries.
- Enforces strict JSON output conforming to Pydantic `QueryList` schema.

### Step 3: Semantic Keyword Optimization (Agent 2)
Each atomic query is passed through the **RAG Query Agent**:
- The agent expands the query with domain-specific terminology optimal for dense vector retrieval.
- Strips filler words and conversational phrasing.

### Step 4: Dense Vector Retrieval & Deduplication
The refined queries are queried against ChromaDB:
1. Embeddings are generated using `UniversalEmbeddings`.
2. HNSW cosine/Euclidean distance search executes with $k = 5$ per query.
3. Raw distance $d$ is converted into normalized similarity:
   $$\text{similarity} = \frac{1}{1 + d}$$
4. Chunks retrieved across multiple sub-queries are **deduplicated** by `chunk_id`, retaining the maximum similarity score.

### Step 5: Context Assembly & Response Synthesis (Agent 3)
1. Retrieved chunks are formatted into numbered reference blocks:
   `[Source 1]: document.pdf (Page 4, Score: 0.82) ...`
2. The **Response Agent** generates the final answer grounded strictly on the retrieved context.
3. In WebSocket streaming mode:
   - First packet emits `{"type": "sources", "sources": [...]}` so the UI immediately highlights the cited sources.
   - Subsequent packets stream content tokens (`{"type": "stream", "content": "..."}`).
4. The assistant message and its source citations are recorded in SQLite `messages` and `chunk_references`.

---

## 3. High-Availability & Resiliency Mechanisms

* **Zero External Dependency Fallback**: When external embedding servers are unreachable, the pipeline falls back to embedded local ONNX embeddings (`all-MiniLM-L6-v2`) without crashing.
* **Reasoning Model Defense**: Thinking models (e.g. Qwen 3.5, DeepSeek R1) that generate `<think>...</think>` tokens or store output in `reasoning_content` are automatically cleansed and parsed.
* **Database Isolation**: The SQLite database utilizes explicit foreign keys with `CASCADE DELETE`, preventing orphaned message rows upon session removal.
