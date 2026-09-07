# Universal LLM & Embedding Compatibility Guide

## 1. The Universal Abstraction Layer

Modern AI ecosystems feature a multitude of inference runtimes: `llama.cpp`, `Ollama`, `vLLM`, `LocalAI`, `LM Studio`, and hosted Cloud APIs (`OpenAI`, `Groq`, `OpenRouter`). Rather than binding the application to proprietary SDKs, our system interfaces through the standard **OpenAI HTTP Protocol** via [`backend/core/llm_provider.py`](file:///home/vicky/Projects/Local_MultiAgentic_RAG_System/backend/core/llm_provider.py).

```
                      UniversalLLMClient / UniversalEmbeddings
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
      Chat Completions API:                       Embeddings API:
     POST /v1/chat/completions                   POST /v1/embeddings
                   │                                           │
       ┌───────────┼───────────┐                   ┌───────────┼───────────┐
       ▼           ▼           ▼                   ▼           ▼           ▼
   llama-server  Ollama      Cloud             llama-server  Ollama      Cloud
    (:8080)     (:11434)  (api.openai.com)      (--embeddings)(:11434) (api.openai.com)
                                                               │
                                                               ▼ (Fallback)
                                                           Embedded ONNX
                                                         (all-MiniLM-L6-v2)
```

---

## 2. Configuration Matrix

All settings are controlled via environment variables or a `.env` file in the project root:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `LLM_PROVIDER` | `openai` | Protocol driver: `openai` or `ollama` |
| `LLM_BASE_URL` | `http://localhost:8080/v1` | Base URL of the OpenAI-compatible server |
| `LLM_API_KEY` | `not-needed` | API key (or `OPENAI_API_KEY`) |
| `LLM_MODEL` | `qwen3.5-4b` | Model name or path |
| `LLM_TEMPERATURE`| `0.7` | Generation temperature |
| `LLM_MAX_TOKENS` | `2048` | Maximum token completion window |
| `EMBEDDING_PROVIDER`| `local` | `local` (ONNX), `openai` (/v1), or `ollama` |
| `EMBEDDING_BASE_URL`| `http://localhost:11434/v1`| Base URL for embeddings API |
| `EMBEDDING_API_KEY` | `not-needed` | API key for embeddings |
| `EMBEDDING_MODEL`| `all-MiniLM-L6-v2` | Model identifier for embeddings |
| `SIMILARITY_THRESHOLD`| `0.3` | Minimum normalized cosine/Euclidean similarity |
| `TOP_K_CHUNKS` | `5` | Top chunks to retrieve per search query |

---

## 3. Server Setup Recipes

### Recipe A: Running with llama.cpp (`llama-server`)

`llama-server` provides high-performance C++ inference with GPU acceleration (CUDA, Metal, Vulkan, ROCm).

1. **Start `llama-server`**:
   ```bash
   llama-server \
     -m /path/to/your/model.gguf \
     -ngl 99 \
     -c 16000 \
     -t 8 \
     --port 8080
   ```
2. **Configure `.env`**:
   ```env
   LLM_PROVIDER=openai
   LLM_BASE_URL=http://localhost:8080/v1
   LLM_API_KEY=not-needed
   LLM_MODEL=your-model-name
   EMBEDDING_PROVIDER=local
   ```

*Note on Embeddings*: By default, `llama-server` runs either in generation mode or embedding mode (`--embeddings`). If running a single server for generation, keep `EMBEDDING_PROVIDER=local` to utilize the embedded ONNX all-MiniLM model with zero extra processes.

---

### Recipe B: Running with Ollama

Ollama exposes both a native REST API and a standard OpenAI-compatible `/v1` endpoint.

1. **Start Ollama & Pull Models**:
   ```bash
   ollama serve
   ollama pull qwen2.5:3b
   ollama pull bge-m3:latest
   ```
2. **Configure `.env` (OpenAI-Compatible Mode)**:
   ```env
   LLM_PROVIDER=openai
   LLM_BASE_URL=http://localhost:11434/v1
   LLM_API_KEY=not-needed
   LLM_MODEL=qwen2.5:3b

   EMBEDDING_PROVIDER=openai
   EMBEDDING_BASE_URL=http://localhost:11434/v1
   EMBEDDING_API_KEY=not-needed
   EMBEDDING_MODEL=bge-m3:latest
   ```
3. **Alternatively (Native Ollama Driver)**:
   ```env
   LLM_PROVIDER=ollama
   LLM_MODEL=qwen2.5:3b
   EMBEDDING_PROVIDER=ollama
   EMBEDDING_MODEL=bge-m3:latest
   ```

---

### Recipe C: Running with Cloud Models (OpenAI, Groq, OpenRouter)

#### OpenAI:
```env
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-proj-...
LLM_MODEL=gpt-4o-mini

EMBEDDING_PROVIDER=openai
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=sk-proj-...
EMBEDDING_MODEL=text-embedding-3-small
```

#### Groq (Ultra-fast cloud inference):
```env
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=gsk_...
LLM_MODEL=llama-3.3-70b-versatile
EMBEDDING_PROVIDER=local
```

#### OpenRouter:
```env
LLM_PROVIDER=openai
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-...
LLM_MODEL=meta-llama/llama-3.2-3b-instruct
EMBEDDING_PROVIDER=local
```
