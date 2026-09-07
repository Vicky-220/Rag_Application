# Multi-Agent Architecture & Structured Output Engineering

## 1. Multi-Agent Design Pattern

Traditional naive RAG architectures send raw user questions directly to vector stores. This produces significant retrieval degradation when users ask:
1. **Pronoun-dependent questions**: *"Tell me more about them"*
2. **Multi-part questions**: *"What are the qualifications for president and how does impeachment work?"*
3. **Conversational follow-ups**: *"What are the penalties if that happens?"*

Our system resolves this using three specialized agent roles:

```
                    ┌───────────────────────────┐
                    │      User Query Input     │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
           ┌─────────────────────────────────────────────┐
           │          Agent 1: Query Parser              │
           │  • Pronoun & entity resolution              │
           │  • Compound question splitting              │
           │  • Structured output via Pydantic QueryList │
           └──────────────────────┬──────────────────────┘
                                  │
                                  ▼ [Explicit Query 1, Query 2, ...]
           ┌─────────────────────────────────────────────┐
           │          Agent 2: RAG Query Agent           │
           │  • Keyword enrichment                       │
           │  • Search syntax optimization               │
           └──────────────────────┬──────────────────────┘
                                  │
                                  ▼ [Dense Vector Search in Chroma]
           ┌─────────────────────────────────────────────┐
           │          Deduplication & Ranking            │
           │  • Unique Chunk ID deduplication            │
           │  • Normalized similarity score ranking      │
           └──────────────────────┬──────────────────────┘
                                  │
                                  ▼ [Context + System Prompt]
           ┌─────────────────────────────────────────────┐
           │       Agent 3: Response Generation          │
           │  • Grounded answer synthesis                │
           │  • Source attribution & streaming tokens    │
           └─────────────────────────────────────────────┘
```

---

## 2. Structured Output Engineering

A core challenge when interfacing with local models (such as `llama.cpp` or local `Ollama` endpoints) is ensuring deterministic structured outputs without model failures.

### The Multi-Tier Structured Output Strategy

The `UniversalLLMClient.parse_structured()` method implements a 4-tier resilient execution strategy:

```
                  Structured Output Request (Pydantic Model)
                                     │
                                     ▼
                ┌────────────────────────────────────────┐
                │ Tier 1: OpenAI Beta Structured Outputs │
                │ (client.beta.chat.completions.parse)   │
                └───────────────────┬────────────────────┘
                                    │ Success? ──► Return parsed model
                                    │ Fail
                                    ▼
                ┌────────────────────────────────────────┐
                │ Tier 2: JSON Object Mode               │
                │ response_format={"type": "json_object"}│
                │ + Injected JSON Schema in prompt       │
                └───────────────────┬────────────────────┘
                                    │ Success? ──► Return validated model
                                    │ Fail
                                    ▼
                ┌────────────────────────────────────────┐
                │ Tier 3: Resilient Extraction & Regex   │
                │ Extract JSON from ```json blocks, raw  │
                │ curly braces, or reasoning_content     │
                └───────────────────┬────────────────────┘
                                    │ Success? ──► Return validated model
                                    │ Fail
                                    ▼
                ┌────────────────────────────────────────┐
                │ Tier 4: Native Ollama Schema (if set)  │
                │ ollama.chat(..., format=model_schema)  │
                └────────────────────────────────────────┘
```

### Handling Reasoning / Thinking Models (e.g. Qwen 3.5, DeepSeek R1)

Modern open-weights models frequently include internal reasoning phases that output thinking blocks either:
1. Wrapped in `<think>...</think>` tags inside `message.content`.
2. Emitted separately in a dedicated `message.reasoning_content` field (in modern `llama-server`).

#### The Problem:
- When a thinking model generates reasoning, it can consume tokens before generating the actual response. If `max_tokens` is too low, the model runs out of budget before the JSON payload is produced.
- In some local server setups, `message.content` might remain empty while the actual JSON object is drafted at the conclusion of `message.reasoning_content`.

#### The Solution (`backend/core/llm_provider.py`):
```python
def _clean_reasoning_tags(text: str) -> str:
    """Removes complete or unclosed <think> blocks."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in cleaned and "</think>" not in cleaned:
        cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()

def _extract_json_block(text: str) -> str:
    """Extracts valid JSON from markdown fences, raw braces, or reasoning text."""
    # 1. Code fence match: ```json { ... } ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 2. Targeted schema match: {"queries": [...]}
    match = re.search(r'(\{\s*"queries"\s*:\s*\[.*?\]\s*\})', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 3. All valid JSON objects (reverse search latest first)
    matches = re.findall(r"(\{.*?\})", text, re.DOTALL)
    for candidate in reversed(matches):
        try:
            json.loads(candidate)
            return candidate.strip()
        except Exception:
            continue
    return text.strip()
```

---

## 3. Agent Prompts & Context Guidance

### Query Parser Agent Prompt
```text
You are a query resolution agent for a RAG system.
Your job is to analyze the user's latest message IN THE CONTEXT of the conversation history,
and produce a list of fully self-contained, explicit search queries that can be sent to a
vector database — WITHOUT any pronouns, references like "them", "it", "those", or vague terms.

Rules:
1. ALWAYS resolve all pronouns and references using the conversation history.
2. If the current message is a follow-up, enrich the query with topic keywords.
3. Split compound questions into separate focused queries.
4. Each query must be fully standalone.
5. Output ONLY a valid JSON object conforming to the schema: {"queries": ["..."]}.
```

### RAG Query Agent Prompt
```text
You are an AI agent that generates optimal search queries for RAG systems.
Given a user query, generate a single, well-crafted search query that will retrieve
the most relevant documents from a vector database. The query should be specific,
clear, and include relevant keywords. Output ONLY the search query.
```

### Response Generation Agent Prompt
```text
You are a helpful AI Assistant powered by a Retrieval Augmented Generation (RAG) system.
Your role is to provide accurate, helpful responses based strictly on the provided context.
If the context does not contain sufficient information to answer the question, clearly state
'I don't have enough information to answer this question.' Always cite the sources when possible.
```
