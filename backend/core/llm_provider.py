"""
Universal LLM and Embedding Provider
Supports OpenAI-compatible APIs (llama.cpp / llama-server, Ollama via /v1, vLLM, LM Studio, Cloud OpenAI/Groq/OpenRouter)
as well as native Ollama and local ONNX embeddings.
"""
import json
import re
import os
from typing import List, Dict, Any, Optional, Type, Generator
from pydantic import BaseModel
from langchain_core.embeddings import Embeddings

from backend.config.settings import (
    LLM_PROVIDER,
    LLM_BASE_URL,
    LLM_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    EMBEDDING_PROVIDER,
    EMBEDDING_BASE_URL,
    EMBEDDING_API_KEY,
    EMBEDDING_MODEL,
)


def _clean_reasoning_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from output if present."""
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in cleaned and "</think>" not in cleaned:
        cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def _extract_json_block(text: str) -> str:
    """Extract JSON content from code fences, reasoning text, or raw string."""
    if not text:
        return ""

    # 1. Try markdown code block ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 2. Try explicit {"queries": [...]} pattern
    match = re.search(r'(\{\s*"queries"\s*:\s*\[.*?\]\s*\})', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 3. Try any valid JSON object from all curly brace blocks (latest first)
    matches = re.findall(r"(\{.*?\})", text, re.DOTALL)
    for candidate in reversed(matches):
        try:
            json.loads(candidate)
            return candidate.strip()
        except Exception:
            continue

    # Fallback to cleaned text
    return _clean_reasoning_tags(text).strip()


class UniversalLLMClient:
    """
    Unified client for LLM generation with OpenAI compatibility & Ollama fallback.
    """

    def __init__(self):
        self.provider = LLM_PROVIDER.lower()
        self.base_url = LLM_BASE_URL
        self.api_key = LLM_API_KEY or "not-needed"
        self.model = LLM_MODEL
        self.temperature = LLM_TEMPERATURE
        self.max_tokens = LLM_MAX_TOKENS

        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        else:
            self.client = None

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Send a chat completion request and return the text response.
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        if self.provider == "openai":
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temp,
            }
            if tokens:
                kwargs["max_tokens"] = tokens
            if response_format:
                kwargs["response_format"] = response_format

            try:
                response = self.client.chat.completions.create(**kwargs)
            except Exception as e:
                # Some servers (e.g. older llama-server) may reject response_format
                if response_format and "response_format" in str(e).lower():
                    kwargs.pop("response_format", None)
                    response = self.client.chat.completions.create(**kwargs)
                else:
                    raise e

            message = response.choices[0].message
            content = message.content or ""
            reasoning = getattr(message, "reasoning_content", "") or ""

            # If content is empty but model reasoned out the answer in reasoning_content
            if not content.strip() and reasoning:
                content = reasoning

            return _clean_reasoning_tags(content)

        else:
            # Native Ollama
            import ollama
            kwargs = {
                "model": self.model,
                "messages": messages,
                "options": {"temperature": temp}
            }
            if response_format:
                kwargs["format"] = response_format
            response = ollama.chat(**kwargs)
            content = response["message"]["content"]
            return _clean_reasoning_tags(content)

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """
        Stream chat response chunks.
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        if self.provider == "openai":
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temp,
                "stream": True,
            }
            if tokens:
                kwargs["max_tokens"] = tokens

            stream = self.client.chat.completions.create(**kwargs)
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if getattr(delta, "content", None):
                        yield delta.content
        else:
            import ollama
            stream = ollama.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={"temperature": temp}
            )
            for chunk in stream:
                if "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]

    def parse_structured(
        self,
        messages: List[Dict[str, str]],
        pydantic_model: Type[BaseModel],
    ) -> BaseModel:
        """
        Execute chat completion guaranteeing Pydantic structured output.
        Compatible with OpenAI Structured Outputs, JSON mode, and Ollama.
        """
        schema_json = json.dumps(pydantic_model.model_json_schema(), indent=2)

        guided_messages = list(messages)
        guidance = (
            f"\nYou MUST respond strictly in valid JSON conforming to this JSON Schema:\n"
            f"```json\n{schema_json}\n```\n"
            "Output ONLY the JSON object, with NO extra conversational text or thoughts."
        )
        if guided_messages and guided_messages[0]["role"] == "system":
            guided_messages[0] = {
                "role": "system",
                "content": guided_messages[0]["content"] + "\n" + guidance
            }
        else:
            guided_messages.insert(0, {"role": "system", "content": guidance})

        if self.provider == "openai":
            # Attempt 1: Try OpenAI beta parse if available
            try:
                if hasattr(self.client.beta.chat.completions, "parse"):
                    completion = self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=guided_messages,
                        response_format=pydantic_model,
                        temperature=0.1,
                    )
                    if completion.choices[0].message.parsed:
                        return completion.choices[0].message.parsed
            except Exception:
                pass

            # Attempt 2: Use json_object response format
            try:
                raw_text = self.chat_completion(
                    messages=guided_messages,
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                clean_json = _extract_json_block(raw_text)
                return pydantic_model.model_validate_json(clean_json)
            except Exception:
                pass

            # Attempt 3: Standard completion with regex extraction
            raw_text = self.chat_completion(
                messages=guided_messages,
                temperature=0.1
            )
            clean_json = _extract_json_block(raw_text)
            return pydantic_model.model_validate_json(clean_json)

        else:
            # Native Ollama
            import ollama
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=guided_messages,
                    format=pydantic_model.model_json_schema()
                )
                clean_json = _extract_json_block(response["message"]["content"])
                return pydantic_model.model_validate_json(clean_json)
            except Exception:
                response = ollama.chat(
                    model=self.model,
                    messages=guided_messages,
                    format="json"
                )
                clean_json = _extract_json_block(response["message"]["content"])
                return pydantic_model.model_validate_json(clean_json)


class UniversalEmbeddings(Embeddings):
    """
    LangChain-compatible Embeddings class supporting:
    1. 'openai': OpenAI-compatible /v1/embeddings (llama-server, Ollama, OpenAI, vLLM)
    2. 'ollama': Native Ollama embedding API
    3. 'local': Embedded Chroma ONNX embeddings (all-MiniLM-L6-v2) for offline use
    """

    def __init__(self):
        self.provider = EMBEDDING_PROVIDER.lower()
        self.model = EMBEDDING_MODEL
        self.base_url = EMBEDDING_BASE_URL
        self.api_key = EMBEDDING_API_KEY or "not-needed"

        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        elif self.provider == "local":
            from chromadb.utils import embedding_functions
            self._local_fn = embedding_functions.DefaultEmbeddingFunction()
        else:
            self.client = None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document strings."""
        if not texts:
            return []

        if self.provider == "openai":
            try:
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.model
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                print(f"[UniversalEmbeddings] Warning: OpenAI-compatible embedding failed: {e}. Falling back to local ONNX.")
                from chromadb.utils import embedding_functions
                fn = embedding_functions.DefaultEmbeddingFunction()
                return [list(map(float, vec)) for vec in fn(texts)]

        elif self.provider == "ollama":
            import ollama
            embeddings = []
            for text in texts:
                res = ollama.embeddings(model=self.model, prompt=text)
                embeddings.append(res["embedding"])
            return embeddings

        else:
            # Local ONNX default embeddings
            results = self._local_fn(texts)
            return [list(map(float, vec)) for vec in results]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        docs = self.embed_documents([text])
        return docs[0] if docs else []


# Global singleton instances
llm_client = UniversalLLMClient()
embeddings_client = UniversalEmbeddings()
