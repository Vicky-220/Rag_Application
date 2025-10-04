import ollama
import colorama
from colorama import init

def rag_query_agent(query: str, context):
    """RAG query agent to answer user queries based on provided context."""
    system_prompt = """You are an AI agent that generate rag search query to search in rag.
    use the context provided to you to generate the best possible query to search in rag."""

    response = ollama.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"}
        ],
        model="qwen2.5:3b-instruct-q8_0"
    )

    answer = response['message']['content']
    print(colorama.Fore.LIGHTMAGENTA_EX + "\n[RAG Query Agent] Generated RAG Search Query:" + colorama.Style.RESET_ALL + f" {answer}\n")
    return answer