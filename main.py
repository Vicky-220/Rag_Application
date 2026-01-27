from typing import List, Dict, Any
from modules.pdf_loader import load_pdf_documents
from modules.text_splitter import split_documents
from modules.populate_database import add_to_vector_db
from modules.embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma
from agents.no_of_queries import parse_queries
from agents.rag_query_agent import rag_query_agent
from agents.response_agent import generate_response
from colorama import init, Fore, Style
import numpy as np

# Initialize colorama
init()

class Chat:
    def __init__(self):
        self.history: List[Dict[str, str]] = []
        self.db = Chroma(
            persist_directory="./chroma_db/",
            embedding_function=get_embedding_function(),
            collection_name="pdf_chunks"
        )
        self.SIMILARITY_THRESHOLD = 0.6
        self.TOP_K = 5
        self.show_chunk_content = False  # Flag to control chunk content display
        # Number of last user<->assistant conversations to pass as context to agents
        # Change this number to adjust how many past conversations are included
        # for `no_of_queries.parse_queries` and `rag_query_agent`.
        self.CONTEXT_TURNS = 3

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def get_context(self) -> str:
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.history])

    def get_recent_conversation_context(self, turns: int) -> str:
        """Return the last `turns` user<->assistant messages as a single string.

        This collects only messages with role 'user' or 'assistant', takes the
        last `turns * 2` entries (user+assistant per turn) and returns them in
        chronological order formatted as 'role: content' lines.
        """
        # Filter history to user and assistant roles only
        filtered = [m for m in self.history if m.get('role') in ('user', 'assistant')]
        if turns <= 0 or not filtered:
            return ""

        # We want up to turns exchanges => up to turns * 2 messages
        num_msgs = turns * 2
        recent = filtered[-num_msgs:]
        return "\n".join([f"{m['role']}: {m['content']}" for m in recent])

    def search_chunks(self, query: str) -> List[Dict[str, Any]]:
        results = self.db.similarity_search_with_score(query, k=self.TOP_K)
        filtered_results = []
        
        for doc, score in results:
            if score >= self.SIMILARITY_THRESHOLD:
                filtered_results.append({
                    "chunk_id": doc.metadata["chunk_id"],
                    "content": doc.page_content,
                    "score": score
                })
        
        return filtered_results

    def process_query(self, user_input: str) -> str:
        # Add user input to history
        self.add_message("user", user_input)
        chat_context = self.get_context()

        # Build the recent conversation context (last N user<->assistant turns)
        recent_context = self.get_recent_conversation_context(self.CONTEXT_TURNS)

        # Step 1: Parse queries using no_of_queries agent
        # Pass only the recent user<->assistant conversation context to the agent
        queries = parse_queries(user_input, recent_context)
        all_relevant_chunks = []

        # Step 2: Process each query through RAG pipeline
        for query in queries:
            # Get refined search query
            # Pass only the recent conversation context to the RAG query agent
            context_dict = {"conversation_history": recent_context}
            refined_query = rag_query_agent(query, context_dict)

            # Search for relevant chunks
            chunks = self.search_chunks(refined_query)
            all_relevant_chunks.extend(chunks)

        # Step 3: Prepare context for response generation
        if all_relevant_chunks:
            # Sort chunks by score in descending order
            all_relevant_chunks.sort(key=lambda x: x['score'], reverse=True)

            rag_context = "\n\n".join([
                f"[Chunk {chunk['chunk_id']} (score: {chunk['score']:.2f})]\n{chunk['content']}"
                for chunk in all_relevant_chunks
            ])
        else:
            rag_context = "No relevant context found."

        # Step 4: Generate response using response agent
        # Build a `messages` list for the response agent from recent conversation
        # so the model has the necessary context. We include the same recent
        # user/assistant turns passed to the other agents.
        messages = []
        if recent_context:
            # recent_context is formatted as lines 'role: content' - convert
            # back into a list of message dicts for the agent
            for line in recent_context.splitlines():
                if ':' in line:
                    role, content = line.split(':', 1)
                    messages.append({"role": role.strip(), "content": content.strip()})

        response = generate_response(user_input, messages=messages, rag_context=rag_context)
        self.add_message("assistant", response)

        # Step 5: Display used chunks
        chunks_info = f"\n\n{Fore.CYAN}Sources used:{Style.RESET_ALL}\n"
        for chunk in all_relevant_chunks:
            chunks_info += f"{Fore.GREEN}- Chunk {chunk['chunk_id']} (score: {chunk['score']:.2f}){Style.RESET_ALL}"
            if self.show_chunk_content:
                chunks_info += f"\n  {chunk['content'][:100]}..."
            chunks_info += "\n"

        return response + chunks_info

def setup_database():
    documents = load_pdf_documents()
    if documents:
        chunks = split_documents(documents)
        add_to_vector_db(chunks)

def main():
    setup_database()
    chat = Chat()
    
    print(f"{Fore.CYAN}RAG Chat initialized. Type 'quit' to exit.{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Tip: Type 'show_chunks' to toggle chunk content display.{Style.RESET_ALL}")
    
    while True:
        user_input = input(f"\n{Fore.GREEN}You:{Style.RESET_ALL} ").strip()
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'show_chunks':
            chat.show_chunk_content = not chat.show_chunk_content
            print(f"{Fore.YELLOW}Chunk content display: {'enabled' if chat.show_chunk_content else 'disabled'}{Style.RESET_ALL}")
            continue
        
        response = chat.process_query(user_input)
        print(f"\n{Fore.BLUE}Assistant:{Style.RESET_ALL}", response)

if __name__ == "__main__":
    main()
    