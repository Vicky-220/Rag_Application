"""
Vector database module for managing Chroma vector store operations
Supports OpenAI-compatible embeddings, Ollama embeddings, and local ONNX embeddings.
Includes full management capabilities: chunk deletion, file deletion, database reset, and search.
"""
import os
import shutil
import warnings
from typing import List, Dict, Any, Optional
from pathlib import Path
from colorama import init, Fore, Style

from langchain_core.documents import Document
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from backend.modules.embedding_function import get_embedding_function
from backend.config.settings import CHROMA_DB_PATH

init()
warnings.filterwarnings("ignore", category=DeprecationWarning)


class VectorStore:
    """
    Manages Chroma vector store operations for RAG system
    """

    def __init__(self, collection_name: str = "pdf_chunks"):
        """
        Initialize VectorStore
        
        Args:
            collection_name (str): Name of the Chroma collection
        """
        self.collection_name = collection_name
        self.embeddings = get_embedding_function()
        self.db = Chroma(
            persist_directory=str(CHROMA_DB_PATH),
            embedding_function=self.embeddings,
            collection_name=collection_name
        )

    def add_documents(self, chunks: List[Document]) -> int:
        """
        Add documents to vector store
        
        Args:
            chunks (List[Document]): List of document chunks to add
            
        Returns:
            int: Number of new documents added
        """
        if not chunks:
            return 0

        chunks_with_ids = self._get_chunk_ids(chunks)
        existing_items = self.db.get(include=[])
        existing_ids = set(existing_items["ids"]) if existing_items and "ids" in existing_items else set()
        
        print(f"Number of existing documents in DB: {len(existing_ids)}")
        
        new_chunks = []
        for chunk in chunks_with_ids:
            chunk_id = chunk.metadata.get("chunk_id")
            if chunk_id not in existing_ids:
                new_chunks.append(chunk)
        
        if len(new_chunks):
            print(f"👉 Adding {len(new_chunks)} new documents to DB...")
            new_chunk_ids = [chunk.metadata["chunk_id"] for chunk in new_chunks]
            self.db.add_documents(new_chunks, ids=new_chunk_ids)
            return len(new_chunks)
        else:
            print("✅ Database is up to date")
            return 0

    def search(self, query: str, k: int = 5, score_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Search for similar chunks in vector store
        
        Args:
            query (str): Search query
            k (int): Number of results to return
            score_threshold (float): Minimum normalized similarity score (0.0 to 1.0)
            
        Returns:
            List[Dict]: List of matching chunks with similarity scores
        """
        try:
            results = self.db.similarity_search_with_score(query, k=k)
        except Exception as e:
            print(f"Vector search error: {e}")
            return []

        filtered_results = []
        for doc, raw_score in results:
            # Chroma returns distance (lower distance = higher similarity)
            # Convert raw distance to normalized similarity score [0.0, 1.0]
            dist = float(raw_score)
            similarity = 1.0 / (1.0 + dist)

            if similarity >= score_threshold:
                filtered_results.append({
                    "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                    "content": doc.page_content,
                    "score": round(similarity, 4),
                    "distance": round(dist, 4),
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", 0)
                })
        
        # Sort by similarity descending
        filtered_results.sort(key=lambda x: x["score"], reverse=True)
        return filtered_results

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """
        Get all chunks from the database with their embeddings and metadata
        
        Returns:
            List[Dict]: All chunks with metadata and embeddings
        """
        results = self.db.get(include=['embeddings', 'documents', 'metadatas'])
        chunks = []
        if not results or not results.get('ids'):
            return chunks

        num_items = len(results['ids'])
        for i in range(num_items):
            doc = results['documents'][i] if results.get('documents') else ""
            metadata = results['metadatas'][i] if results.get('metadatas') else {}
            embedding = results['embeddings'][i] if results.get('embeddings') is not None and len(results['embeddings']) > i else []
            chunks.append({
                'id': results['ids'][i],
                'content': doc,
                'metadata': metadata,
                'embedding': embedding
            })
        return chunks

    def get_structure(self) -> Dict[str, Dict[Any, List[Dict[str, Any]]]]:
        """
        Return a nested structure: {source: {page: [chunks]}}
        
        Returns:
            Dict: Nested structure of knowledge base
        """
        chunks = self.get_all_chunks()
        structure: Dict[str, Dict[Any, List[Dict[str, Any]]]] = {}
        for chunk in chunks:
            source = chunk['metadata'].get('source', 'unknown')
            page = chunk['metadata'].get('page', 0)
            structure.setdefault(source, {}).setdefault(page, []).append(chunk)
        return structure

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection
        
        Returns:
            Dict: Statistics including total chunks, sources, pages
        """
        structure = self.get_structure()
        total_chunks = sum(
            sum(len(chunks) for chunks in pages.values())
            for pages in structure.values()
        )
        return {
            "total_chunks": total_chunks,
            "total_sources": len(structure),
            "sources": list(structure.keys())
        }

    def delete_chunks(self, chunk_ids: List[str]) -> int:
        """
        Delete specific chunks by their IDs
        
        Args:
            chunk_ids: List of chunk ID strings
            
        Returns:
            int: Number of chunks deleted
        """
        if not chunk_ids:
            return 0
        try:
            self.db.delete(ids=chunk_ids)
            return len(chunk_ids)
        except Exception as e:
            print(f"Error deleting chunks: {e}")
            return 0

    def delete_page(self, source: str, page: int) -> int:
        """
        Delete all chunks associated with a specific file and page
        
        Args:
            source: Source document filename/path
            page: Page number
            
        Returns:
            int: Number of chunks deleted
        """
        chunks = self.get_all_chunks()
        ids_to_delete = [
            c['id'] for c in chunks
            if (c['metadata'].get('source') == source or os.path.basename(c['metadata'].get('source', '')) == os.path.basename(source))
            and int(c['metadata'].get('page', -1)) == int(page)
        ]
        return self.delete_chunks(ids_to_delete)

    def delete_file(self, source: str) -> int:
        """
        Delete all chunks associated with a specific file
        
        Args:
            source: Source document filename or path
            
        Returns:
            int: Number of chunks deleted
        """
        chunks = self.get_all_chunks()
        ids_to_delete = [
            c['id'] for c in chunks
            if c['metadata'].get('source') == source or os.path.basename(c['metadata'].get('source', '')) == os.path.basename(source)
        ]
        return self.delete_chunks(ids_to_delete)

    def reset_database(self) -> bool:
        """
        Clear all documents and reset the collection
        """
        try:
            self.db.delete_collection()
            self.db = Chroma(
                persist_directory=str(CHROMA_DB_PATH),
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            return True
        except Exception as e:
            print(f"Error resetting database: {e}")
            return False

    @staticmethod
    def _get_chunk_ids(chunks: List[Document]) -> List[Document]:
        """
        Generate unique chunk IDs based on source and page
        
        Args:
            chunks (List[Document]): List of document chunks
            
        Returns:
            List[Document]: Chunks with added chunk_id in metadata
        """
        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", 0)
            current_page_id = f"{source}:{page}"

            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id
            chunk.metadata["chunk_id"] = chunk_id

        return chunks
