"""
Knowledge base API routes
Handles knowledge base operations, vector database management, and vector visualization
"""
import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from backend.modules.vector_db import VectorStore
from backend.modules.visualizer import VectorVisualizer
from backend.modules.pdf_loader import get_available_pdfs
from backend.config.settings import KNOWLEDGE_BASE_DIR

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

vector_store = VectorStore()
visualizer = VectorVisualizer(vector_store)


@router.get("/stats")
async def get_knowledge_stats():
    """Get statistics about the knowledge base"""
    stats = vector_store.get_collection_stats()
    pdf_files = get_available_pdfs()

    return {
        "vector_db": stats,
        "pdf_files": pdf_files,
        "total_pdfs": len(pdf_files),
        "knowledge_base_path": str(KNOWLEDGE_BASE_DIR)
    }


@router.get("/structure")
async def get_knowledge_structure():
    """Get nested structure of knowledge base"""
    structure = vector_store.get_structure()

    formatted_structure = {}
    for source, pages in structure.items():
        formatted_structure[os.path.basename(source)] = {
            "pages": len(pages),
            "chunks": sum(len(chunks) for chunks in pages.values()),
            "page_details": {
                page: len(chunks)
                for page, chunks in pages.items()
            }
        }

    return {"structure": formatted_structure}


@router.get("/chunks")
async def get_all_chunks(limit: int = None):
    """Get all chunks from knowledge base"""
    chunks = vector_store.get_all_chunks()

    if limit:
        chunks = chunks[:limit]

    return {
        "total_chunks": len(chunks),
        "returned": len(chunks),
        "chunks": [
            {
                "id": chunk['id'],
                "source": chunk['metadata'].get('source'),
                "page": chunk['metadata'].get('page'),
                "content": chunk['content'][:200] + "..."  # Preview
            }
            for chunk in chunks
        ]
    }


@router.post("/refresh")
async def refresh_knowledge_base():
    """Refresh vector database from knowledge base files"""
    try:
        from backend.modules.pdf_loader import load_pdf_documents
        from backend.modules.text_splitter import split_documents

        documents = load_pdf_documents()
        if not documents:
            return {
                "status": "error",
                "message": "No PDF documents found in knowledge_base directory"
            }

        chunks = split_documents(documents)
        added = vector_store.add_documents(chunks)

        return {
            "status": "success",
            "total_documents": len(documents),
            "total_chunks": len(chunks),
            "new_chunks_added": added
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@router.get("/search")
async def search_knowledge(query: str, k: int = 5):
    """Search knowledge base"""
    results = vector_store.search(query, k=k)

    return {
        "query": query,
        "results_count": len(results),
        "results": results
    }


@router.get("/visualization", response_class=HTMLResponse)
async def get_visualization_html():
    """
    Render 3D vector space visualization directly as interactive HTML.
    """
    html_content = visualizer.generate_html_plot()
    return HTMLResponse(content=html_content)


@router.get("/visualization/data")
async def get_visualization_data():
    """
    Get 2D/3D PCA projection data and semantic edges as JSON.
    """
    return visualizer.get_visualization_data()


@router.delete("/file/{filename}")
async def delete_file_from_knowledge(filename: str):
    """
    Delete a document file from the knowledge base directory and Chroma collection.
    """
    # Delete from Chroma
    deleted_chunks = vector_store.delete_file(filename)

    # Delete from filesystem if present
    file_path = os.path.join(KNOWLEDGE_BASE_DIR, filename)
    file_removed = False
    if os.path.exists(file_path):
        os.remove(file_path)
        file_removed = True

    return {
        "status": "success",
        "filename": filename,
        "deleted_chunks": deleted_chunks,
        "file_removed_from_disk": file_removed
    }


@router.post("/reset")
async def reset_vector_database():
    """
    Reset and clear the entire Chroma collection.
    """
    success = vector_store.reset_database()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to reset vector database")
    return {"status": "success", "message": "Vector database reset successfully"}
