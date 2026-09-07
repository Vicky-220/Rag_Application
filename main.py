"""
Main FastAPI application for Local MultiAgentic RAG System
Entry point for the backend server and frontend SPA host
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from pathlib import Path
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_core._api.deprecation import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

# Import routes
from backend.api.routes import chat, knowledge
from backend.config.settings import API_HOST, API_PORT

# Initialize FastAPI app
app = FastAPI(
    title="Local MultiAgentic RAG System",
    description="A production-grade RAG system with multi-agent architecture and OpenAI-compatible support",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(chat.router)
app.include_router(knowledge.router)


@app.get("/api/info")
async def api_info():
    """API info endpoint"""
    return {
        "name": "Local MultiAgentic RAG System",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "chat": "/api/chat/message",
            "websocket": "/api/chat/ws/{session_id}",
            "knowledge_stats": "/api/knowledge/stats",
            "visualization": "/api/knowledge/visualization",
            "visualization_data": "/api/knowledge/visualization/data",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Local MultiAgentic RAG System"
    }


# Serve frontend SPA if built
frontend_path = Path(__file__).parent / "frontend" / "dist"
if frontend_path.exists():
    # Mount assets folder
    assets_path = frontend_path / "assets"
    if assets_path.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")

    @app.get("/")
    async def serve_index():
        """Serve frontend index.html"""
        return FileResponse(frontend_path / "index.html")

    @app.get("/{full_path:path}")
    async def serve_spa_fallback(full_path: str):
        """SPA fallback for frontend client-side routing"""
        file_path = frontend_path / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(frontend_path / "index.html")
else:
    @app.get("/")
    async def root():
        """Fallback root endpoint when frontend is not built"""
        return {
            "name": "Local MultiAgentic RAG System",
            "version": "1.0.0",
            "status": "running",
            "notice": "Frontend not built. Run 'cd frontend && npm install && npm run build'",
            "endpoints": {
                "chat": "/api/chat/message",
                "websocket": "/api/chat/ws/{session_id}",
                "knowledge": "/api/knowledge/stats",
                "visualization": "/api/knowledge/visualization",
                "docs": "/docs"
            }
        }


if __name__ == "__main__":
    import uvicorn
    print(f"Starting server on http://{API_HOST}:{API_PORT}")
    print(f"API Docs available at http://{API_HOST}:{API_PORT}/docs")
    print(f"Vector Visualization available at http://{API_HOST}:{API_PORT}/api/knowledge/visualization")
    uvicorn.run(app, host=API_HOST, port=API_PORT, reload=False)