"""
Chat API routes for FastAPI application
Handles chat endpoint, streaming, message processing, and source persistence.
"""
from fastapi import APIRouter, WebSocket, HTTPException, UploadFile, File
import uuid
import json
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from backend.core.rag_chat import RAGChat
from backend.database.chat_db import ChatDatabase
from backend.modules.pdf_loader import load_pdf_documents
from backend.modules.text_splitter import split_documents
from backend.modules.vector_db import VectorStore

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Global instances
chat_instances: Dict[str, RAGChat] = {}
db = ChatDatabase()
vector_store = VectorStore()


class MessageRequest(BaseModel):
    """Chat message request"""
    message: str
    session_id: Optional[str] = None


class MessageResponse(BaseModel):
    """Chat message response"""
    response: str
    session_id: str
    sources: list = []


class SessionInfo(BaseModel):
    """Session information"""
    id: str
    title: str
    created_at: str
    updated_at: str


def get_or_create_chat(session_id: str) -> RAGChat:
    """Get existing chat instance or create new one populated with history"""
    if session_id not in chat_instances:
        chat_instances[session_id] = RAGChat()
        # Pre-populate history from SQLite DB
        existing_msgs = db.get_messages(session_id)
        for m in existing_msgs:
            chat_instances[session_id].add_to_history(m["role"], m["content"])
    return chat_instances[session_id]


@router.post("/message")
async def send_message(request: MessageRequest) -> MessageResponse:
    """
    Send a message and get a response with retrieved sources
    """
    session_id = request.session_id or str(uuid.uuid4())

    # Create session in database if new
    if not db.get_session(session_id):
        # Generate initial title from first query
        initial_title = request.message[:30] + ("..." if len(request.message) > 30 else "")
        db.create_session(session_id, title=initial_title)

    chat = get_or_create_chat(session_id)

    # Process query through complete pipeline
    result = chat.process_query(request.message)
    response_text = result["response"]
    retrieved_sources = result.get("sources", [])

    # Save user and assistant messages with source references to DB
    db.add_message(session_id, "user", request.message)
    db.add_message(session_id, "assistant", response_text, sources=retrieved_sources)

    return MessageResponse(
        response=response_text,
        session_id=session_id,
        sources=retrieved_sources
    )


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for streaming responses with sources
    """
    await websocket.accept()

    if not db.get_session(session_id):
        db.create_session(session_id)

    chat = get_or_create_chat(session_id)

    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")

            if not user_message:
                continue

            db.add_message(session_id, "user", user_message)

            full_response = ""
            retrieved_sources = []

            for event in chat.process_query_stream(user_message):
                if event.get("type") == "sources":
                    retrieved_sources = event.get("sources", [])
                    await websocket.send_text(json.dumps({
                        "type": "sources",
                        "sources": retrieved_sources
                    }))
                elif event.get("type") == "content":
                    chunk_text = event.get("content", "")
                    full_response += chunk_text
                    await websocket.send_text(json.dumps({
                        "type": "stream",
                        "content": chunk_text
                    }))

            # Save assistant message with sources to database
            db.add_message(session_id, "assistant", full_response, sources=retrieved_sources)

            # Send completion signal
            await websocket.send_text(json.dumps({
                "type": "complete",
                "content": full_response,
                "sources": retrieved_sources
            }))

    except Exception as e:
        print(f"WebSocket closed/error: {e}")
        try:
            await websocket.close(code=1000)
        except Exception:
            pass


@router.get("/sessions")
async def get_sessions():
    """Get all chat sessions"""
    sessions = db.get_all_sessions()
    return {
        "sessions": sessions,
        "count": len(sessions)
    }


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session and message history with sources"""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = db.get_messages(session_id)
    return {
        "session": session,
        "messages": messages,
        "message_count": len(messages)
    }


@router.post("/session/create")
async def create_session(title: str = "New Chat"):
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    db.create_session(session_id, title)
    return {
        "session_id": session_id,
        "title": title,
        "status": "created"
    }


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    success = db.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete session")

    if session_id in chat_instances:
        del chat_instances[session_id]

    return {"status": "deleted", "session_id": session_id}


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process PDF file into knowledge base
    """
    if file.content_type != "application/pdf" and not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        from backend.config.settings import KNOWLEDGE_BASE_DIR
        import os

        file_path = os.path.join(KNOWLEDGE_BASE_DIR, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        documents = load_pdf_documents()
        if documents:
            chunks = split_documents(documents)
            added = vector_store.add_documents(chunks)

            return {
                "status": "success",
                "filename": file.filename,
                "chunks_added": added
            }
        else:
            return {
                "status": "success",
                "filename": file.filename,
                "chunks_added": 0,
                "message": "File saved but no text extracted"
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")
