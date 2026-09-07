"""
SQLite database module for storing chat sessions, messages, and metadata
Provides ORM models and database operations with foreign key integrity.
"""
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional
from backend.config.settings import CHAT_DB_PATH


class ChatDatabase:
    """
    Manages SQLite database for chat sessions, messages, and chunk references
    """

    def __init__(self, db_path: str = None):
        """
        Initialize ChatDatabase
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path or str(CHAT_DB_PATH)
        self.init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with foreign keys enabled"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_database(self):
        """Initialize database schema"""
        conn = self._get_connection()
        c = conn.cursor()

        # Sessions table
        c.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT DEFAULT 'New Chat',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        ''')

        # Messages table
        c.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT,
                tokens_used INTEGER DEFAULT 0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        ''')

        # Sources/Chunks reference table
        c.execute('''
            CREATE TABLE IF NOT EXISTS chunk_references (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id INTEGER NOT NULL,
                chunk_id TEXT NOT NULL,
                source_file TEXT NOT NULL,
                page_number INTEGER,
                relevance_score REAL,
                FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
            )
        ''')

        # Conversation metadata table
        c.execute('''
            CREATE TABLE IF NOT EXISTS conversation_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                UNIQUE(session_id, key)
            )
        ''')

        conn.commit()
        conn.close()

    def create_session(self, session_id: str, title: str = "New Chat", metadata: Dict = None) -> bool:
        """
        Create a new chat session
        """
        try:
            conn = self._get_connection()
            c = conn.cursor()
            metadata_json = json.dumps(metadata) if metadata else None
            c.execute(
                "INSERT INTO sessions (id, title, metadata) VALUES (?, ?, ?)",
                (session_id, title, metadata_json)
            )
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: List[Dict] = None,
        tokens_used: int = 0
    ) -> int:
        """
        Add a message to a session and record chunk references
        """
        conn = self._get_connection()
        c = conn.cursor()

        # Ensure session exists
        c.execute("INSERT OR IGNORE INTO sessions (id) VALUES (?)", (session_id,))

        # Insert message
        sources_json = json.dumps(sources) if sources else None
        c.execute(
            "INSERT INTO messages (session_id, role, content, sources, tokens_used) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, sources_json, tokens_used)
        )
        message_id = c.lastrowid

        # Insert chunk references if provided
        if sources and message_id:
            for s in sources:
                chunk_id = s.get("chunk_id") or s.get("id") or "unknown"
                source_file = s.get("source") or "unknown"
                page = s.get("page") or 0
                score = s.get("score") or 0.0
                c.execute(
                    "INSERT INTO chunk_references (message_id, chunk_id, source_file, page_number, relevance_score) VALUES (?, ?, ?, ?, ?)",
                    (message_id, chunk_id, source_file, page, score)
                )

        # Update session updated_at
        c.execute(
            "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (session_id,)
        )

        conn.commit()
        conn.close()

        return message_id

    def get_messages(self, session_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve messages from a session
        """
        conn = self._get_connection()
        c = conn.cursor()

        if limit:
            c.execute(
                "SELECT id, role, content, sources, tokens_used, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                (session_id, limit)
            )
        else:
            c.execute(
                "SELECT id, role, content, sources, tokens_used, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp",
                (session_id,)
            )

        rows = c.fetchall()
        conn.close()

        messages = []
        for msg_id, role, content, sources_json, tokens_used, timestamp in rows:
            sources = json.loads(sources_json) if sources_json else []
            messages.append({
                "id": msg_id,
                "role": role,
                "content": content,
                "sources": sources,
                "tokens_used": tokens_used,
                "timestamp": timestamp
            })

        return messages

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session information
        """
        conn = self._get_connection()
        c = conn.cursor()
        c.execute(
            "SELECT id, title, created_at, updated_at, metadata FROM sessions WHERE id = ?",
            (session_id,)
        )
        row = c.fetchone()
        conn.close()

        if row:
            session_id, title, created_at, updated_at, metadata_json = row
            metadata = json.loads(metadata_json) if metadata_json else {}
            return {
                "id": session_id,
                "title": title,
                "created_at": created_at,
                "updated_at": updated_at,
                "metadata": metadata
            }
        return None

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """
        Get all chat sessions ordered by most recent
        """
        conn = self._get_connection()
        c = conn.cursor()
        c.execute(
            "SELECT id, title, created_at, updated_at, metadata FROM sessions ORDER BY updated_at DESC"
        )
        rows = c.fetchall()
        conn.close()

        sessions = []
        for session_id, title, created_at, updated_at, metadata_json in rows:
            metadata = json.loads(metadata_json) if metadata_json else {}
            sessions.append({
                "id": session_id,
                "title": title,
                "created_at": created_at,
                "updated_at": updated_at,
                "metadata": metadata
            })

        return sessions

    def update_session_title(self, session_id: str, title: str) -> bool:
        """
        Update session title
        """
        try:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute(
                "UPDATE sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (title, session_id)
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error updating session title: {e}")
            return False

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and cascade delete all its messages and chunk references
        """
        try:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error deleting session: {e}")
            return False

    def set_metadata(self, session_id: str, key: str, value: Any) -> bool:
        """
        Set session metadata
        """
        try:
            conn = self._get_connection()
            c = conn.cursor()
            value_str = json.dumps(value) if not isinstance(value, str) else value
            c.execute(
                "INSERT OR REPLACE INTO conversation_metadata (session_id, key, value) VALUES (?, ?, ?)",
                (session_id, key, value_str)
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error setting metadata: {e}")
            return False

    def get_metadata(self, session_id: str, key: str) -> Optional[Any]:
        """
        Get session metadata
        """
        conn = self._get_connection()
        c = conn.cursor()
        c.execute(
            "SELECT value FROM conversation_metadata WHERE session_id = ? AND key = ?",
            (session_id, key)
        )
        row = c.fetchone()
        conn.close()

        if row:
            try:
                return json.loads(row[0])
            except json.JSONDecodeError:
                return row[0]
        return None
