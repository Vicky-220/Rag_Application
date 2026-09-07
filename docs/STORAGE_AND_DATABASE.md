# Database Storage & Relational Persistence Engineering

## 1. Relational Schema Architecture

The persistent state of chat sessions, multi-turn dialogue, metadata, and retrieved source chunk citations is managed through an embedded **SQLite** database (`chat_history.db`) via [`backend/database/chat_db.py`](file:///home/vicky/Projects/Local_MultiAgentic_RAG_System/backend/database/chat_db.py).

```
┌──────────────────────────────────────────────┐
│                   sessions                   │
├──────────────────────────────────────────────┤
│ id: TEXT (PK)                                │
│ title: TEXT                                  │
│ created_at: TIMESTAMP                        │
│ updated_at: TIMESTAMP                        │
│ metadata: TEXT (JSON)                        │
└──────────────────────┬───────────────────────┘
                       │ 1:N (ON DELETE CASCADE)
                       ├─────────────────────────────────────────┐
                       ▼                                         ▼
┌──────────────────────────────────────────────┐ ┌──────────────────────────────────────────────┐
│                   messages                   │ │            conversation_metadata             │
├──────────────────────────────────────────────┤ ├──────────────────────────────────────────────┤
│ id: INTEGER (PK, AUTOINCREMENT)              │ │ id: INTEGER (PK, AUTOINCREMENT)              │
│ session_id: TEXT (FK -> sessions.id)         │ │ session_id: TEXT (FK -> sessions.id)         │
│ role: TEXT ("user" | "assistant")            │ │ key: TEXT                                    │
│ content: TEXT                                │ │ value: TEXT (JSON)                           │
│ sources: TEXT (JSON Array)                   │ │ UNIQUE(session_id, key)                      │
│ tokens_used: INTEGER                         │ └──────────────────────────────────────────────┘
│ timestamp: TIMESTAMP                         │
└──────────────────────┬───────────────────────┘
                       │ 1:N (ON DELETE CASCADE)
                       ▼
┌──────────────────────────────────────────────┐
│               chunk_references               │
├──────────────────────────────────────────────┤
│ id: INTEGER (PK, AUTOINCREMENT)              │
│ message_id: INTEGER (FK -> messages.id)      │
│ chunk_id: TEXT                               │
│ source_file: TEXT                            │
│ page_number: INTEGER                         │
│ relevance_score: REAL                        │
└──────────────────────────────────────────────┘
```

---

## 2. Foreign Key Enforcement

By default, SQLite disables foreign key constraint checks on newly established connections. In our modernized implementation, foreign key enforcement is explicitly activated:

```python
def _get_connection(self) -> sqlite3.Connection:
    """Acquires a database connection with foreign key cascades enabled."""
    conn = sqlite3.connect(self.db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn
```

When a user triggers `DELETE /api/chat/session/{session_id}`:
1. The target record in `sessions` is deleted.
2. All linked records in `messages` are automatically removed.
3. All linked chunk citations in `chunk_references` and session settings in `conversation_metadata` are pruned without leaving orphaned rows.

---

## 3. Source Attribution & Citation Tracking

Whenever a RAG-assisted response is generated, each retrieved chunk is serialized into the message payload and recorded in `chunk_references`:

```python
if sources and message_id:
    for s in sources:
        chunk_id = s.get("chunk_id") or s.get("id") or "unknown"
        source_file = s.get("source") or "unknown"
        page = s.get("page") or 0
        score = s.get("score") or 0.0
        c.execute(
            """
            INSERT INTO chunk_references 
            (message_id, chunk_id, source_file, page_number, relevance_score) 
            VALUES (?, ?, ?, ?, ?)
            """,
            (message_id, chunk_id, source_file, page, score)
        )
```

This enables:
1. **Explainable AI & Auditing**: Direct querying of exact document pages and similarity thresholds that informed a given assistant response.
2. **Frontend Rehydration**: When a previous session is selected from the sidebar, prior messages and their associated source cards are immediately loaded into the UI.
