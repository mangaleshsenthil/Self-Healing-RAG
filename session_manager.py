"""
session_manager.py
------------------
Manages chat sessions stored as local JSON files.
Each session = one JSON file in the sessions/ folder.
"""

import os
import json
import uuid
from datetime import datetime
from typing import Optional

SESSIONS_DIR = "sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)


def _session_path(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def create_session(title: str = "New Chat") -> dict:
    """Create a brand new session."""
    session = {
        "id":         str(uuid.uuid4()),
        "title":      title,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "messages":   []
    }
    _save_session(session)
    return session


def load_session(session_id: str) -> Optional[dict]:
    """Load a session by ID. Returns None if not found."""
    path = _session_path(session_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_session(session: dict):
    """Save session to disk."""
    path = _session_path(session["id"])
    session["updated_at"] = datetime.now().isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2, ensure_ascii=False)


def add_message(session_id: str, role: str, content: str, meta: dict = None) -> dict:
    """
    Add a message to a session.
    role: 'user' | 'assistant'
    meta: extra pipeline data (chunks, critique, attempts, etc.)
    """
    session = load_session(session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found.")

    message = {
        "id":         str(uuid.uuid4()),
        "role":       role,
        "content":    content,
        "timestamp":  datetime.now().isoformat(),
        "meta":       meta or {}
    }
    session["messages"].append(message)

    # Auto-title from first user message
    if role == "user" and session["title"] == "New Chat":
        session["title"] = content[:50] + ("..." if len(content) > 50 else "")

    _save_session(session)
    return message


def list_sessions() -> list:
    """Return all sessions sorted by most recently updated."""
    sessions = []
    for fname in os.listdir(SESSIONS_DIR):
        if fname.endswith(".json"):
            path = os.path.join(SESSIONS_DIR, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    s = json.load(f)
                    sessions.append({
                        "id":         s["id"],
                        "title":      s["title"],
                        "created_at": s["created_at"],
                        "updated_at": s["updated_at"],
                        "message_count": len(s["messages"])
                    })
            except Exception:
                continue
    return sorted(sessions, key=lambda x: x["updated_at"], reverse=True)


def delete_session(session_id: str) -> bool:
    """Delete a session file."""
    path = _session_path(session_id)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False
