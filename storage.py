import sqlite3
import uuid
from datetime import datetime

from config import DB_PATH


def ensure_db():
    conn = sqlite3.connect(DB_PATH)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES chat_sessions(id)
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )

    conn.commit()

    row = conn.execute("SELECT id FROM chat_sessions LIMIT 1").fetchone()
    if not row:
        session_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO chat_sessions(id, title, created_at) VALUES (?, ?, ?)",
            (session_id, "Основной чат", datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()

    conn.close()


def get_setting(key, default=""):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
    conn.close()
    return row[0] if row else default


def set_setting(key, value):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO app_settings(key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, str(value)),
    )
    conn.commit()
    conn.close()


def list_sessions():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, title FROM chat_sessions ORDER BY created_at ASC"
    ).fetchall()
    conn.close()
    return rows


def create_session(title=None):
    session_id = str(uuid.uuid4())
    if not title:
        title = f"Сценарий {datetime.now().strftime('%H:%M:%S')}"
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO chat_sessions(id, title, created_at) VALUES (?, ?, ?)",
        (session_id, title, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()
    return session_id


def rename_session_if_needed(session_id, first_user_text):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT title FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
    if row and row[0].startswith("Сценарий "):
        title = first_user_text.strip()[:40] or row[0]
        conn.execute("UPDATE chat_sessions SET title = ? WHERE id = ?", (title, session_id))
        conn.commit()
    conn.close()


def save_message(session_id, role, content):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO messages(session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (session_id, role, content, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def load_session_history(session_id, limit=1000):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id ASC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    conn.close()
    return rows


def get_recent_messages(session_id, limit=120):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    conn.close()
    rows.reverse()
    return [{"role": role, "content": content} for role, content in rows]