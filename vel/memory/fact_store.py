# vel/memory/fact_store.py
"""
Fact Store: Namespaced key-value store for long-term structured data.

This is NOT traditional episodic memory (conversation turns). Instead, it's a
persistent store for facts, preferences, and structured metadata that should
persist across conversations.

Use cases:
- User preferences (theme, language, expertise level)
- Project metadata (current project, technologies used)
- Domain knowledge (company facts, API endpoints)
- Application state (feature flags, configuration)
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any
from pathlib import Path
import sqlite3, json, time

class FactStore:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path)
        self.db.execute("PRAGMA journal_mode=WAL;")
        self.db.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS kv_store(
          namespace TEXT NOT NULL,
          k TEXT NOT NULL,
          v TEXT NOT NULL,
          created_at REAL DEFAULT (strftime('%s','now')),
          updated_at REAL DEFAULT (strftime('%s','now')),
          PRIMARY KEY (namespace, k)
        );
        CREATE INDEX IF NOT EXISTS idx_kv_ns ON kv_store(namespace);
        """)
        self.db.commit()

    def put(self, namespace: str, key: str, value: Any):
        js = json.dumps(value)
        self.db.execute("""
            INSERT INTO kv_store(namespace,k,v) VALUES(?,?,?)
            ON CONFLICT(namespace,k) DO UPDATE SET v=excluded.v, updated_at=?
        """, (namespace, key, js, time.time()))
        self.db.commit()

    def get(self, namespace: str, key: str) -> Optional[Any]:
        r = self.db.execute("SELECT v FROM kv_store WHERE namespace=? AND k=?", (namespace, key)).fetchone()
        return None if not r else json.loads(r["v"])

    def list(self, namespace: str, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self.db.execute(
            "SELECT k, v, updated_at FROM kv_store WHERE namespace=? ORDER BY updated_at DESC LIMIT ?",
            (namespace, limit)
        ).fetchall()
        return [{"key": r["k"], "value": json.loads(r["v"]), "updated_at": r["updated_at"]} for r in rows]

    def search(self, namespace: str, prefix: str, limit: int = 50) -> List[Dict[str, Any]]:
        like = f"{prefix}%"
        rows = self.db.execute(
            "SELECT k, v, updated_at FROM kv_store WHERE namespace=? AND k LIKE ? ORDER BY updated_at DESC LIMIT ?",
            (namespace, like, limit)
        ).fetchall()
        return [{"key": r["k"], "value": json.loads(r["v"]), "updated_at": r["updated_at"]} for r in rows]

    def delete(self, namespace: str, key: str):
        self.db.execute("DELETE FROM kv_store WHERE namespace=? AND k=?", (namespace, key))
        self.db.commit()
