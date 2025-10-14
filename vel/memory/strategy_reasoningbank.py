# vel/memory/strategy_reasoningbank.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Iterable, List, Optional, Tuple
from pathlib import Path
import sqlite3, json, numpy as np
from time import time

@dataclass
class StrategyItem:
    id: Optional[int]
    signature_json: str
    strategy_text: str
    anti_patterns: List[str]
    evidence_refs: List[str]
    confidence: float

class Embeddings:
    """Pluggable encoder: inject sentence-transformers / OpenAI / etc."""
    def __init__(self, encode_fn):
        self._encode = encode_fn
    def encode(self, texts: List[str]) -> np.ndarray:
        arr = self._encode(texts).astype(np.float32)
        return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)

class ReasoningBankStore:
    def __init__(self, db_path: str, emb: Embeddings):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path)
        self.db.execute("PRAGMA journal_mode=WAL;")
        self.db.row_factory = sqlite3.Row
        self.emb = emb
        self._init_schema()

    def _init_schema(self):
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS rb_strategies(
          id INTEGER PRIMARY KEY,
          signature_json TEXT NOT NULL,
          strategy_text  TEXT NOT NULL,
          anti_patterns  TEXT DEFAULT '[]',
          evidence_refs  TEXT DEFAULT '[]',
          confidence     REAL DEFAULT 0.5,
          created_at     REAL DEFAULT (strftime('%s','now')),
          updated_at     REAL DEFAULT (strftime('%s','now'))
        );
        CREATE TABLE IF NOT EXISTS rb_embeddings(
          strategy_id INTEGER PRIMARY KEY,
          embedding   BLOB NOT NULL,
          dim         INTEGER NOT NULL,
          FOREIGN KEY(strategy_id) REFERENCES rb_strategies(id) ON DELETE CASCADE
        );
        """)
        self.db.commit()

    @staticmethod
    def _embed_text(signature: Dict[str, Any], strategy_text: str) -> str:
        sig = " ".join(f"{k}:{v}" for k, v in sorted(signature.items()))
        return f"{sig} || strategy:{strategy_text}".strip()

    def upsert_strategy(self,
        signature: Dict[str, Any],
        strategy_text: str,
        anti_patterns: Iterable[str] = (),
        evidence_refs: Iterable[str] = (),
        confidence: float = 0.6
    ) -> int:
        sig = json.dumps(signature, sort_keys=True)
        cur = self.db.cursor()
        cur.execute("""INSERT INTO rb_strategies(signature_json,strategy_text,anti_patterns,evidence_refs,confidence)
                       VALUES (?,?,?,?,?)""",
                    (sig, strategy_text, json.dumps(list(anti_patterns)),
                     json.dumps(list(evidence_refs)), confidence))
        sid = cur.lastrowid
        vec = self.emb.encode([self._embed_text(signature, strategy_text)])[0]
        cur.execute("INSERT INTO rb_embeddings(strategy_id,embedding,dim) VALUES (?,?,?)",
                    (sid, vec.tobytes(), int(vec.shape[0])))
        self.db.commit()
        return sid

    def retrieve(self, signature: Dict[str, Any], k: int = 5, min_conf: float = 0.3) -> List[StrategyItem]:
        qvec = self.emb.encode([self._embed_text(signature, "")])[0]
        rows = self.db.execute("""SELECT s.*, e.embedding, e.dim
                                  FROM rb_strategies s JOIN rb_embeddings e ON e.strategy_id=s.id""").fetchall()
        scored: List[Tuple[float, sqlite3.Row]] = []
        for r in rows:
            if float(r["confidence"]) < min_conf:
                continue
            emb = np.frombuffer(r["embedding"], dtype=np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            sim = float(np.dot(qvec, emb))
            score = 0.75 * sim + 0.25 * float(r["confidence"])
            scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[StrategyItem] = []
        for _, r in scored[:k]:
            out.append(StrategyItem(
                id=r["id"],
                signature_json=r["signature_json"],
                strategy_text=r["strategy_text"],
                anti_patterns=json.loads(r["anti_patterns"] or "[]"),
                evidence_refs=json.loads(r["evidence_refs"] or "[]"),
                confidence=float(r["confidence"]),
            ))
        return out

    def update_confidence(self, strategy_id: int, success: bool, alpha: float = 0.1):
        row = self.db.execute("SELECT confidence FROM rb_strategies WHERE id=?", (strategy_id,)).fetchone()
        if not row:
            return
        c = float(row["confidence"]) + (alpha if success else -alpha)
        c = min(1.0, max(0.0, c))
        self.db.execute("UPDATE rb_strategies SET confidence=?, updated_at=? WHERE id=?",
                        (c, time(), strategy_id))
        self.db.commit()

    def add_anti_patterns(self, strategy_id: int, patterns: Iterable[str]):
        row = self.db.execute("SELECT anti_patterns FROM rb_strategies WHERE id=?", (strategy_id,)).fetchone()
        if not row:
            return
        ap = list({*(json.loads(row["anti_patterns"] or "[]")), *patterns})
        self.db.execute("UPDATE rb_strategies SET anti_patterns=?, updated_at=? WHERE id=?",
                        (json.dumps(ap), time(), strategy_id))
        self.db.commit()

class ReasoningBank:
    def __init__(self, store: ReasoningBankStore):
        self.store = store
    def get_advice(self, signature: Dict[str, Any], k: int = 5) -> List[StrategyItem]:
        return self.store.retrieve(signature, k=k)
    def mark_outcome(self, strategy_ids, success: bool, fail_notes=()):
        for sid in strategy_ids:
            self.store.update_confidence(sid, success)
            if not success and fail_notes:
                self.store.add_anti_patterns(sid, fail_notes)
