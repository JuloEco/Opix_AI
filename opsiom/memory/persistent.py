# ============================================================================
# memory/persistent.py — Mémoire persistante (README section 8)
#
#   "Mémoire persistante" + "Recherche sémantique" + "Gestion des souvenirs"
#   (feuille de route, Phase 6)
#
# Stockage SQLite (bibliothèque standard, aucune dépendance additionnelle) :
# un "souvenir" par ligne (texte + métadonnées + vecteur d'embedding en
# BLOB). La recherche sémantique réutilise `rag.embeddings.EmbeddingModel`
# (import paresseux pour ne pas forcer une dépendance à `rag/` si seule la
# mémoire est utilisée) et calcule la similarité cosinus en Python — la
# volumétrie attendue ici (souvenirs par utilisateur) reste largement dans
# la zone où FAISS serait une complexité inutile (contrairement à
# rag/database.py, qui indexe potentiellement beaucoup plus de documents).
# ============================================================================

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    text TEXT NOT NULL,
    category TEXT,
    created_at REAL NOT NULL,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
"""


@dataclass
class Memory:
    id: int
    user_id: str
    text: str
    category: str | None
    created_at: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "text": self.text,
            "category": self.category,
            "created_at": self.created_at,
        }


class PersistentMemory:
    def __init__(self, db_path: str = "opsiom_memory.sqlite3", embedding_model=None):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._embedding_model = embedding_model  # lazy, voir _get_embedding_model()

    def _get_embedding_model(self):
        if self._embedding_model is None:
            from rag.embeddings import EmbeddingModel  # import paresseux, voir en-tête

            self._embedding_model = EmbeddingModel()
        return self._embedding_model

    def remember(self, user_id: str, text: str, category: str | None = None) -> Memory:
        text = text.strip()
        if not text:
            raise ValueError("Impossible de mémoriser un texte vide.")

        vector = self._get_embedding_model().encode([text])[0]
        created_at = time.time()
        cur = self._conn.execute(
            "INSERT INTO memories (user_id, text, category, created_at, embedding) VALUES (?, ?, ?, ?, ?)",
            (user_id, text, category, created_at, vector.astype(np.float32).tobytes()),
        )
        self._conn.commit()
        return Memory(id=cur.lastrowid, user_id=user_id, text=text, category=category, created_at=created_at)

    def forget(self, user_id: str, memory_id: int) -> bool:
        cur = self._conn.execute("DELETE FROM memories WHERE id = ? AND user_id = ?", (memory_id, user_id))
        self._conn.commit()
        return cur.rowcount > 0

    def list_all(self, user_id: str, limit: int = 200) -> list[Memory]:
        rows = self._conn.execute(
            "SELECT id, user_id, text, category, created_at FROM memories "
            "WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [Memory(*row) for row in rows]

    def search(self, user_id: str, query: str, top_k: int = 5) -> list[tuple[Memory, float]]:
        rows = self._conn.execute(
            "SELECT id, user_id, text, category, created_at, embedding FROM memories WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        if not rows:
            return []

        query_vec = self._get_embedding_model().encode_query(query)
        scored = []
        for row_id, uid, text, category, created_at, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            score = float(np.dot(query_vec, vec))  # vecteurs déjà normalisés par EmbeddingModel
            scored.append((Memory(row_id, uid, text, category, created_at), score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def close(self) -> None:
        self._conn.close()
