# ============================================================================
# rag/database.py — Base vectorielle FAISS + métadonnées des chunks
#
# Conforme au README (section 7.2 "Base vectorielle") : FAISS pour démarrer,
# avec un stockage parallèle des métadonnées (texte, source, section, page —
# section 7.3 "Chunking") puisque FAISS lui-même ne stocke que des vecteurs.
# ============================================================================

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class Chunk:
    """Un fragment de document indexé, avec ses métadonnées (README 7.3)."""

    text: str
    source: str
    chunk_id: str
    section: str | None = None
    page: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class VectorDatabase:
    """Index FAISS (produit scalaire sur vecteurs normalisés = similarité
    cosinus) + liste parallèle de métadonnées, avec sauvegarde/chargement
    disque pour ne pas ré-indexer à chaque démarrage de l'API."""

    def __init__(self, dim: int):
        import faiss

        self._faiss = faiss
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        if len(vectors) != len(chunks):
            raise ValueError("Le nombre de vecteurs doit correspondre au nombre de chunks.")
        if len(vectors) == 0:
            return
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 20) -> list[tuple[Chunk, float]]:
        if self.index.ntotal == 0:
            return []
        top_k = min(top_k, self.index.ntotal)
        q = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
        scores, idxs = self.index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def __len__(self) -> int:
        return len(self.chunks)

    # -- Persistance ---------------------------------------------------
    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        self._faiss.write_index(self.index, os.path.join(dir_path, "index.faiss"))
        with open(os.path.join(dir_path, "chunks.jsonl"), "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        print(f"💾 Base vectorielle sauvegardée dans '{dir_path}' ({len(self.chunks):,} chunks).")

    @classmethod
    def load(cls, dir_path: str) -> "VectorDatabase":
        import faiss

        index = faiss.read_index(os.path.join(dir_path, "index.faiss"))
        db = cls.__new__(cls)
        db._faiss = faiss
        db.index = index
        db.dim = index.d
        db.chunks = []
        chunks_path = os.path.join(dir_path, "chunks.jsonl")
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    db.chunks.append(Chunk(**json.loads(line)))
        print(f"📂 Base vectorielle chargée depuis '{dir_path}' ({len(db.chunks):,} chunks).")
        return db
