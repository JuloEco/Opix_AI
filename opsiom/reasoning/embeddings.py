# ============================================================================
# rag/embeddings.py — Encodage vectoriel des chunks et des questions
#
# Opsiom (le LLM génératif) n'a pas d'embedding sémantique entraîné pour la
# recherche — ce n'est pas son rôle (cf. README section 7 : "Le RAG permet au
# modèle d'utiliser des informations externes sans devoir les mémoriser").
# On utilise donc un modèle d'embeddings dédié, séparé du modèle génératif.
#
# Priorité : sentence-transformers multilingue (bonne qualité, gère le
# français). Repli automatique sur une méthode sans dépendance lourde
# (hashing + TF-IDF simplifié) si `sentence-transformers` n'est pas installé,
# pour que le RAG reste utilisable même dans un environnement minimal.
# ============================================================================

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np

DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingModel:
    """Encode des textes en vecteurs numpy normalisés (cosinus = produit
    scalaire). Utilisé à la fois pour indexer les chunks et pour encoder les
    questions au moment de la recherche."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self._backend = None
        self._dim = None
        try:
            from sentence_transformers import SentenceTransformer

            self._backend = SentenceTransformer(model_name)
            self._dim = self._backend.get_sentence_embedding_dimension()
            print(f"✅ EmbeddingModel: sentence-transformers ('{model_name}', dim={self._dim})")
        except Exception as e:
            print(
                f"⚠️  sentence-transformers indisponible ({e}). "
                "Repli sur un encodeur hashing/TF-IDF simplifié (moins précis, "
                "mais sans dépendance externe)."
            )
            self._dim = 512  # dimension du repli hashing

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Renvoie un tableau (N, dim) de vecteurs normalisés (norme L2 = 1)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        if self._backend is not None:
            vecs = self._backend.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(vecs, dtype=np.float32)

        return self._hashing_encode(texts)

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query])[0]

    # -- Repli sans dépendance externe --------------------------------------
    _TOKEN_RE = re.compile(r"[a-zàâäéèêëïîôöùûüçñ0-9]+", re.IGNORECASE)

    def _hashing_encode(self, texts: list[str]) -> np.ndarray:
        """Encodage TF-IDF-like avec hashing trick : rapide, aucune dépendance,
        qualité nettement inférieure à un vrai modèle d'embeddings mais
        suffisant pour dépanner ou pour des tests hors-ligne."""
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        doc_freq: Counter = Counter()
        tokenized = []
        for text in texts:
            tokens = self._TOKEN_RE.findall(text.lower())
            tokenized.append(tokens)
            doc_freq.update(set(tokens))

        n_docs = max(1, len(texts))
        for i, tokens in enumerate(tokenized):
            if not tokens:
                continue
            term_freq = Counter(tokens)
            for term, tf in term_freq.items():
                idf = math.log(1 + n_docs / (1 + doc_freq[term]))
                weight = (tf / len(tokens)) * idf
                bucket = int(hashlib.md5(term.encode("utf-8")).hexdigest(), 16) % self.dim
                vecs[i, bucket] += weight

        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms
