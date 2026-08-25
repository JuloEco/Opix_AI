# ============================================================================
# rag/reranker.py — Re-ranking Top 20 → Top 3-5 (README section 7.1)
#
#   Ne pas faire :  Question → recherche → 20 documents → Opsiom
#   Préférer     :  ... → Top 20 → Re-ranking → Top 3-5 → Opsiom
#
# La recherche vectorielle (bi-encoder) est rapide mais approximative. Le
# re-ranking utilise un modèle plus coûteux mais plus précis (cross-encoder,
# qui regarde la question ET le chunk ensemble) sur un petit nombre de
# candidats seulement — bon compromis qualité/coût, et surtout ça réduit le
# bruit envoyé au petit modèle Opsiom (~196M paramètres, cf. README 14).
# ============================================================================

from __future__ import annotations

import re
from collections import Counter

from .database import Chunk

DEFAULT_CROSS_ENCODER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class Reranker:
    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER):
        self.model_name = model_name
        self._backend = None
        try:
            from sentence_transformers import CrossEncoder

            self._backend = CrossEncoder(model_name)
            print(f"✅ Reranker: cross-encoder ('{model_name}')")
        except Exception as e:
            print(
                f"⚠️  cross-encoder indisponible ({e}). "
                "Repli sur un score lexical (chevauchement de mots) pour le re-ranking."
            )

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """Reçoit les candidats (chunk, score_vectoriel) déjà triés par la
        recherche vectorielle, et renvoie les `top_k` meilleurs après
        re-scoring. Le score renvoyé est celui du reranker (pas le score
        vectoriel d'origine)."""
        if not candidates:
            return []

        if self._backend is not None:
            pairs = [(query, chunk.text) for chunk, _ in candidates]
            scores = self._backend.predict(pairs)
            reranked = sorted(zip((c for c, _ in candidates), scores), key=lambda x: x[1], reverse=True)
            return [(chunk, float(score)) for chunk, score in reranked[:top_k]]

        return self._lexical_rerank(query, candidates, top_k)

    # -- Repli sans dépendance externe --------------------------------------
    _TOKEN_RE = re.compile(r"[a-zàâäéèêëïîôöùûüçñ0-9]+", re.IGNORECASE)

    def _lexical_rerank(
        self, query: str, candidates: list[tuple[Chunk, float]], top_k: int
    ) -> list[tuple[Chunk, float]]:
        """Score = chevauchement pondéré des tokens de la requête présents
        dans le chunk (proche d'un BM25 très simplifié), combiné avec le
        score vectoriel d'origine pour ne pas perdre l'information sémantique
        quand le recouvrement lexical est faible."""
        query_tokens = Counter(self._TOKEN_RE.findall(query.lower()))
        if not query_tokens:
            return candidates[:top_k]

        scored = []
        for chunk, vector_score in candidates:
            chunk_tokens = Counter(self._TOKEN_RE.findall(chunk.text.lower()))
            overlap = sum(min(count, chunk_tokens[term]) for term, count in query_tokens.items())
            lexical_score = overlap / max(1, sum(query_tokens.values()))
            combined = 0.5 * lexical_score + 0.5 * vector_score
            scored.append((chunk, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
