# ============================================================================
# memory/short_term.py — Mémoire conversationnelle (README section 8)
#
#   "Mémoire conversationnelle" (feuille de route, Phase 6)
#
# Fenêtre glissante en mémoire (par session), avec troncature approximative
# par caractères plutôt que par tokens — pas de dépendance au tokenizer
# d'Opsiom ici (README section 14.5 : "Context management: ne pas envoyer
# inutilement de très longs contextes"). Une session correspond typiquement
# à une conversation utilisateur (identifiée côté API par un session_id).
#
# ⚠️ PAS persistant entre redémarrages du process — pour la mémoire
# long-terme, voir memory/persistent.py.
# ============================================================================

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str


class ConversationMemory:
    def __init__(self, max_turns: int = 12, max_chars: int = 4000):
        self.max_turns = max_turns
        self.max_chars = max_chars
        self._sessions: dict[str, deque[Turn]] = {}

    def add(self, session_id: str, role: str, content: str) -> None:
        history = self._sessions.setdefault(session_id, deque(maxlen=self.max_turns))
        history.append(Turn(role=role, content=content))

    def get(self, session_id: str) -> list[Turn]:
        return list(self._sessions.get(session_id, []))

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def format_for_prompt(self, session_id: str, user_tag: str, assistant_tag: str) -> str:
        """Formate l'historique avec les tags de dialogue, tronqué par
        caractères en partant de la fin (les tours les plus récents priment)
        si `max_chars` est dépassé."""
        turns = self.get(session_id)
        formatted = []
        for turn in turns:
            tag = user_tag if turn.role == "user" else assistant_tag
            formatted.append(f"{tag}\n{turn.content}")

        text = "\n".join(formatted)
        if len(text) > self.max_chars:
            # Tronque en tête (le plus ancien) pour garder les tours récents.
            text = text[-self.max_chars:]
            candidates = [i for i in (text.find(user_tag), text.find(assistant_tag)) if i != -1]
            first_tag_idx = min(candidates) if candidates else 0
            text = text[first_tag_idx:]
        return text
