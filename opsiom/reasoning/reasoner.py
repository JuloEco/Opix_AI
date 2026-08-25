# ============================================================================
# reasoning/reasoner.py — Orchestrateur du mode Reasoning (README section 6)
#
#   Question → Compréhension → Décomposition → Résolution → Vérification →
#   Réponse finale
#
# La réponse finale est renvoyée séparément de la trace de raisonnement
# (README, feuille de route Phase 3 : "Réponse finale séparée du
# raisonnement"), pour que l'API ne renvoie au frontend que ce qui doit être
# affiché à l'utilisateur, tout en gardant la trace disponible pour le debug
# ou l'évaluation (README section 16).
# ============================================================================

from __future__ import annotations

from dataclasses import dataclass, field

from .planner import Planner, Plan, Resolution
from .prompts import DIRECT_ANSWER_PROMPT
from .verifier import Verifier, VerificationResult


@dataclass
class ReasoningTrace:
    question: str
    plan: Plan | None = None
    resolution: Resolution | None = None
    verification: VerificationResult | None = None
    final_answer: str = ""
    used_reasoning: bool = True


class Reasoner:
    """Point d'entrée unique du mode Reasoning. Peut être appelé directement,
    ou via l'endpoint POST /api/reason (voir api/reason.py)."""

    def __init__(self, llm_client, verify: bool = True):
        self.llm_client = llm_client
        self.planner = Planner(llm_client)
        self.verifier = Verifier(llm_client) if verify else None

    def should_reason(self, question: str, min_length: int = 15) -> bool:
        """Heuristique simple pour décider si une question mérite le mode
        Reasoning (multi-étapes) ou une réponse directe (README, section 10 :
        c'est normalement le rôle du Router, mais une heuristique locale
        suffit pour un usage autonome de ce module).

        Déclenche le raisonnement si la question contient un chiffre (souvent
        signe de calcul), un connecteur logique, ou est simplement assez
        longue pour être multi-étapes."""
        lowered = question.lower()
        has_digit = any(ch.isdigit() for ch in question)
        has_connector = any(
            kw in lowered for kw in ("si ", "combien", "pourquoi", "explique", "puis", "et si", "sachant que")
        )
        return has_digit or has_connector or len(question) >= min_length

    def answer(self, question: str, force_reasoning: bool | None = None) -> ReasoningTrace:
        use_reasoning = self.should_reason(question) if force_reasoning is None else force_reasoning

        if not use_reasoning:
            direct = self.llm_client.chat(question, max_new_tokens=150, temperature=0.7)
            return ReasoningTrace(question=question, final_answer=direct, used_reasoning=False)

        plan = self.planner.decompose(question)
        resolution = self.planner.resolve(question, plan)
        final_answer = resolution.final_answer

        verification = None
        if self.verifier is not None:
            resolution_text = "\n".join(resolution.step_results) or resolution.raw_text
            verification = self.verifier.verify(question, resolution_text, final_answer)
            if verification.verdict == "INCORRECT":
                final_answer = verification.final_answer

        return ReasoningTrace(
            question=question,
            plan=plan,
            resolution=resolution,
            verification=verification,
            final_answer=final_answer,
            used_reasoning=True,
        )
