# ============================================================================
# reasoning/planner.py — Compréhension, décomposition et résolution
#
# Couvre les 3 premières étapes du flux README (section 6) :
#   Question → Compréhension → Décomposition → Résolution
#
# Important (README, section 6, encadré "Important") : augmenter
# `max_new_tokens` ne suffit pas — le modèle doit apprendre à structurer sa
# réponse. Ce module ne remplace donc pas le fine-tuning nécessaire (README,
# feuille de route Phase 3 : "Dataset de problèmes multi-étapes"), il fournit
# l'échafaudage (prompts + parsing) qui exploite cette capacité une fois
# qu'elle est entraînée, et reste utilisable "as-is" en zero-shot en attendant.
# ============================================================================

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .prompts import DECOMPOSITION_PROMPT, RESOLUTION_PROMPT


@dataclass
class Plan:
    understanding: str
    steps: list[str]
    raw_text: str = ""


@dataclass
class Resolution:
    step_results: list[str] = field(default_factory=list)
    final_answer: str = ""
    raw_text: str = ""


class Planner:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def decompose(self, question: str, max_new_tokens: int = 220) -> Plan:
        prompt = self.llm_client.format_chat_prompt(
            DECOMPOSITION_PROMPT.format(question=question)
        )
        raw = self.llm_client.generate(
            prompt, max_new_tokens=max_new_tokens, temperature=0.3, stop=["<|Utilisateur|>"]
        )
        text = self.llm_client._strip_prompt(raw, prompt)
        return self._parse_plan(text)

    def resolve(self, question: str, plan: Plan, max_new_tokens: int = 300) -> Resolution:
        plan_text = self._format_plan_for_resolution(plan)
        prompt = self.llm_client.format_chat_prompt(
            RESOLUTION_PROMPT.format(question=question, plan=plan_text)
        )
        raw = self.llm_client.generate(
            prompt, max_new_tokens=max_new_tokens, temperature=0.3, stop=["<|Utilisateur|>"]
        )
        text = self.llm_client._strip_prompt(raw, prompt)
        return self._parse_resolution(text)

    # -- Parsing ---------------------------------------------------------
    _STEP_LINE_RE = re.compile(r"^\s*\d+[.)]\s*(.+)$", re.MULTILINE)

    def _parse_plan(self, text: str) -> Plan:
        understanding_match = re.search(r"COMPRÉHENSION\s*:\s*(.+)", text)
        understanding = understanding_match.group(1).strip() if understanding_match else ""

        steps_section = text.split("ÉTAPES:", 1)[-1] if "ÉTAPES:" in text else text
        steps = [m.strip() for m in self._STEP_LINE_RE.findall(steps_section)]

        if not steps:
            # Repli : si le modèle n'a pas respecté le format, on traite la
            # question comme une étape unique plutôt que d'échouer.
            steps = [understanding] if understanding else []

        return Plan(understanding=understanding, steps=steps, raw_text=text)

    def _parse_resolution(self, text: str) -> Resolution:
        step_results = [m.strip() for m in self._STEP_LINE_RE.findall(text.split("RÉSOLUTION:", 1)[-1])]

        final_match = re.search(r"RÉPONSE_FINALE\s*:\s*(.+)", text, re.DOTALL)
        final_answer = final_match.group(1).strip() if final_match else text.strip()

        return Resolution(step_results=step_results, final_answer=final_answer, raw_text=text)

    @staticmethod
    def _format_plan_for_resolution(plan: Plan) -> str:
        lines = [f"Compréhension : {plan.understanding}"] if plan.understanding else []
        lines += [f"{i + 1}. {step}" for i, step in enumerate(plan.steps)]
        return "\n".join(lines)
