# ============================================================================
# router/router.py — Router / Agent (README section 10 et 18)
#
#   Question -> Router ("Quelle capacité utiliser ?") -> Reasoning / RAG / Tools
#            -> Opsiom -> Vérification -> Réponse
#
# Point d'entrée unique recommandé par le README (section 18, "Architecture
# finale recommandée") pour l'API principale : décide quelle(s) capacité(s)
# mobiliser, les orchestre, et renvoie une réponse unique. Chaque capacité
# (Reasoner, RAGPipeline, ToolRegistry) reste indépendante et injectée en
# constructeur (README section 2) — le Router fonctionne en mode dégradé si
# l'une d'elles n'est pas fournie (ex: pas de RAG -> jamais de décision "rag").
# ============================================================================

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Capture l'expression arithmétique elle-même (ex: "12 + 30" dans "Combien
# font 12 + 30 ?"), pas toute la phrase — tools/calculator.py n'accepte que
# de l'arithmétique pure (voir tools/calculator.py, safe_eval).
_CALCUL_RE = re.compile(r"[\d.,]+\s*(?:[+\-*/×÷^]\s*[\d.,()]+\s*)+")
_DATE_KEYWORDS = ("quelle heure", "quel jour", "date d'aujourd'hui", "on est quel jour", "quelle est la date")
_SEARCH_KEYWORDS = ("cherche sur internet", "recherche sur le web", "trouve sur internet", "actualité", "dernières nouvelles")
_CODE_KEYWORDS = ("exécute ce code", "lance ce script", "vérifie ce code python")
_DOC_KEYWORDS = ("dans la documentation", "d'après mes documents", "dans le cours", "selon le fichier")


@dataclass
class RouterDecision:
    capability: str  # "tool" | "rag" | "reasoning" | "direct"
    tool_name: str | None = None
    tool_args: dict = field(default_factory=dict)
    reason: str = ""


@dataclass
class AgentAnswer:
    response: str
    capability_used: str
    sources: list[str] = field(default_factory=list)
    tool_result: dict | None = None
    trace: dict | None = None


class Router:
    """Décide quelle capacité mobiliser pour répondre à une question, et
    orchestre l'appel correspondant. Reste volontairement basé sur des
    heuristiques simples (mots-clés/regex) plutôt qu'un classifieur appris —
    cohérent avec `Reasoner.should_reason()` (README, INTEGRATION.md section
    4) et suffisant pour un premier Router fonctionnel (README, feuille de
    route Phase 7). À terme, ces heuristiques peuvent être remplacées par un
    appel à Opsiom lui-même (classification zero-shot du type de requête)
    sans changer l'interface de `Router.answer()`."""

    def __init__(
        self,
        llm_client,
        reasoner=None,
        rag_pipeline=None,
        tool_registry=None,
        conversation_memory=None,
        persistent_memory=None,
    ):
        self.llm_client = llm_client
        self.reasoner = reasoner
        self.rag_pipeline = rag_pipeline
        self.tool_registry = tool_registry
        self.conversation_memory = conversation_memory
        self.persistent_memory = persistent_memory

    def decide(self, question: str) -> RouterDecision:
        lowered = question.lower()

        if self.tool_registry is not None:
            calc_match = _CALCUL_RE.search(question)
            if calc_match:
                return RouterDecision(
                    "tool", tool_name="calculator", tool_args={"expression": calc_match.group(0).strip()},
                    reason="expression arithmétique détectée",
                )
            if any(kw in lowered for kw in _DATE_KEYWORDS):
                return RouterDecision("tool", tool_name="datetime", reason="question sur la date/heure")
            if any(kw in lowered for kw in _CODE_KEYWORDS):
                return RouterDecision("tool", tool_name="python", reason="exécution de code demandée")
            if any(kw in lowered for kw in _SEARCH_KEYWORDS):
                return RouterDecision(
                    "tool", tool_name="web_search", tool_args={"query": question},
                    reason="recherche web explicitement demandée",
                )

        if self.rag_pipeline is not None and any(kw in lowered for kw in _DOC_KEYWORDS):
            return RouterDecision("rag", reason="référence explicite à des documents indexés")

        if self.reasoner is not None and self.reasoner.should_reason(question):
            return RouterDecision("reasoning", reason="question multi-étapes probable")

        return RouterDecision("direct", reason="conversation simple")

    def _memory_context(self, user_id: str | None, question: str) -> str:
        """Récupère les souvenirs persistants pertinents (README section 8)
        et les formate comme un petit préambule de contexte, à la manière du
        contexte RAG (rag/pipeline.py, CONTEXT_PROMPT_TEMPLATE)."""
        if self.persistent_memory is None or not user_id:
            return ""
        try:
            results = self.persistent_memory.search(user_id, question, top_k=3)
        except Exception:
            return ""
        lines = [m.text for m, score in results if score > 0.2]
        if not lines:
            return ""
        return "Éléments à connaître sur l'utilisateur :\n" + "\n".join(f"- {line}" for line in lines)

    def answer(
        self,
        question: str,
        session_id: str | None = None,
        user_id: str | None = None,
        force_capability: str | None = None,
        **gen_kwargs,
    ) -> AgentAnswer:
        decision = self.decide(question) if force_capability is None else RouterDecision(force_capability)

        if self.conversation_memory is not None and session_id:
            self.conversation_memory.add(session_id, "user", question)

        if decision.capability == "tool":
            result = self.tool_registry.execute(decision.tool_name, **decision.tool_args)
            if result.ok:
                # Le résultat brut de l'outil est reformulé par Opsiom pour
                # rester dans le ton conversationnel (README section 9), pas
                # renvoyé tel quel.
                followup = (
                    f"L'outil '{decision.tool_name}' a renvoyé : {result.output}\n"
                    f"Reformule ce résultat en une réponse claire et naturelle à la question : {question}"
                )
                response = self.llm_client.chat(followup, max_new_tokens=120, temperature=0.5)
            else:
                response = f"Je n'ai pas pu utiliser l'outil demandé : {result.error}"
            answer = AgentAnswer(
                response=response, capability_used="tool",
                tool_result={"tool": decision.tool_name, "ok": result.ok, "output": result.output, "error": result.error},
            )

        elif decision.capability == "rag":
            rag_result = self.rag_pipeline.answer(question, **gen_kwargs)
            answer = AgentAnswer(response=rag_result.answer, capability_used="rag", sources=rag_result.sources)

        elif decision.capability == "reasoning":
            trace = self.reasoner.answer(question)
            answer = AgentAnswer(
                response=trace.final_answer,
                capability_used="reasoning",
                trace={
                    "understanding": trace.plan.understanding if trace.plan else None,
                    "steps": trace.plan.steps if trace.plan else [],
                },
            )

        else:
            context = self._memory_context(user_id, question)
            prompt_question = f"{context}\n\n{question}" if context else question
            response = self.llm_client.chat(prompt_question, **gen_kwargs)
            answer = AgentAnswer(response=response, capability_used="direct")

        if self.conversation_memory is not None and session_id:
            self.conversation_memory.add(session_id, "assistant", answer.response)

        return answer
