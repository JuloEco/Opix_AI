# ============================================================================
# reasoning/verifier.py — Étape de vérification (README section 6 et 14.4)
#
#   "Vérification : Faire contrôler certains résultats." (README, section 14)
#
# Deux mécanismes, du plus fiable au plus général :
#   1. Vérification arithmétique sûre (ast, sans eval()) quand la résolution
#      contient des expressions numériques du type "24 × 0.35 = 8.40" — comme
#      dans l'exemple donné par le README lui-même.
#   2. Vérification par le modèle (auto-critique) pour les cas non numériques,
#      ou quand aucune expression arithmétique n'a été détectée.
#
# La vérification arithmétique n'exécute JAMAIS de code utilisateur arbitraire
# (pas d'exec/eval) : seule une grammaire restreinte (+ - * / ** parenthèses,
# nombres) est acceptée, évaluée via le module `ast`.
# ============================================================================

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass

from .prompts import VERIFICATION_PROMPT

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_EXPR_EQ_RE = re.compile(
    r"([0-9.,()xX×÷+\-*/\s]{3,}?)\s*=\s*([0-9]+(?:[.,][0-9]+)?)"
)


@dataclass
class VerificationResult:
    verdict: str  # "CORRECT" | "INCORRECT" | "NON_VERIFIABLE"
    final_answer: str
    details: str = ""


def _safe_eval_expr(expr: str) -> float | None:
    """Évalue une expression arithmétique simple de façon sûre (pas d'exec/eval).
    Renvoie None si l'expression n'est pas une expression numérique valide."""
    normalized = expr.replace(",", ".").replace("×", "*").replace("x", "*").replace("÷", "/")
    try:
        node = ast.parse(normalized, mode="eval").body
    except (SyntaxError, ValueError):
        return None
    return _eval_node(node)


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if left is None or right is None:
            return None
        try:
            return _ALLOWED_OPS[type(node.op)](left, right)
        except ZeroDivisionError:
            return None
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        operand = _eval_node(node.operand)
        return None if operand is None else _ALLOWED_OPS[type(node.op)](operand)
    return None


def check_arithmetic(resolution_text: str, tolerance: float = 0.01) -> list[tuple[str, bool]]:
    """Extrait les expressions du type "a op b = c" dans le texte de
    résolution et vérifie chacune indépendamment. Renvoie une liste
    (expression, est_correcte)."""
    results = []
    for match in _EXPR_EQ_RE.finditer(resolution_text):
        expr_str, claimed_str = match.group(1).strip(), match.group(2).replace(",", ".")
        # Retire un éventuel numéro de liste en tête ("1. ", "2) ") qui casse
        # le parsing arithmétique mais fait partie du format de résolution
        # attendu (voir reasoning/prompts.py, RESOLUTION_PROMPT).
        expr_str = re.sub(r"^\d+[.)]\s*", "", expr_str)
        computed = _safe_eval_expr(expr_str)
        if computed is None:
            continue
        try:
            claimed = float(claimed_str)
        except ValueError:
            continue
        is_correct = abs(computed - claimed) <= tolerance * max(1.0, abs(claimed))
        results.append((match.group(0).strip(), is_correct))
    return results


class Verifier:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def verify(
        self,
        question: str,
        resolution_text: str,
        final_answer: str,
        max_new_tokens: int = 150,
    ) -> VerificationResult:
        arithmetic_checks = check_arithmetic(resolution_text)
        if arithmetic_checks:
            all_correct = all(ok for _, ok in arithmetic_checks)
            details = "; ".join(f"{expr} -> {'OK' if ok else 'ERREUR'}" for expr, ok in arithmetic_checks)
            if all_correct:
                return VerificationResult(verdict="CORRECT", final_answer=final_answer, details=details)
            # Une erreur arithmétique détectée : on ne tente pas de deviner la
            # correction nous-mêmes, on redemande au modèle en signalant l'erreur.
            if self.llm_client is not None:
                return self._verify_with_llm(question, resolution_text, final_answer, max_new_tokens, hint=details)
            return VerificationResult(
                verdict="INCORRECT", final_answer=final_answer,
                details=f"Erreur arithmétique détectée sans llm_client pour corriger: {details}",
            )

        if self.llm_client is None:
            return VerificationResult(
                verdict="NON_VERIFIABLE", final_answer=final_answer,
                details="Pas d'expression arithmétique détectée et aucun llm_client fourni pour l'auto-critique.",
            )
        return self._verify_with_llm(question, resolution_text, final_answer, max_new_tokens)

    def _verify_with_llm(
        self, question: str, resolution_text: str, final_answer: str, max_new_tokens: int, hint: str = ""
    ) -> VerificationResult:
        prompt_text = VERIFICATION_PROMPT.format(
            question=question, resolution=resolution_text, answer=final_answer
        )
        if hint:
            prompt_text += f"\n(Indice : une vérification automatique a détecté un problème : {hint})"

        prompt = self.llm_client.format_chat_prompt(prompt_text)
        raw = self.llm_client.generate(
            prompt, max_new_tokens=max_new_tokens, temperature=0.2, stop=["<|Utilisateur|>"]
        )
        text = self.llm_client._strip_prompt(raw, prompt)

        verdict_match = re.search(r"VERDICT\s*:\s*(CORRECT|INCORRECT)", text)
        correction_match = re.search(r"CORRECTION\s*:\s*(.+)", text, re.DOTALL)

        verdict = verdict_match.group(1) if verdict_match else "NON_VERIFIABLE"
        corrected_answer = correction_match.group(1).strip() if correction_match else final_answer

        return VerificationResult(verdict=verdict, final_answer=corrected_answer, details=text)
