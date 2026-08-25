# ============================================================================
# tools/calculator.py — Outil "Calculatrice" (README section 9 et 14.2)
#
# Évaluation arithmétique sûre (pas d'exec/eval) : seule une grammaire
# restreinte (+ - * / ** parenthèses, nombres, et un petit nombre de
# fonctions mathématiques whitelistées) est acceptée, via le module `ast` —
# même approche de sécurité que reasoning/verifier.py (check_arithmetic),
# volontairement dupliquée ici pour que `tools/` reste indépendant de
# `reasoning/` (README section 2 : composants remplaçables indépendamment).
# ============================================================================

from __future__ import annotations

import ast
import math
import operator

from .base import Tool, ToolResult

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
}


class CalculatorError(ValueError):
    pass


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise CalculatorError(f"Constante non numérique interdite : {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        try:
            return _ALLOWED_BINOPS[type(node.op)](left, right)
        except ZeroDivisionError as e:
            raise CalculatorError("Division par zéro.") from e
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise CalculatorError("Fonction non autorisée.")
        if node.keywords:
            raise CalculatorError("Arguments nommés non autorisés.")
        args = [_eval_node(a) for a in node.args]
        return _ALLOWED_FUNCS[node.func.id](*args)
    raise CalculatorError(f"Expression non autorisée : {ast.dump(node)}")


def safe_eval(expression: str) -> float:
    normalized = expression.strip().replace(",", ".").replace("×", "*").replace("÷", "/")
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as e:
        raise CalculatorError(f"Expression invalide : {e}") from e
    return _eval_node(tree.body)


class CalculatorTool(Tool):
    name = "calculator"
    description = (
        "Évalue une expression arithmétique (+ - * / ** et quelques fonctions : "
        "sqrt, log, sin, cos...)."
    )

    def run(self, expression: str = "", **kwargs) -> ToolResult:
        if not expression:
            return ToolResult(ok=False, output="", error="Le paramètre 'expression' est requis.")
        try:
            result = safe_eval(expression)
        except CalculatorError as e:
            return ToolResult(ok=False, output="", error=str(e))
        return ToolResult(ok=True, output=str(result), data={"result": result})
