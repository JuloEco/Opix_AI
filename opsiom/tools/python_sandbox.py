# ============================================================================
# tools/python_sandbox.py — Outil "Python sandbox" (README section 15.1)
#
#   "Un outil Python ne doit surtout pas exécuter directement exec(user_input)
#   dans le processus Flask."
#   Utilisateur -> API -> Sandbox -> Python limité -> Résultat
#
# Implémentation :
#   1. Un filtre AST statique rejette les imports et appels manifestement
#      dangereux (os, subprocess, socket, sys, shutil, open, eval, exec,
#      __import__, accès aux dunders...) AVANT toute exécution.
#   2. Le code est exécuté dans un SOUS-PROCESSUS séparé (jamais dans le
#      processus Flask lui-même), avec un timeout et — sur POSIX — des
#      limites de ressources (CPU, mémoire, pas de fork) via `resource`.
#   3. stdout/stderr sont capturés et tronqués.
#
# ⚠️ Ceci reste un sandbox "best effort" en profondeur de défense, PAS une
# isolation complète de niveau production (pas de namespace/cgroup/conteneur
# séparé). Pour un déploiement exposé à des utilisateurs non fiables, il est
# fortement recommandé de faire tourner ce sous-processus dans un vrai
# conteneur jetable (Docker --network none --read-only, gVisor, Firecracker,
# ou un service dédié type Piston/Judge0) plutôt que de se reposer uniquement
# sur ce filtre + ces limites de ressources.
# ============================================================================

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap

from .base import Tool, ToolResult

DEFAULT_TIMEOUT_S = 5.0
DEFAULT_MEMORY_LIMIT_MB = 256
MAX_OUTPUT_CHARS = 4000

_FORBIDDEN_NAMES = {
    "os", "subprocess", "socket", "sys", "shutil", "importlib", "ctypes",
    "multiprocessing", "threading", "pathlib", "pickle", "marshal",
    "__import__", "eval", "exec", "compile", "open", "input", "exit", "quit",
    "globals", "locals", "vars", "help", "breakpoint",
}
_FORBIDDEN_ATTR_PREFIXES = ("__",)


class SandboxRejected(ValueError):
    pass


def _static_check(code: str) -> None:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise SandboxRejected(f"Erreur de syntaxe : {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name.split(".")[0] for alias in node.names]
            if any(n in _FORBIDDEN_NAMES for n in names):
                raise SandboxRejected(f"Import interdit : {', '.join(names)}")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise SandboxRejected(f"Nom interdit : '{node.id}'")
        if isinstance(node, ast.Attribute) and node.attr.startswith(_FORBIDDEN_ATTR_PREFIXES):
            raise SandboxRejected(f"Accès à un attribut interdit : '{node.attr}'")


def _make_preexec_fn(timeout_s: float, memory_limit_mb: int):
    """Construit le callback `preexec_fn` appliqué dans l'enfant juste avant
    exec() — POSIX uniquement. Renvoie None sur les plateformes où `resource`
    n'est pas disponible (ex: Windows), auquel cas seul le timeout du
    sous-processus (voir `run()`) protège encore l'appelant."""
    try:
        import resource
    except ImportError:
        return None

    def _set_limits():
        mem_bytes = memory_limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (int(timeout_s) + 2, int(timeout_s) + 2))
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))  # pas de fork depuis l'enfant
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))  # pas d'écriture fichier

    return _set_limits


class PythonSandboxTool(Tool):
    name = "python"
    description = (
        "Exécute un extrait de code Python isolé (calculs, manipulation de "
        "texte/listes) et renvoie stdout."
    )

    def __init__(self, timeout_s: float = DEFAULT_TIMEOUT_S, memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB):
        self.timeout_s = timeout_s
        self.memory_limit_mb = memory_limit_mb

    def run(self, code: str = "", **kwargs) -> ToolResult:
        if not code.strip():
            return ToolResult(ok=False, output="", error="Le paramètre 'code' est requis.")

        try:
            _static_check(code)
        except SandboxRejected as e:
            return ToolResult(ok=False, output="", error=f"Code rejeté par le filtre de sécurité : {e}")

        wrapped = textwrap.dedent(code)
        preexec_fn = _make_preexec_fn(self.timeout_s, self.memory_limit_mb) if sys.platform != "win32" else None

        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-S", "-c", wrapped],  # -I isolé, -S pas de site-packages tiers
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                preexec_fn=preexec_fn,
                env={"PATH": "/usr/bin:/bin"},  # environnement minimal, pas de secrets (HF_TOKEN, etc.)
            )
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, output="", error=f"Timeout dépassé ({self.timeout_s}s).")
        except Exception as e:  # pragma: no cover - garde-fou
            return ToolResult(ok=False, output="", error=f"Échec d'exécution du sandbox : {e}")

        stdout = (proc.stdout or "")[:MAX_OUTPUT_CHARS]
        stderr = (proc.stderr or "")[:MAX_OUTPUT_CHARS]

        if proc.returncode != 0:
            return ToolResult(ok=False, output=stdout, error=stderr or f"Code de sortie {proc.returncode}")
        return ToolResult(ok=True, output=stdout, data={"stderr": stderr})
