# ============================================================================
# tools/registry.py — Registre central des outils (README section 9 et 10)
#
# Le Router (router/router.py) et l'API (api/tools.py) passent par ce
# registre pour découvrir et exécuter les outils disponibles, sans jamais
# importer une implémentation concrète directement.
# ============================================================================

from __future__ import annotations

from .base import Tool, ToolResult
from .calculator import CalculatorTool
from .datetime_tool import DateTimeTool
from .python_sandbox import PythonSandboxTool
from .web_search import WebSearchTool


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for tool in (tools if tools is not None else self.default_tools()):
            self.register(tool)

    @staticmethod
    def default_tools() -> list[Tool]:
        """Outils activés par défaut (README, feuille de route Phase 5).
        `web_search` est enregistré mais renverra une erreur explicite tant
        qu'un backend n'a pas été injecté (voir tools/web_search.py)."""
        return [CalculatorTool(), DateTimeTool(), PythonSandboxTool(), WebSearchTool()]

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[dict]:
        return [t.to_dict() for t in self._tools.values()]

    def execute(self, name: str, **kwargs) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(
                ok=False, output="", error=f"Outil inconnu : '{name}'. Outils disponibles : {list(self._tools)}"
            )
        try:
            return tool.run(**kwargs)
        except Exception as e:  # pragma: no cover - garde-fou
            return ToolResult(ok=False, output="", error=f"Erreur lors de l'exécution de '{name}' : {e}")
