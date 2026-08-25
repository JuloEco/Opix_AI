# ============================================================================
# tools/base.py — Interface commune des outils (README section 9)
#
#   "Externaliser les tâches précises." (README, section 14.2)
#
# Chaque outil expose : un nom, une description, et une méthode run(**kwargs)
# qui renvoie un ToolResult. Le Router (voir router/router.py) et l'API
# (api/tools.py) ne connaissent que cette interface, jamais l'implémentation
# concrète de chaque outil — même philosophie que model/llm_client.py pour
# le modèle (README section 2 : "Cette séparation permet de remplacer chaque
# composant indépendamment.").
# ============================================================================

from __future__ import annotations

import abc
from dataclasses import dataclass, field


@dataclass
class ToolResult:
    ok: bool
    output: str
    error: str | None = None
    data: dict = field(default_factory=dict)


class Tool(abc.ABC):
    name: str = "tool"
    description: str = ""

    @abc.abstractmethod
    def run(self, **kwargs) -> ToolResult:
        raise NotImplementedError

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description}
