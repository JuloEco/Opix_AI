# ============================================================================
# tools/web_search.py — Outil "Recherche" (README section 9 et feuille de
# route Phase 5 : "Recherche", "APIs externes")
#
# Volontairement découplé d'un fournisseur précis (pas de clé API en dur
# dans ce dépôt) : `WebSearchTool` délègue à une fonction `search_fn(query,
# top_k) -> list[dict]` injectée à la construction, même logique que
# `model/llm_client.py` pour le modèle. Branche ici l'API de ton choix
# (SerpAPI, Bing Search, Brave Search, une instance SearXNG auto-hébergée...).
# ============================================================================

from __future__ import annotations

from typing import Callable

from .base import Tool, ToolResult

SearchFn = Callable[[str, int], list[dict]]


def _no_backend_configured(query: str, top_k: int) -> list[dict]:
    raise RuntimeError(
        "Aucun backend de recherche web configuré. Fournis un `search_fn` à "
        "WebSearchTool(search_fn=...) — ex: un appel à SerpAPI, Bing Search, "
        "Brave Search, ou une instance SearXNG auto-hébergée."
    )


class WebSearchTool(Tool):
    name = "web_search"
    description = "Recherche des informations sur le web (nécessite un backend configuré)."

    def __init__(self, search_fn: SearchFn = _no_backend_configured):
        self.search_fn = search_fn

    def run(self, query: str = "", top_k: int = 5, **kwargs) -> ToolResult:
        if not query:
            return ToolResult(ok=False, output="", error="Le paramètre 'query' est requis.")
        try:
            results = self.search_fn(query, top_k)
        except Exception as e:
            return ToolResult(ok=False, output="", error=str(e))

        if not results:
            return ToolResult(ok=True, output="Aucun résultat trouvé.", data={"results": []})

        lines = [
            f"- {r.get('title', '?')} — {r.get('url', '')}\n  {r.get('snippet', '')}" for r in results
        ]
        return ToolResult(ok=True, output="\n".join(lines), data={"results": results})
