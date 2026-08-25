# ============================================================================
# tools/datetime_tool.py — Outil "Date / heure" (README, feuille de route
# Phase 5 : "Date / heure")
# ============================================================================

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .base import Tool, ToolResult

DEFAULT_TIMEZONE = "Europe/Paris"

_JOURS_FR = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
_MOIS_FR = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
]


class DateTimeTool(Tool):
    name = "datetime"
    description = "Renvoie la date et l'heure actuelles (fuseau horaire configurable, défaut Europe/Paris)."

    def run(self, timezone: str = DEFAULT_TIMEZONE, **kwargs) -> ToolResult:
        try:
            tz = ZoneInfo(timezone)
        except ZoneInfoNotFoundError:
            return ToolResult(ok=False, output="", error=f"Fuseau horaire inconnu : '{timezone}'.")

        now = datetime.now(tz)
        jour = _JOURS_FR[now.weekday()]
        mois = _MOIS_FR[now.month - 1]
        formatted = f"{jour} {now.day} {mois} {now.year}, {now.strftime('%H:%M')} ({timezone})"

        return ToolResult(
            ok=True,
            output=formatted,
            data={
                "iso": now.isoformat(),
                "timestamp": now.timestamp(),
                "timezone": timezone,
            },
        )
