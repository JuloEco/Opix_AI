# ============================================================================
# api/tools.py — Endpoints outils (README section 11)
#
#   GET  /api/tools            -> liste des outils disponibles
#   POST /api/tools/execute    -> exécute un outil
#
# À enregistrer dans app.py :
#   from api.tools import build_tools_blueprint
#   app.register_blueprint(build_tools_blueprint(tool_registry))
# ============================================================================

from __future__ import annotations

from flask import Blueprint, jsonify, request


def build_tools_blueprint(tool_registry) -> Blueprint:
    bp = Blueprint("tools", __name__)

    @bp.get("/api/tools")
    def list_tools():
        return jsonify({"tools": tool_registry.list_tools()})

    @bp.post("/api/tools/execute")
    def execute_tool():
        data = request.get_json(silent=True) or {}
        name = (data.get("tool") or "").strip()
        args = data.get("args") or {}
        if not name:
            return jsonify({"error": "Le champ 'tool' est requis."}), 400
        if not isinstance(args, dict):
            return jsonify({"error": "Le champ 'args' doit être un objet JSON."}), 400

        result = tool_registry.execute(name, **args)
        return jsonify({"ok": result.ok, "output": result.output, "error": result.error, "data": result.data})

    return bp
