# ============================================================================
# api/agent.py — Endpoint POST /api/agent (README section 10, 11 et 18)
#
# Point d'entrée "intelligent" unique : le Router décide seul quelle
# capacité mobiliser (direct / reasoning / RAG / outil) pour chaque message.
# Complète /api/chat (direct), /api/reason, /api/rag et /api/tools/execute,
# qui restent disponibles pour un appel explicite à une capacité précise.
# ============================================================================

from __future__ import annotations

from flask import Blueprint, jsonify, request


def build_agent_blueprint(router) -> Blueprint:
    bp = Blueprint("agent", __name__)

    @bp.post("/api/agent")
    def agent_endpoint():
        data = request.get_json(silent=True) or {}
        question = (data.get("message") or data.get("question") or "").strip()
        if not question:
            return jsonify({"error": "Le champ 'message' (ou 'question') est requis."}), 400
        if len(question) > 2000:
            return jsonify({"error": "Question trop longue (2000 caractères max)."}), 413

        session_id = data.get("session_id")
        user_id = data.get("user_id")
        force_capability = data.get("force_capability")  # "tool" | "rag" | "reasoning" | "direct" | absent

        try:
            result = router.answer(
                question,
                session_id=session_id,
                user_id=user_id,
                force_capability=force_capability,
                max_new_tokens=data.get("max_new_tokens", 200),
                temperature=data.get("temperature", 0.7),
            )
        except Exception as e:  # pragma: no cover - garde-fou API
            return jsonify({"error": f"Erreur agent : {e}"}), 500

        return jsonify(
            {
                "response": result.response,
                "capability_used": result.capability_used,
                "sources": result.sources,
                "tool_result": result.tool_result,
                "trace": result.trace,
            }
        )

    return bp
