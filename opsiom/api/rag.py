# ============================================================================
# api/rag.py — Endpoint POST /api/rag (README section 11)
#
# À enregistrer dans app.py :
#
#   from api.rag import build_rag_blueprint
#   app.register_blueprint(build_rag_blueprint(rag_pipeline))
#
# où `rag_pipeline` est une instance de rag.pipeline.RAGPipeline déjà
# construite au démarrage de l'API (index chargé une seule fois, pas à
# chaque requête — cf. README section 15 sur les limites de taille/timeouts).
# ============================================================================

from __future__ import annotations

from flask import Blueprint, jsonify, request


def build_rag_blueprint(rag_pipeline) -> Blueprint:
    bp = Blueprint("rag", __name__)

    @bp.post("/api/rag")
    def rag_endpoint():
        data = request.get_json(silent=True) or {}
        question = (data.get("message") or data.get("question") or "").strip()
        if not question:
            return jsonify({"error": "Le champ 'message' (ou 'question') est requis."}), 400
        if len(question) > 2000:
            return jsonify({"error": "Question trop longue (2000 caractères max)."}), 413

        # Mode "contexte seul" : ne renvoie que les passages pertinents, sans
        # appeler le modèle génératif (utile pour un frontend qui veut
        # afficher les sources avant/à la place de la réponse).
        context_only = bool(data.get("context_only", False))

        try:
            if context_only:
                chunks = rag_pipeline.retrieve_context(question)
                return jsonify(
                    {
                        "sources": sorted({c.source for c in chunks}),
                        "chunks": [c.to_dict() for c in chunks],
                    }
                )

            result = rag_pipeline.answer(
                question,
                max_new_tokens=data.get("max_new_tokens", 250),
                temperature=data.get("temperature", 0.6),
            )
            return jsonify(
                {
                    "response": result.answer,
                    "sources": result.sources,
                    "chunks": [c.to_dict() for c in result.context_chunks],
                }
            )
        except Exception as e:  # pragma: no cover - garde-fou API
            return jsonify({"error": f"Erreur RAG : {e}"}), 500

    return bp
