# ============================================================================
# api/reason.py — Endpoint POST /api/reason (README section 11)
#
# À enregistrer dans app.py :
#
#   from api.reason import build_reasoning_blueprint
#   app.register_blueprint(build_reasoning_blueprint(reasoner))
#
# où `reasoner` est une instance de reasoning.reasoner.Reasoner construite au
# démarrage (partage le même llm_client que le reste de l'API, cf.
# model/llm_client.py, pour éviter de charger le modèle plusieurs fois).
# ============================================================================

from __future__ import annotations

from flask import Blueprint, jsonify, request


def build_reasoning_blueprint(reasoner) -> Blueprint:
    bp = Blueprint("reasoning", __name__)

    @bp.post("/api/reason")
    def reason_endpoint():
        data = request.get_json(silent=True) or {}
        question = (data.get("message") or data.get("question") or "").strip()
        if not question:
            return jsonify({"error": "Le champ 'message' (ou 'question') est requis."}), 400
        if len(question) > 2000:
            return jsonify({"error": "Question trop longue (2000 caractères max)."}), 413

        force_reasoning = data.get("force_reasoning")  # true / false / absent (auto)
        include_trace = bool(data.get("include_trace", False))

        try:
            trace = reasoner.answer(question, force_reasoning=force_reasoning)
        except Exception as e:  # pragma: no cover - garde-fou API
            return jsonify({"error": f"Erreur Reasoning : {e}"}), 500

        response = {
            "response": trace.final_answer,
            "used_reasoning": trace.used_reasoning,
        }
        # La trace complète (plan, résolution, vérification) reste séparée de
        # la réponse finale par défaut (README, Phase 3 : "Réponse finale
        # séparée du raisonnement") — utile pour le debug/l'évaluation sans
        # polluer ce qui est montré à l'utilisateur.
        if include_trace and trace.used_reasoning:
            response["trace"] = {
                "understanding": trace.plan.understanding if trace.plan else None,
                "steps": trace.plan.steps if trace.plan else [],
                "step_results": trace.resolution.step_results if trace.resolution else [],
                "verification_verdict": trace.verification.verdict if trace.verification else None,
            }

        return jsonify(response)

    return bp
