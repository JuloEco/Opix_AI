# ============================================================================
# api/memory.py — Endpoints mémoire persistante (README section 11)
#
#   POST   /api/memory                -> mémoriser un fait
#   GET    /api/memory?user_id=...    -> lister les souvenirs
#   GET    /api/memory/search?...     -> recherche sémantique
#   DELETE /api/memory/<id>           -> oublier un souvenir
#
# Le user_id identifie l'utilisateur (compte Omni, cf. README section 12) —
# jamais déduit implicitement, toujours passé explicitement par l'appelant.
# ============================================================================

from __future__ import annotations

from flask import Blueprint, jsonify, request


def build_memory_blueprint(persistent_memory) -> Blueprint:
    bp = Blueprint("memory", __name__)

    def _require_user_id(source) -> tuple[str | None, tuple | None]:
        user_id = (source.get("user_id") or "").strip()
        if not user_id:
            return None, (jsonify({"error": "Le champ 'user_id' est requis."}), 400)
        return user_id, None

    @bp.post("/api/memory")
    def remember():
        data = request.get_json(silent=True) or {}
        user_id, err = _require_user_id(data)
        if err:
            return err
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "Le champ 'text' est requis."}), 400
        if len(text) > 2000:
            return jsonify({"error": "Souvenir trop long (2000 caractères max)."}), 413

        try:
            memory = persistent_memory.remember(user_id, text, category=data.get("category"))
        except Exception as e:  # pragma: no cover - garde-fou
            return jsonify({"error": f"Erreur mémoire : {e}"}), 500
        return jsonify(memory.to_dict()), 201

    @bp.get("/api/memory")
    def list_memories():
        user_id, err = _require_user_id(request.args)
        if err:
            return err
        memories = persistent_memory.list_all(user_id)
        return jsonify({"memories": [m.to_dict() for m in memories]})

    @bp.get("/api/memory/search")
    def search_memories():
        user_id, err = _require_user_id(request.args)
        if err:
            return err
        query = (request.args.get("q") or "").strip()
        if not query:
            return jsonify({"error": "Le paramètre 'q' est requis."}), 400
        top_k = int(request.args.get("top_k", 5))

        results = persistent_memory.search(user_id, query, top_k=top_k)
        return jsonify({"results": [{"memory": m.to_dict(), "score": score} for m, score in results]})

    @bp.delete("/api/memory/<int:memory_id>")
    def forget(memory_id: int):
        user_id, err = _require_user_id(request.args)
        if err:
            return err
        deleted = persistent_memory.forget(user_id, memory_id)
        if not deleted:
            return jsonify({"error": "Souvenir introuvable."}), 404
        return jsonify({"deleted": True})

    return bp
