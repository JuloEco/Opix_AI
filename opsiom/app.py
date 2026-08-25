# ============================================================================
# app.py — Point d'entrée de l'API Flask Opsiom (README section 11 et 18)
#
#   Question -> API Flask -> Router -> Reasoning / RAG / Tools -> Opsiom
#            -> Vérification -> Réponse
#
# Assemble tous les modules livrés dans INTEGRATION.md : model/llm_client.py,
# rag/, reasoning/, tools/, memory/, router/, et les blueprints de api/.
#
# Variables d'environnement (toutes optionnelles, valeurs par défaut ci-dessous) :
#   OPSIOM_CHECKPOINT_PATH   chemin du checkpoint chat (défaut: chat_model.pt)
#   OPSIOM_TOKENIZER_HF_REPO repo HF du tokenizer (défaut: JuloEco/opsiom-fr-tokenizer)
#   OPSIOM_TOKENIZER_LOCAL   chemin local de repli du tokenizer
#   OPSIOM_RAG_INDEX_DIR     dossier d'un index RAG déjà construit (sauvegardé
#                            via Retriever.save(), voir INTEGRATION.md section 3)
#   OPSIOM_RAG_SOURCE_DIR    dossier de documents à indexer si aucun index n'existe
#   OPSIOM_MEMORY_DB_PATH    chemin de la base SQLite mémoire (défaut: opsiom_memory.sqlite3)
#   PORT                     port d'écoute (défaut: 5000)
#
# Démarrage :
#   python app.py
# ============================================================================

from __future__ import annotations

import os
import sys

from flask import Flask, jsonify, request

# -- Modèle ------------------------------------------------------------------
from model.llm_client import LocalOpsiomClient

# -- RAG -----------------------------------------------------------------
from rag import Retriever, RAGPipeline

# -- Reasoning -------------------------------------------------------------
from reasoning import Reasoner

# -- Tools (README section 9) -----------------------------------------------
from tools import ToolRegistry

# -- Memory (README section 8) ----------------------------------------------
from memory import ConversationMemory, PersistentMemory

# -- Router (README section 10 et 18) ----------------------------------------
from router import Router

# -- Blueprints API ----------------------------------------------------------
from api.rag import build_rag_blueprint
from api.reason import build_reasoning_blueprint
from api.tools import build_tools_blueprint
from api.memory import build_memory_blueprint
from api.agent import build_agent_blueprint


CHECKPOINT_PATH = os.environ.get("OPSIOM_CHECKPOINT_PATH", "chat_model.pt")
TOKENIZER_HF_REPO = os.environ.get("OPSIOM_TOKENIZER_HF_REPO", "JuloEco/opsiom-fr-tokenizer")
TOKENIZER_LOCAL_PATH = os.environ.get("OPSIOM_TOKENIZER_LOCAL", "fr_bpe_tokenizer.json")
RAG_INDEX_DIR = os.environ.get("OPSIOM_RAG_INDEX_DIR", "").strip()
RAG_SOURCE_DIR = os.environ.get("OPSIOM_RAG_SOURCE_DIR", "").strip()
MEMORY_DB_PATH = os.environ.get("OPSIOM_MEMORY_DB_PATH", "opsiom_memory.sqlite3")
PORT = int(os.environ.get("PORT", "5000"))


def _load_llm_client() -> LocalOpsiomClient:
    if not os.path.exists(CHECKPOINT_PATH):
        print(
            f"❌ Checkpoint introuvable : '{CHECKPOINT_PATH}'.\n"
            "   -> Entraîne d'abord un modèle (train_opsiom.py / main.py), ou "
            "règle OPSIOM_CHECKPOINT_PATH vers un checkpoint existant "
            "(idéalement 'chat_model.pt', produit par TRAINING_MODE='chat')."
        )
        sys.exit(1)

    print(f"📦 Chargement du modèle depuis '{CHECKPOINT_PATH}'...")
    try:
        client = LocalOpsiomClient(
            checkpoint_path=CHECKPOINT_PATH,
            tokenizer_hf_repo=TOKENIZER_HF_REPO,
            tokenizer_local_path=TOKENIZER_LOCAL_PATH,
        )
    except Exception as e:
        # model/llm_client.py gère déjà la compatibilité train_opsiom.py /
        # main.py (voir INTEGRATION.md) — une erreur ici est donc probablement
        # un vrai problème de checkpoint/tokenizer, pas l'incohérence connue.
        print(f"❌ Échec du chargement du modèle : {e}")
        sys.exit(1)
    return client


def _build_rag_pipeline(llm_client) -> RAGPipeline | None:
    """Construit le pipeline RAG si un index existant ou un dossier source a
    été configuré. Renvoie None sinon (le Router reste utilisable sans RAG,
    voir router/router.py : capability 'rag' n'est jamais choisie dans ce cas)."""
    if RAG_INDEX_DIR and os.path.isdir(RAG_INDEX_DIR):
        print(f"📂 Chargement de l'index RAG existant depuis '{RAG_INDEX_DIR}'...")
        retriever = Retriever.load(RAG_INDEX_DIR)
    elif RAG_SOURCE_DIR and os.path.isdir(RAG_SOURCE_DIR):
        print(f"🏗️ Indexation du dossier '{RAG_SOURCE_DIR}' (RAG)...")
        retriever = Retriever()
        retriever.index_directory(RAG_SOURCE_DIR)
        if RAG_INDEX_DIR:
            retriever.save(RAG_INDEX_DIR)
    else:
        print("ℹ️ Aucun index RAG configuré (OPSIOM_RAG_INDEX_DIR / OPSIOM_RAG_SOURCE_DIR "
              "absents ou introuvables) — RAG désactivé pour cette session.")
        return None

    return RAGPipeline(retriever, llm_client=llm_client)


def create_app() -> Flask:
    app = Flask(__name__)

    llm_client = _load_llm_client()
    rag_pipeline = _build_rag_pipeline(llm_client)
    reasoner = Reasoner(llm_client)
    tool_registry = ToolRegistry()  # calculator, datetime, python, web_search (README section 9)
    conversation_memory = ConversationMemory()
    persistent_memory = PersistentMemory(db_path=MEMORY_DB_PATH)

    router = Router(
        llm_client,
        reasoner=reasoner,
        rag_pipeline=rag_pipeline,
        tool_registry=tool_registry,
        conversation_memory=conversation_memory,
        persistent_memory=persistent_memory,
    )

    # -- Route directe minimale (README section 11), utile pour un appel
    # simple sans passer par le Router, et pour APIOpsiomClient (voir
    # model/llm_client.py) si RAG/Reasoning tournent un jour dans un
    # processus séparé qui appelle cette API en HTTP. --------------------
    @app.post("/api/chat")
    def chat_endpoint():
        data = request.get_json(silent=True) or {}
        message = (data.get("message") or "").strip()
        if not message:
            return jsonify({"error": "Le champ 'message' est requis."}), 400
        if len(message) > 2000:
            return jsonify({"error": "Message trop long (2000 caractères max)."}), 413

        raw_prompt = bool(data.get("raw_prompt", False))
        gen_kwargs = dict(
            max_new_tokens=data.get("max_new_tokens", 200),
            temperature=data.get("temperature", 0.7),
            top_k=data.get("top_k", 40),
            top_p=data.get("top_p", 0.9),
            repetition_penalty=data.get("repetition_penalty", 1.3),
        )
        try:
            if raw_prompt:
                # Le prompt est déjà entièrement formaté par l'appelant
                # (tags inclus) — ne pas le ré-emballer (voir APIOpsiomClient).
                raw = llm_client.generate(message, **gen_kwargs)
                response = llm_client._strip_prompt(raw, message)
            else:
                response = llm_client.chat(message, **gen_kwargs)
        except Exception as e:  # pragma: no cover - garde-fou API
            return jsonify({"error": f"Erreur de génération : {e}"}), 500

        return jsonify({"response": response})

    @app.get("/api/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "checkpoint": CHECKPOINT_PATH,
                "rag_enabled": rag_pipeline is not None,
            }
        )

    app.register_blueprint(build_rag_blueprint(rag_pipeline)) if rag_pipeline else None
    app.register_blueprint(build_reasoning_blueprint(reasoner))
    app.register_blueprint(build_tools_blueprint(tool_registry))
    app.register_blueprint(build_memory_blueprint(persistent_memory))
    app.register_blueprint(build_agent_blueprint(router))

    print("✅ Opsiom API prête.")
    print("   Routes : /api/chat  /api/reason  /api/tools  /api/tools/execute")
    print("            /api/memory  /api/agent  /api/health"
          + ("  /api/rag" if rag_pipeline else " (RAG désactivé)"))

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=PORT)
