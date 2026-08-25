# Intégration RAG + Reasoning + Tools + Memory + Router dans Opsiom

Ce dossier ajoute les briques décrites dans le README qui manquaient encore :
**RAG** (section 7), **Reasoning** (section 6), **Outils** (section 9),
**Mémoire** (section 8) et **Router / Agent** (section 10), avec la
structure de fichiers annoncée en section 11.1 :

```
opsiom/
├── app.py                       # nouveau — point d'entrée Flask (assemble tout)
├── model/
│   └── llm_client.py          # interface commune vers le modèle
├── rag/
│   ├── embeddings.py
│   ├── database.py
│   ├── retriever.py
│   ├── reranker.py
│   └── pipeline.py
├── reasoning/
│   ├── prompts.py
│   ├── planner.py
│   ├── verifier.py
│   └── reasoner.py
├── tools/                      # nouveau — README section 9 (Phase 5)
│   ├── base.py                 # interface Tool / ToolResult
│   ├── calculator.py
│   ├── datetime_tool.py
│   ├── python_sandbox.py
│   ├── web_search.py
│   └── registry.py
├── scripts/
│   └── generate_reasoning_dataset.py  # nouveau — génération de dataset via un LLM enseignant
├── memory/                     # nouveau — README section 8 (Phase 6)
│   ├── short_term.py           # historique de conversation en mémoire
│   └── persistent.py           # souvenirs long-terme, SQLite + recherche sémantique
├── router/                     # nouveau — README section 10 (Phase 7)
│   └── router.py                # décide direct / reasoning / rag / outil
└── api/
    ├── rag.py                  # POST /api/rag
    ├── reason.py                # POST /api/reason
    ├── tools.py                 # GET /api/tools, POST /api/tools/execute
    ├── memory.py                 # POST/GET/DELETE /api/memory
    └── agent.py                  # POST /api/agent (point d'entrée unique du Router)
```

## 1. Pourquoi `model/llm_client.py` ?

`rag/`, `reasoning/`, `tools/`, `memory/` et `router/` n'importent jamais
directement `ModernLLM` ou `FrenchTokenizerWrapper`. Ils dépendent
uniquement de l'interface `LLMClient` (méthodes `generate()` / `chat()`),
avec deux implémentations :

- `LocalOpsiomClient` : charge le checkpoint en mémoire (utile dans l'API
  Flask principale).
- `APIOpsiomClient` : appelle `POST /api/chat` en HTTP (utile si les
  briques ci-dessus tournent dans un processus séparé du modèle).

Ça respecte la philosophie du README (section 2) : *"Cette séparation permet
de remplacer chaque composant indépendamment."*

### ⚠️ Incohérence connue entre `train_opsiom.py` et `main.py` — corrigée

Le dépôt fait cohabiter deux scripts autonomes (pensés pour tourner seuls
dans une cellule Colab/Kaggle) qui redéfinissent chacun localement
`ModernLLM` / `ModelArgs` / `FrenchTokenizerWrapper` :

| | `train_opsiom.py` | `main.py` |
|---|---|---|
| Constructeur du tokenizer | `FrenchTokenizerWrapper(hf_repo="...", local_path="...")` | `FrenchTokenizerWrapper(tokenizer_obj)` (objet `tokenizers.Tokenizer` déjà chargé) |
| Attribut de fin de séquence | `.eos_token_id` | `.eot_token` |
| Signature `encode()` | `encode(text)` | `encode(text, allowed_special="all")` |
| `ModelArgs.dim` par défaut | `384` (preset `Opsiom-Nano`) | `1024` (preset `Opsiom-Large`, mais peut différer selon la version du script) |

`ModernLLM.generate()` référence l'un ou l'autre nom d'attribut selon le
fichier d'origine — un `LocalOpsiomClient` écrit pour une seule des deux
interfaces plante donc systématiquement avec l'autre.

**Correctif appliqué dans cette livraison** (`model/llm_client.py`) :

- `_import_architecture_module()` cherche d'abord `model.architecture` (le
  module unifié recommandé par le README 11.1 — à créer si ce n'est pas
  déjà fait), puis retombe sur `train_opsiom`, puis sur `main`.
- `_build_tokenizer()` essaie l'appel par kwargs (`hf_repo=`, `local_path=`)
  et, s'il échoue avec un `TypeError`, charge un `tokenizers.Tokenizer` à la
  main et le passe en positionnel — donc compatible avec les deux
  constructeurs.
- `_normalize_tokenizer()` ajoute l'alias d'attribut manquant
  (`eos_token_id` ↔ `eot_token`) et rend `.encode()` tolérant à un
  éventuel kwarg `allowed_special` non supporté.
- `LocalOpsiomClient` ne s'appuie **jamais** sur les valeurs par défaut de
  `ModelArgs` (qui diffèrent, ex. `dim`) : il reconstruit systématiquement
  l'architecture depuis `ckpt["args"]`, l'objet réellement sauvegardé au
  moment de l'entraînement, et échoue explicitement si ce champ est absent
  plutôt que de risquer un mismatch de dimensions silencieux.

**Solution définitive recommandée** (pas encore faite dans ce dépôt) :
extraire une seule définition de `ModernLLM` / `ModelArgs` /
`FrenchTokenizerWrapper` dans `model/architecture.py`, importée à la fois
par `train_opsiom.py` et par `main.py`, pour supprimer la duplication à la
source plutôt que de la contourner à l'exécution.

## 2. `app.py` — assemble tout, prêt à lancer

Le câblage décrit ci-dessus est déjà fait dans `app.py` (nouveau) : il
charge le modèle, construit RAG/Reasoning/Tools/Memory/Router, enregistre
tous les blueprints, et expose `/api/chat`, `/api/reason`, `/api/rag` (si
configuré), `/api/tools`, `/api/tools/execute`, `/api/memory`, `/api/agent`
et `/api/health`.

```bash
export OPSIOM_CHECKPOINT_PATH=chat_model.pt
python app.py
```

Variables d'environnement disponibles (toutes optionnelles) : voir l'en-tête
de `app.py`. Sans `OPSIOM_RAG_INDEX_DIR` ni `OPSIOM_RAG_SOURCE_DIR`, le RAG
est simplement désactivé (`/api/rag` non enregistrée, le Router ne choisit
jamais la capacité `"rag"`) — le reste de l'API fonctionne normalement.

## 3. Construire l'index une seule fois, pas à chaque requête

```python
retriever = Retriever()
retriever.index_directory("learncode")
retriever.save("index_dir")
```

Puis, au démarrage de l'API :

```python
retriever = Retriever.load("index_dir")
```

## 4. Ce qui reste volontairement hors scope ici

- `Router.decide()` reste une heuristique locale par mots-clés/regex — pas
  un classifieur appris (README, feuille de route Phase 7 : à terme,
  remplaçable par un appel à Opsiom lui-même sans changer l'interface de
  `Router.answer()`).
- `tools/web_search.py` ne fournit aucun backend de recherche par défaut
  (pas de clé API en dur dans ce dépôt) — à brancher explicitement.
- `tools/python_sandbox.py` est un sandbox "best effort" en profondeur de
  défense (filtre AST + sous-processus + limites `resource`), **pas** une
  isolation de niveau production (pas de conteneur/cgroup dédié) — voir les
  commentaires en tête de ce fichier avant un déploiement exposé à des
  utilisateurs non fiables.
- L'intégration Omni/LearnCode/Classroom (README section 12, Phase 8) reste
  hors scope de cette livraison.

## 5. `scripts/generate_reasoning_dataset.py` — dataset de raisonnement via un LLM enseignant

Génère un fichier `dialogues_reasoning_fr.jsonl` (format `{"user":...,
"assistant":...}`, directement compatible avec `train_opsiom.py::Config.CHAT_LOCAL_JSONL`)
en interrogeant un modèle enseignant (Mistral-7B-Instruct-v0.3, ou un repli
non gated si aucun `HF_TOKEN` n'est configuré) chargé en 4-bit — tient sur un
T4 16 Go (Colab/Kaggle gratuits).

Le champ `"assistant"` embarque, pour une partie des exemples (contrôlée par
`Config.REASONING_FRACTION`), une trace de raisonnement entre deux tags
spéciaux avant la réponse finale :

```
<|reasoning|>
<étapes de raisonnement>
<|endreasoning|>
<réponse finale>
```

**Étape indispensable avant fine-tuning** : `<|reasoning|>` et
`<|endreasoning|>` doivent être ajoutés comme tokens spéciaux au tokenizer
BPE d'Opsiom (comme `<|endoftext|>` l'est déjà), via
`add_reasoning_tokens_to_tokenizer()` en fin du script — sinon le tokenizer
byte-level les découpe en fragments arbitraires et le modèle ne peut pas les
apprendre comme marqueurs structurels. Cela change `vocab_size`, donc la
couche d'embedding du modèle de base doit être redimensionnée avant de
charger un ancien checkpoint dessus (voir le docstring de la fonction pour
le détail).

Le script sauvegarde en continu (une ligne JSONL par exemple, flush
régulier) et reprend automatiquement où il s'est arrêté si relancé après une
coupure de session Colab/Kaggle (les questions déjà présentes dans le
fichier de sortie sont ignorées).

## 6. Repli sans dépendances lourdes

`sentence-transformers` et `faiss-cpu` sont recommandés (voir
`requirements-rag-reasoning.txt`) mais pas strictement obligatoires :
`rag/embeddings.py` et `rag/reranker.py` détectent leur absence et basculent
sur un repli fonctionnel (hashing/TF-IDF + score lexical), avec un
avertissement au démarrage. `faiss-cpu` reste en revanche requis pour
`rag/database.py` (pas de repli prévu pour l'index vectoriel lui-même).

`tools/`, `memory/` et `router/` n'introduisent aucune dépendance
supplémentaire au-delà de ce qui est déjà listé (voir le bas de
`requirements-rag-reasoning.txt` pour le détail : `resource`, `zoneinfo`,
`sqlite3` sont tous dans la bibliothèque standard).
