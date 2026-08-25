# Opsiom

Un modèle de langage (LLM) français construit **de zéro** — tokenizer, architecture, pré-entraînement, fine-tuning chat, jusqu'à l'infrastructure d'entraînement sur Colab/Kaggle. Aucune brique n'est empruntée à un modèle pré-entraîné existant : tout est entraîné depuis des poids aléatoires sur du texte français.

## Sommaire

- [Vue d'ensemble](#vue-densemble)
- [Architecture](#architecture)
- [Tailles de modèle](#tailles-de-modèle)
- [Tokenizer](#tokenizer)
- [Pipeline de données](#pipeline-de-données)
- [Entraînement](#entraînement)
- [Infrastructure Colab / Kaggle](#infrastructure-colab--kaggle)
- [Résultats actuels](#résultats-actuels)
- [Limites connues](#limites-connues)
- [Extensions expérimentales](#extensions-expérimentales)
- [Structure du projet](#structure-du-projet)
- [Reproduire un entraînement](#reproduire-un-entraînement)
- [Feuille de route](#feuille-de-route)

---

## Vue d'ensemble

Opsiom est un projet éducatif de LLM autorégressif en français, dans l'esprit LLaMA/Qwen (RoPE, GQA, SwiGLU, KV-cache), pensé pour tourner entièrement sur des GPU gratuits (Colab T4, Kaggle T4×2). L'objectif n'est pas de rivaliser avec les modèles commerciaux, mais de comprendre et maîtriser chaque étage du pipeline : tokenisation, pré-entraînement, fine-tuning d'instructions, et — en cours d'exploration — RAG et raisonnement.

Deux phases d'entraînement distinctes :

1. **Base** (`main.py`) — pré-entraînement next-token sur du texte brut français (Wikipédia, FineWeb-2, TinyStories-French).
2. **Chat** (`train_chat.py`) — fine-tuning du modèle base sur des paires question/réponse pour le rendre utilisable en dialogue.

## Architecture

Transformer décodeur autorégressif, avec :

| Composant | Choix |
|---|---|
| Normalisation | RMSNorm |
| Position | RoPE (Rotary Position Embeddings) |
| Attention | GQA (Grouped-Query Attention) via `scaled_dot_product_attention` (FlashAttention) |
| MLP | SwiGLU |
| Poids | `tok_embeddings` et `lm_head` liés (weight tying) |
| Génération | KV-cache, température / top-k / top-p / pénalité de répétition |

Cette architecture est générique et paramétrée entièrement par `ModelArgs` (`dim`, `n_layers`, `n_heads`, `n_kv_heads`, `max_seq_len`) — changer de taille de modèle ne demande de toucher qu'à cette dataclass.

## Tailles de modèle

| Preset | dim | layers | heads | kv_heads | max_seq_len | Paramètres | Statut |
|---|---|---|---|---|---|---|---|
| Opsiom-Micro | 256 | 6 | 4 | 2 | 192 | ~5M | non testé en profondeur |
| Opsiom-Nano | 384 | 8 | 8 | 4 | 256 | ~15-26M | testé, très sous-entraîné (~5-6M tokens vus, besoin réel ~515M) |
| Opsiom-Small | 576 | 10 | 9 | 3 | 320 | ~45M | testé, amélioration partielle sur Nano |
| Opsiom-Medium | 768 | 12 | 12 | 4 | 384 | ~95M | testé, également sous-entraîné (corpus insuffisant pour cette taille) |
| Opsiom-Large | 1024 | 16 | 16 | 4 | 512 | ~196,77M | entraînement Chinchilla-compliant en cours (voir ci-dessous) |

Règle de dimensionnement utilisée : **Chinchilla** (~20 tokens d'entraînement par paramètre). Micro à Medium ont tous été entraînés avec un corpus très inférieur à ce ratio — c'est la cause principale de leurs limites (voir [Résultats actuels](#résultats-actuels)). Opsiom-Large est le premier modèle du projet dont le corpus de pré-entraînement respecte ce ratio :

```
196,77M paramètres × 20 ≈ 3,94 milliards de tokens visés
Corpus réellement construit : 3 935 400 284 tokens (~7,8 Go de texte)
```

⚠️ Respecter Chinchilla optimise la perte de pré-entraînement du modèle **base** — ça ne garantit pas à soi seul un bon comportement conversationnel, qui dépend séparément de la qualité et du volume du fine-tuning chat (voir plus bas).

## Tokenizer

BPE byte-level (façon GPT-2, donc jamais de token inconnu/OOV), vocabulaire de 16 000 tokens, entraîné sur un échantillon de paragraphes Wikipédia français. Volontairement compact comparé au vocabulaire GPT-2 original (50 257 tokens) : un vocabulaire français dédié plus petit libère du budget de paramètres pour les couches Transformer plutôt que pour la table d'embeddings.

Token spécial : `<|endoftext|>` (séparateur de documents). Le tokenizer est entraîné une seule fois puis mis en cache (`fr_bpe_tokenizer.json`) — les runs suivants le rechargent directement.

## Pipeline de données

### Mode base (pré-entraînement)

Corpus construit en mélangeant plusieurs sources pondérées, streamées (aucun téléchargement complet préalable), jusqu'à une cible de tokens fixée par le ratio Chinchilla :

| Source | Poids visé | Rôle |
|---|---|---|
| TinyStories-French (`iproskurina/TinyStories-French`) | intégré une fois en amont | Style narratif simple, adapté à un petit modèle |
| Wikipédia FR (`wikimedia/wikipedia`, config `20231101.fr`) | 15% | Registre encyclopédique |
| FineWeb-2 FR (`HuggingFaceFW/fineweb-2`, config `fra_Latn`) | 70% | Web filtré/dédupliqué, gros volume |
| OSCAR-2301 FR (`oscar-corpus/OSCAR-2301`) | 15%, opportuniste | Diversité supplémentaire — nécessite un `HF_TOKEN` ayant accepté les conditions d'accès ; ignoré proprement sinon |

⚠️ **Répartition réelle observée sur le dernier run** : OSCAR-2301 n'a pas pu être chargé (accès non configuré), donc la répartition finale a été ~26% Wikipédia / ~74% FineWeb-2 plutôt que 15/70/15 — les poids indiquent une intention, pas une garantie (chaque source manquante voit son poids redistribué implicitement vers les sources encore actives).

Un filtre anti-LaTeX nettoie les paragraphes Wikipédia contenant des formules brutes non rendues (`\frac{}{}`, `$...$`, etc.), qui sinon polluent les statistiques de sous-mots du tokenizer et produisent des artefacts dans la génération.

### Mode chat (fine-tuning instructions)

| Source | Type | Volume utilisé |
|---|---|---|
| `jpacifico/French-Alpaca-dataset-Instruct-55K` | Instructions courtes (Alpaca) | jusqu'à ~55K, filtré |
| `tbboukhari/Alpaca-in-french` | Instructions courtes (traduction indépendante d'Alpaca) | jusqu'à ~52K, filtré |
| `OpenAssistant/oasst1` (sous-ensemble `lang="fr"`) | Dialogue humain réel, reconstruit depuis l'arbre de conversation | ~800-2000 paires (c'est tout ce qui existe en français dans ce dataset) |
| Corpus narratif local (`dialogues_fr.jsonl`) | Prose originale, reformattée en paires *(prompt narratif → texte)* | variable, généré via `build_story_jsonl_from_corpus()` |

**Point important** : les deux sources Alpaca sont deux traductions indépendantes du **même** jeu source (Stanford Alpaca) — les combiner apporte de la robustesse de formulation, pas une réelle expansion de couverture factuelle. La vraie diversité vient d'`oasst1` (dialogue naturel) et du corpus narratif local (seule source contenant réellement des histoires/contes — les jeux Alpaca en sont quasiment dépourvus).

## Entraînement

### Hyperparamètres (mode base)

- Optimiseur : AdamW (β=(0.9, 0.95), weight decay 0.1 sur les poids ≥2D, 0 sur le reste)
- Planning LR : warmup linéaire puis cosine decay, `MAX_LR=3e-4`, `MIN_LR=3e-5`
- Mixed precision : bf16 (fallback fp16 si le GPU ne supporte pas bf16, ex. T4)
- Grad clipping : 1.0

**Reprise (`RESUME_FROM_CHECKPOINT`)** : deux garde-fous ajoutés après des régressions observées en pratique —
1. En reprise, le pic de LR et le warmup sont automatiquement réduits (`RESUME_LR_SCALE`) plutôt que de repartir sur le planning complet d'un entraînement from-scratch, qui « éjecte » un modèle déjà convergé de son minimum.
2. Si le checkpoint trouvé ne correspond pas à l'architecture `ModelArgs` actuelle (ex. changement de taille de modèle sans renommer l'ancien fichier), il est **ignoré avec un avertissement explicite** plutôt que de faire planter le run — utile en particulier en multi-GPU, où un crash sur un rang tue tout le job.

### Hyperparamètres (mode chat)

- LR constant (`CHAT_LR`, pas de warmup/cosine — contrairement au mode base, donc moins exposé au même risque de « choc » en reprise)
- Nombre d'epochs et taille de batch ajustés au volume réel du corpus combiné

### Éval — `eval_prompts.py`

Harnais d'évaluation qualitative : rejoue un jeu de prompts **fixe** (facile/factuel, définitions courtes, complétion narrative, instruction narrative, instructions courtes), avec une **graine fixée**, pour comparer deux checkpoints de façon reproductible plutôt que de juger « à l'oreille » sur des sorties non contrôlées. Sauvegarde chaque run dans `eval_results/` avec les métadonnées du checkpoint (step, val_loss, mtime) et avertit explicitement si deux évaluations portent sur le même checkpoint (donc que toute différence observée n'est que du bruit d'échantillonnage).

## Infrastructure Colab / Kaggle

- Détection automatique de l'environnement (Colab → Google Drive, Kaggle → `/kaggle/working`, sinon local).
- **Corpus base en fichier binaire memmap** (`pretrain_corpus.bin`) plutôt qu'en RAM — indispensable au-delà de quelques centaines de millions de tokens (un tenseur de 3,9 milliards de tokens en int64 pèserait ~31 Go).
- **Échantillonnage aléatoire avec remise** (`RandomWindowIterableDataset`) plutôt qu'un vrai shuffle : `torch.randperm` sur un dataset de plusieurs milliards de fenêtres alloue des dizaines de Go rien que pour les indices, ce qui a provoqué un OOM-kill en pratique.
- **Multi-GPU** (Kaggle T4×2) via `torch.multiprocessing.spawn` + `accelerate` (DDP), avec `NCCL_P2P_DISABLE=1` (les instances Kaggle T4×2 n'ont pas de lien P2P/NVLink). La préparation des données (téléchargement + tokenisation) se fait **avant** le lancement des processus GPU, pas pendant — cette phase est 100% réseau/CPU et faisait courir un risque de pression mémoire inutile avec deux contextes CUDA déjà actifs.
- **Sauvegarde externe automatique (Hugging Face Hub)** : `best_model.pt` et le tokenizer sont poussés vers un repo privé HF dès qu'ils s'améliorent localement, plus un filet de sécurité toutes les 10 minutes indépendant des améliorations — `/kaggle/working` n'étant pas persistant sans commit explicite, c'est la seule garantie de ne pas perdre un run en cas de crash, coupure, ou fermeture de session.

## Résultats actuels

### Val loss par taille de modèle

| Modèle | val_loss | Contexte |
|---|---|---|
| Opsiom-Nano | ~2.85 – 3.09 | Corpus base très réduit (~5-6M tokens) |
| Opsiom-Small | ~2.09 | Amélioration nette sur Nano, mais toujours sous-entraîné au ratio Chinchilla |
| Opsiom-Medium | — | Sous-entraîné, pas de gain net confirmé vs Small |
| Opsiom-Large | *(en cours)* | Corpus Chinchilla-compliant construit ; premier run bloqué par un OOM CUDA (`BATCH_SIZE` réduit depuis) |

### Exemples qualitatifs (Opsiom-Small, dim=576)

| Prompt | Réponse | Verdict |
|---|---|---|
| *Quelle est la capitale de la France ?* | *La capitale de la France est Paris.* | ✅ correct |
| *Quel est le plus grand océan du monde ?* | *Le plus grand océan du monde est l'océan Pacifique.* | ✅ correct |
| *Qu'est-ce que la photosynthèse ?* | *La photosynthèse est le processus par lequel les plantes convertissent la lumière en énergie chimique...* | ✅ correct et cohérent |
| *Explique ce qu'est un volcan.* | *Un oncle est un type de mélange d'air...* | ❌ incohérent |
| *Raconte-moi une histoire.* | *Alice, la tête est un cadeau. Elle se rend chez elle...* | ❌ incohérent |
| *Écris-moi un petit conte.* | *Je ne peux pas prendre de risques.* | ⚠️ probable artefact d'`oasst1` (réponse façon refus d'assistant) |

Schéma général observé sur toutes les tailles testées : **bon sur les faits très fréquents et peu ambigus** (mémorisés quasi littéralement depuis le fine-tuning), **faible sur tout le reste** — faits moins fréquents, définitions moins canoniques, et surtout tout ce qui demande de la génération libre (récit, poésie).

## Limites connues

- **Fenêtre de contexte réduite** (256 à 512 tokens selon la taille) — contraint fortement tout ce qui nécessite d'injecter du contexte (RAG, historique de conversation long, exemples few-shot).
- **Pas de compréhension du system prompt** — le modèle n'a jamais été entraîné avec un rôle "system" distinct ; toute instruction de persona/style placée hors du tour utilisateur est traitée comme du texte ordinaire, sans poids conditionnant particulier.
- **Pas de capacité de raisonnement multi-étapes démontrée** — les échecs sur des faits simples en zero-shot suggèrent qu'un raisonnement explicite (décomposition, auto-vérification) n'est pas encore un socle fiable ; voir [Extensions expérimentales](#extensions-expérimentales).
- **Sensibilité aux artefacts de sur-apprentissage** — motifs répétés dans les données de fine-tuning (ex. templates trop uniformes) peuvent produire un écho verbatim plutôt qu'une généralisation.
- **Écarts entre scripts** — `train_opsiom.py` et `main.py` redéfinissent chacun localement `ModernLLM`/`ModelArgs`/`FrenchTokenizerWrapper` avec des interfaces légèrement différentes (nom de l'attribut de fin de séquence, signature de `.encode()`, valeurs par défaut). Une factorisation dans un module partagé `model/architecture.py` est recommandée mais pas encore faite.

## Extensions expérimentales

Un module plus ambitieux (RAG, raisonnement, outils, mémoire, routeur/agent) existe en périphérie du projet mais reste **non validé en pratique** à ce stade :

| Brique | Statut |
|---|---|
| RAG (`rag/`) | Conçu proprement (retriever + reranker + repli sans dépendances lourdes), mais **sans budget de tokens géré** avant l'appel au modèle — risque réel de troncature silencieuse du contexte récupéré vu la fenêtre de 256-512 tokens |
| Raisonnement (`reasoning/`) | Planner/Verifier zero-shot à l'inférence — repose sur une capacité de décomposition/auto-vérification non démontrée pour ce modèle ; le Verifier utilise le même modèle faible, donc peut tout aussi bien dégrader une bonne réponse |
| Génération de dataset de raisonnement (`scripts/generate_reasoning_dataset.py`) | Approche la plus prometteuse du lot : distillation de traces de raisonnement depuis un modèle enseignant (Mistral-7B) pour fine-tuning, plutôt que du prompting zero-shot |
| Outils (`tools/`) | Sandbox Python en défense-en-profondeur (filtre AST + sous-processus + limites `resource`), explicitement documenté comme "best effort", pas une isolation de production |
| Routeur / Agent (`router/`) | Heuristiques par mots-clés, assumées comme un pis-aller en attendant un classifieur appris |

**Priorité recommandée avant d'activer ces briques en production** : RAG avec un vrai budget de contexte, puis fine-tuning sur les traces distillées (mesuré via `eval_prompts.py`) — le reste (tools/router/memory complets) apporte de la complexité sans gain démontré tant que la qualité de base du chat n'est pas plus solide.

## Structure du projet

```
mini_llm_fr/
├── main.py                    # Pré-entraînement "base", à grande échelle (Colab/Kaggle, multi-GPU)
├── train_chat.py               # Fine-tuning "chat" (instructions), depuis un checkpoint base
├── test_IA.py                   # Moteur d'inférence interactif (streaming, historique multi-tours)
├── eval_prompts.py              # Harnais d'évaluation qualitative reproductible
├── fr_bpe_tokenizer.json        # Tokenizer BPE entraîné (mis en cache)
├── best_model.pt                # Checkpoint du modèle base
├── chat_model.pt                # Checkpoint fine-tuné chat
├── dialogues_fr.jsonl            # Corpus narratif local additionnel
└── opsiom/                      # Extensions expérimentales (RAG, reasoning, tools, memory, router)
```

## Reproduire un entraînement

```python
# 1. Pré-entraînement base (Colab ou Kaggle, cellule unique)
%run main.py

# 2. Fine-tuning chat, à partir du checkpoint base obtenu
%run train_chat.py    # mode "chat" dans Config

# 3. Test interactif
%run test_IA.py

# 4. Évaluation reproductible (avant/après comparaison)
%run eval_prompts.py
```

Sur Kaggle, privilégier **"Save Version" → "Save & Run All (Commit)"** plutôt que l'édition interactive pour les runs longs : ce mode s'exécute sur les serveurs Kaggle indépendamment de la session navigateur locale.

## Feuille de route

- [ ] Terminer et évaluer un run complet d'Opsiom-Large (Chinchilla-compliant)
- [ ] Factoriser `ModernLLM`/`ModelArgs`/`FrenchTokenizerWrapper` dans `model/architecture.py` partagé
- [ ] Ajouter un budget de tokens explicite au pipeline RAG avant toute mise en production
- [ ] Générer un dataset de raisonnement distillé et mesurer son effet réel via `eval_prompts.py`
- [ ] Explorer une source de diversité factuelle au-delà des traductions d'Alpaca (ex. dialogue humain à plus grande échelle)
