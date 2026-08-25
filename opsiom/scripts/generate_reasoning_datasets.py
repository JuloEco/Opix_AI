# ============================================================================
# scripts/generate_reasoning_dataset.py — Générateur de dataset conversationnel
# avec traces de raisonnement, via un LLM "enseignant" (Mistral-7B ou
# équivalent), prêt pour une cellule Google Colab ou Kaggle.
#
# Objectif (README section 6, encadré "Important") :
#
#   "Augmenter simplement max_new_tokens ne suffit pas. Le modèle doit
#    apprendre à utiliser les tokens supplémentaires pour effectuer une
#    résolution structurée."
#
# Ce script génère des exemples (question, réponse) où la réponse contient
# une trace de raisonnement explicite entre deux tags spéciaux, suivie de la
# réponse finale :
#
#   <|reasoning|>
#   <étapes de raisonnement>
#   <|endreasoning|>
#   <réponse finale>
#
# Le fichier produit est un JSONL directement compatible avec le format
# attendu par train_opsiom.py (Config.CHAT_LOCAL_JSONL) : une ligne par
# exemple, {"user": "...", "assistant": "..."} — le champ "assistant"
# contient déjà les tags <|reasoning|>/<|endreasoning|>, donc AUCUNE
# modification de train_opsiom.py n'est nécessaire pour les ingérer : il
# suffit de pointer Config.CHAT_LOCAL_JSONL vers le fichier produit ici (ou
# de le fusionner avec un dialogues_fr.jsonl existant).
#
# ⚠️ Pour que le modèle apprenne VRAIMENT à s'arrêter après la réponse
# finale, <|reasoning|> et <|endreasoning|> doivent être ajoutés comme
# tokens spéciaux au tokenizer BPE d'Opsiom (comme <|endoftext|> l'est déjà,
# voir train_opsiom.py::FrenchTokenizerWrapper) AVANT de fine-tuner dessus,
# sans quoi le tokenizer byte-level les découpera en fragments arbitraires.
# Voir la fonction `add_reasoning_tokens_to_tokenizer()` en fin de fichier.
#
# ----------------------------------------------------------------------------
# UTILISATION SUR COLAB / KAGGLE
# ----------------------------------------------------------------------------
#   1. Colle ce fichier dans une cellule (ou %%writefile puis %run).
#   2. Ajuste la section CONFIGURATION ci-dessous (modèle enseignant, sujets,
#      nombre d'exemples par sujet).
#   3. Si tu utilises un modèle gated (ex: mistralai/Mistral-7B-Instruct-v0.3),
#      exporte HF_TOKEN (secret Colab/Kaggle) et accepte les conditions sur
#      la page du modèle. Sinon, le script bascule automatiquement sur un
#      modèle Mistral-based NON gated (voir FALLBACK_MODEL_NAME).
#   4. Exécute. Le script installe ses dépendances, charge le modèle
#      enseignant en 4-bit, génère, et sauvegarde en continu (reprise
#      possible en cas de coupure de session — les questions déjà générées
#      sont ignorées au relancement).
# ============================================================================

import subprocess
import sys


def _pip_install(packages: list[str]) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=False)


_pip_install(["transformers>=4.42", "accelerate", "bitsandbytes", "sentencepiece", "huggingface_hub"])

import gc
import json
import os
import random
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

class Config:
    # --- Modèle enseignant ---
    # Gated (nécessite HF_TOKEN + acceptation des conditions sur huggingface.co) :
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
    # Repli automatique si MODEL_NAME est inaccessible (pas de HF_TOKEN, accès
    # refusé, etc.) — Mistral-based, NON gated, qualité comparable en instruct :
    FALLBACK_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
    LOAD_IN_4BIT = True  # nécessaire pour tenir sur un T4 16 Go — laisser à True

    # --- Environnement de sauvegarde (même logique que train_opsiom.py) ---
    USE_GOOGLE_DRIVE = False
    DRIVE_MOUNT_POINT = "/content/drive"
    PROJECT_DIR = "/kaggle/working"
    OUTPUT_JSONL_FILENAME = "dialogues_reasoning_fr.jsonl"  # -> Config.CHAT_LOCAL_JSONL côté train_opsiom.py

    # --- Tags de raisonnement (doivent être ajoutés au tokenizer Opsiom avant
    # fine-tuning, voir add_reasoning_tokens_to_tokenizer() en fin de fichier) ---
    REASONING_START_TAG = "<|reasoning|>"
    REASONING_END_TAG = "<|endreasoning|>"

    # --- Sujets couverts (librement extensible) ---
    TOPICS = [
        "mathématiques du quotidien (pourcentages, proportions, unités)",
        "logique et énigmes simples",
        "histoire de France",
        "géographie mondiale",
        "sciences : physique de base",
        "sciences : biologie et corps humain",
        "programmation en Python",
        "grammaire et orthographe française",
        "cuisine et recettes",
        "économie du quotidien (budget, épargne, intérêts)",
        "actualité générale et société (sujets non datés)",
        "philosophie et éthique",
        "santé et bien-être (conseils généraux, non médicaux)",
        "technologie et informatique",
        "environnement et écologie",
        "voyage et culture générale",
        "sport et activité physique",
        "littérature et écriture",
        "psychologie du quotidien",
        "droit et citoyenneté (notions générales)",
    ]
    QUESTIONS_PER_TOPIC = 25       # nombre de questions générées par sujet
    REASONING_FRACTION = 0.6       # proportion de questions forcées "multi-étapes"
    MAX_NEW_TOKENS_QUESTIONS = 700
    MAX_NEW_TOKENS_ANSWER = 500
    TEMPERATURE_QUESTIONS = 0.9
    TEMPERATURE_ANSWER = 0.4
    SAVE_EVERY = 5                  # flush disque tous les N exemples
    SEED = 1337


# ============================================================================
# 2. Environnement (Drive / Kaggle / local) — repris de train_opsiom.py
# ============================================================================

def _resolve_project_dir(cfg: "Config") -> str:
    if cfg.USE_GOOGLE_DRIVE:
        try:
            from google.colab import drive  # type: ignore

            print(f"💾 Montage de Google Drive sur '{cfg.DRIVE_MOUNT_POINT}'...")
            drive.mount(cfg.DRIVE_MOUNT_POINT)
            os.makedirs(cfg.PROJECT_DIR, exist_ok=True)
            return cfg.PROJECT_DIR
        except ImportError:
            print("ℹ️ Pas dans Colab — Drive ignoré.")
    if os.path.exists("/kaggle/working"):
        os.makedirs(cfg.PROJECT_DIR, exist_ok=True)
        print(f"ℹ️ Environnement Kaggle détecté — sauvegarde dans '{cfg.PROJECT_DIR}'.")
        return cfg.PROJECT_DIR
    print("ℹ️ Ni Colab ni Kaggle détecté — sauvegarde dans le répertoire courant.")
    return "."


# ============================================================================
# 3. Chargement du modèle enseignant (4-bit, avec repli non-gated)
# ============================================================================

def load_teacher_model(cfg: "Config"):
    hf_token = os.environ.get("HF_TOKEN")

    quant_config = None
    if cfg.LOAD_IN_4BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    def _try_load(name: str):
        print(f"📥 Chargement du modèle enseignant '{name}' (4-bit={cfg.LOAD_IN_4BIT})...")
        tok = AutoTokenizer.from_pretrained(name, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            name,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            token=hf_token,
        )
        model.eval()
        return tok, model

    try:
        return _try_load(cfg.MODEL_NAME)
    except Exception as e:
        print(f"⚠️ Échec du chargement de '{cfg.MODEL_NAME}' ({e}).")
        print(f"↪️ Repli sur le modèle non gated '{cfg.FALLBACK_MODEL_NAME}'...")
        return _try_load(cfg.FALLBACK_MODEL_NAME)


def teacher_generate(tokenizer, model, user_prompt: str, max_new_tokens: int, temperature: float) -> str:
    """Génère une complétion via le template de chat natif du modèle
    enseignant (Mistral/Zephyr utilisent tous deux `apply_chat_template`)."""
    messages = [{"role": "user", "content": user_prompt}]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
        model.device
    )
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ============================================================================
# 4. Génération des questions par sujet
# ============================================================================

_QUESTION_LINE_RE = re.compile(r"^\s*\d+[.)]\s*(.+)$", re.MULTILINE)


def generate_questions(tokenizer, model, topic: str, n: int, cfg: "Config") -> list[str]:
    prompt = (
        f"Génère {n} questions variées et intéressantes en français sur le thème : {topic}.\n"
        "Mélange des questions factuelles simples ET des questions qui demandent un raisonnement "
        "en plusieurs étapes (calcul, déduction logique, comparaison, causalité).\n"
        "Réponds STRICTEMENT sous forme de liste numérotée, une question par ligne, sans autre texte :\n"
        "1. ...\n2. ...\n"
    )
    raw = teacher_generate(
        tokenizer, model, prompt,
        max_new_tokens=cfg.MAX_NEW_TOKENS_QUESTIONS,
        temperature=cfg.TEMPERATURE_QUESTIONS,
    )
    questions = [q.strip() for q in _QUESTION_LINE_RE.findall(raw) if len(q.strip()) > 8]
    return questions[:n]


# ============================================================================
# 5. Génération de la réponse avec trace de raisonnement
# ============================================================================

ANSWER_PROMPT_TEMPLATE = """Réponds à la question suivante en français. Réfléchis d'abord étape par \
étape (calculs intermédiaires, déductions), PUIS donne la réponse finale.

Réponds STRICTEMENT dans ce format, sans rien ajouter avant ou après :

RAISONNEMENT:
<tes étapes de raisonnement, concises, une idée par ligne>
RÉPONSE_FINALE:
<réponse finale, complète, directement utilisable par l'utilisateur, sans réexpliquer le raisonnement>

Question : {question}
"""

_RAISONNEMENT_RE = re.compile(r"RAISONNEMENT\s*:\s*(.+?)\s*RÉPONSE_FINALE\s*:", re.DOTALL)
_REPONSE_FINALE_RE = re.compile(r"RÉPONSE_FINALE\s*:\s*(.+)", re.DOTALL)


def generate_reasoning_answer(tokenizer, model, question: str, cfg: "Config") -> tuple[str, str] | None:
    raw = teacher_generate(
        tokenizer, model, ANSWER_PROMPT_TEMPLATE.format(question=question),
        max_new_tokens=cfg.MAX_NEW_TOKENS_ANSWER, temperature=cfg.TEMPERATURE_ANSWER,
    )
    reasoning_match = _RAISONNEMENT_RE.search(raw)
    final_match = _REPONSE_FINALE_RE.search(raw)
    if not reasoning_match or not final_match:
        return None  # génération mal formée, on ignore cet exemple plutôt que de polluer le dataset

    reasoning = reasoning_match.group(1).strip()
    final_answer = final_match.group(1).strip()
    if len(reasoning) < 5 or len(final_answer) < 2:
        return None
    return reasoning, final_answer


def generate_direct_answer(tokenizer, model, question: str, cfg: "Config") -> str:
    """Pour la fraction de questions NON forcées en mode raisonnement (README,
    Phase 3 : le modèle doit aussi savoir répondre directement quand une
    décomposition n'apporte rien) — réponse concise, sans trace de raisonnement."""
    prompt = f"Réponds de façon concise et directe, en français, à la question suivante :\n{question}"
    return teacher_generate(
        tokenizer, model, prompt, max_new_tokens=cfg.MAX_NEW_TOKENS_ANSWER, temperature=cfg.TEMPERATURE_ANSWER
    )


# ============================================================================
# 6. Formatage au format Opsiom + écriture JSONL (reprise possible)
# ============================================================================

def format_assistant_field(cfg: "Config", reasoning: str | None, final_answer: str) -> str:
    """Construit le contenu du champ "assistant" du JSONL, au format attendu
    par train_opsiom.py::build_chat_examples() (qui ajoute lui-même le tag de
    fin de séquence — ne PAS le dupliquer ici)."""
    if reasoning is None:
        return final_answer
    return f"{cfg.REASONING_START_TAG}\n{reasoning}\n{cfg.REASONING_END_TAG}\n{final_answer}"


def load_already_generated_questions(path: str) -> set[str]:
    seen = set()
    if not os.path.exists(path):
        return seen
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                seen.add(row["user"].strip())
            except (json.JSONDecodeError, KeyError):
                continue
    if seen:
        print(f"♻️  {len(seen):,} question(s) déjà générée(s) précédemment — ignorées (reprise).")
    return seen


def append_example(path: str, question: str, assistant_field: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"user": question, "assistant": assistant_field}, ensure_ascii=False) + "\n")


# ============================================================================
# 7. Boucle principale
# ============================================================================

def main():
    cfg = Config()
    random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    project_dir = _resolve_project_dir(cfg)
    output_path = os.path.join(project_dir, cfg.OUTPUT_JSONL_FILENAME)

    tokenizer, model = load_teacher_model(cfg)
    already_seen = load_already_generated_questions(output_path)

    total_generated = 0
    total_skipped = 0
    t_start = time.time()

    print(f"\n🏗️ Génération sur {len(cfg.TOPICS)} sujet(s), "
          f"~{cfg.QUESTIONS_PER_TOPIC} question(s) par sujet...\n")

    for topic_idx, topic in enumerate(cfg.TOPICS, start=1):
        print(f"[{topic_idx}/{len(cfg.TOPICS)}] 📚 Sujet : {topic}")
        questions = generate_questions(tokenizer, model, topic, cfg.QUESTIONS_PER_TOPIC, cfg)
        print(f"   ↳ {len(questions)} question(s) générée(s).")

        for q_idx, question in enumerate(questions, start=1):
            if question in already_seen:
                total_skipped += 1
                continue

            force_reasoning = random.random() < cfg.REASONING_FRACTION
            try:
                if force_reasoning:
                    result = generate_reasoning_answer(tokenizer, model, question, cfg)
                    if result is None:
                        total_skipped += 1
                        continue
                    reasoning, final_answer = result
                    assistant_field = format_assistant_field(cfg, reasoning, final_answer)
                else:
                    final_answer = generate_direct_answer(tokenizer, model, question, cfg)
                    if len(final_answer) < 2:
                        total_skipped += 1
                        continue
                    assistant_field = format_assistant_field(cfg, None, final_answer)
            except Exception as e:  # pragma: no cover - garde-fou, ne bloque pas toute la génération
                print(f"   ⚠️ Erreur sur la question '{question[:50]}...' ({e}) — ignorée.")
                total_skipped += 1
                continue

            append_example(output_path, question, assistant_field)
            already_seen.add(question)
            total_generated += 1

            if total_generated % cfg.SAVE_EVERY == 0:
                elapsed = time.time() - t_start
                print(f"   💾 {total_generated} exemples sauvegardés "
                      f"({total_skipped} ignorés) — {elapsed:.0f}s écoulées.")

            # Libère la mémoire GPU périodiquement (utile sur un T4 16 Go
            # avec un modèle 7B en 4-bit, marge réduite).
            if total_generated % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    print(f"\n✅ Génération terminée : {total_generated} exemples ajoutés "
          f"({total_skipped} ignorés) en {(time.time() - t_start) / 60:.1f} min.")
    print(f"📂 Fichier : '{output_path}'")
    print(
        "\n👉 Pour l'utiliser : dans train_opsiom.py, mets\n"
        f"   Config.CHAT_LOCAL_JSONL = '{cfg.OUTPUT_JSONL_FILENAME}'\n"
        "   (ou fusionne son contenu avec un dialogues_fr.jsonl existant), "
        "puis lance TRAINING_MODE='chat'.\n"
        "   ⚠️ N'oublie pas d'ajouter les tags <|reasoning|>/<|endreasoning|> au "
        "tokenizer AVANT l'entraînement (voir add_reasoning_tokens_to_tokenizer() "
        "ci-dessous)."
    )


# ============================================================================
# 8. Ajout des tags de raisonnement au tokenizer Opsiom (à lancer UNE FOIS,
# avant le fine-tuning chat, sur le même fr_bpe_tokenizer.json que celui
# utilisé par train_opsiom.py)
# ============================================================================

def add_reasoning_tokens_to_tokenizer(
    tokenizer_path: str = "fr_bpe_tokenizer.json",
    output_path: str | None = None,
    reasoning_start_tag: str = Config.REASONING_START_TAG,
    reasoning_end_tag: str = Config.REASONING_END_TAG,
) -> None:
    """Ajoute <|reasoning|> et <|endreasoning|> comme tokens spéciaux
    "atomiques" au tokenizer BPE d'Opsiom, à la manière de <|endoftext|>
    (voir train_opsiom.py::build_or_load_french_tokenizer). Sans cette étape,
    le tokenizer byte-level découperait ces tags en fragments arbitraires, et
    le modèle ne pourrait jamais apprendre à les reconnaître comme des
    marqueurs structurels.

    ⚠️ Ajouter des tokens spéciaux change le vocab_size du tokenizer — donc
    la couche d'embedding du modèle doit être agrandie en conséquence AVANT
    de charger un ancien checkpoint dessus (model.tok_embeddings et
    model.lm_head, poids liés). Si tu repars d'un checkpoint de base existant
    pour le fine-tuning chat (cas normal, cf. train_opsiom.py::run_chat_training),
    fais cet appel puis redimensionne les deux couches concernées (nouvelles
    lignes initialisées aléatoirement pour les 2 nouveaux tokens) avant
    d'appeler model.load_state_dict() — ce script ne fait QUE la partie
    tokenizer, la partie modèle doit être adaptée dans train_opsiom.py.
    """
    from tokenizers import Tokenizer

    output_path = output_path or tokenizer_path
    tok = Tokenizer.from_file(tokenizer_path)
    added = tok.add_special_tokens([reasoning_start_tag, reasoning_end_tag])
    tok.save(output_path)
    print(
        f"✅ {added} nouveau(x) token(s) spécial(aux) ajouté(s) "
        f"('{reasoning_start_tag}', '{reasoning_end_tag}') — "
        f"nouveau vocab_size: {tok.get_vocab_size()}. Sauvegardé dans '{output_path}'."
    )


if __name__ == "__main__":
    main()
