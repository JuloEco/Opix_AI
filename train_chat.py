# ============================================================================
# train_chat.py — Fine-tuning "dialogue" (mode Utilisateur/Assistant)
# ============================================================================
#
# CE QUE FAIT CE SCRIPT :
#   Il reprend le modèle déjà entraîné (best_model.pt, sur contes + Wikipedia)
#   et poursuit son entraînement sur un petit corpus de dialogues formaté avec
#   les mêmes tags [Utilisateur]/[Assistant] que test_IA.py, avec la loss
#   masquée sur la question (on n'entraîne le modèle qu'à prédire la RÉPONSE,
#   pas à réciter la question de l'utilisateur).
#
# CE QUE CE SCRIPT NE FAIT PAS :
#   Il ne transforme pas comme par magie un mini-modèle de quelques millions de
#   paramètres en assistant conversationnel compétent. Le jeu d'exemples fourni
#   ci-dessous (SEED_DIALOGUES) est volontairement petit : il sert de point de
#   départ fonctionnel et pour vérifier que le mécanisme marche. Pour un vrai
#   changement de comportement, il vous faudra un corpus de dialogues bien plus
#   large (au moins quelques milliers d'exemples) — voir la section
#   "ÉTENDRE LE CORPUS" plus bas dans ce fichier.
#
# UTILISATION :
#   python train_chat.py
#   (il cherche best_model.pt et fr_bpe_tokenizer.json dans le même dossier,
#   comme test_IA.py, et écrit chat_model.pt)
#
# ============================================================================

import os
import sys
import json
import random
import math

import torch
import torch.nn.functional as F

from test_IA import ModernLLM, ModelArgs, FrenchTokenizerWrapper, ChatFormatter


# ============================================================================
# Corpus de dialogues — [ (question, réponse), ... ]
# ============================================================================
#
# ÉTENDRE LE CORPUS :
#   Le moyen le plus simple d'ajouter des exemples sans toucher au code est de
#   créer un fichier `dialogues_fr.jsonl` à côté de ce script, une paire par
#   ligne :
#     {"user": "Comment tu t'appelles ?", "assistant": "Je m'appelle Assistant, ravi de discuter avec toi !"}
#     {"user": "Quelle est la capitale de la France ?", "assistant": "La capitale de la France est Paris."}
#   Ce script le charge automatiquement s'il existe et fusionne avec
#   SEED_DIALOGUES ci-dessous. Vous pouvez aussi générer ce fichier à partir
#   d'un dataset d'instructions français existant (ex: via `datasets` sur
#   Hugging Face) en le reformattant dans ce format JSONL.

SEED_DIALOGUES: list[tuple[str, str]] = [
    ("Bonjour !", "Bonjour ! Comment puis-je t'aider aujourd'hui ?"),
    ("Salut", "Salut ! Ravi de te parler."),
    ("Ça va ?", "Ça va très bien, merci ! Et toi, comment vas-tu ?"),
    ("Comment tu vas ?", "Je vais bien, merci de demander !"),
    ("Comment tu t'appelles ?", "Je n'ai pas vraiment de nom, tu peux m'appeler Assistant."),
    ("Qui es-tu ?", "Je suis un assistant conversationnel qui essaie de répondre à tes questions."),
    ("Merci beaucoup", "Avec plaisir !"),
    ("Merci", "De rien !"),
    ("Au revoir", "Au revoir, à bientôt !"),
    ("À plus tard", "À plus tard !"),
    ("Quel temps fait-il ?", "Je n'ai pas accès à la météo en temps réel, désolé."),
    ("Quelle est la capitale de la France ?", "La capitale de la France est Paris."),
    ("Quelle est la capitale de l'Italie ?", "La capitale de l'Italie est Rome."),
    ("Combien font deux plus deux ?", "Deux plus deux font quatre."),
    ("Combien font trois plus cinq ?", "Trois plus cinq font huit."),
    ("Raconte-moi une histoire courte.", "Il était une fois une petite étoile qui rêvait de briller pour tout le monde."),
    ("Quel est ton plat préféré ?", "Je n'ai pas de préférences culinaires, mais on me parle souvent de la ratatouille !"),
    ("Peux-tu m'aider ?", "Bien sûr, dis-moi ce dont tu as besoin."),
    ("J'ai besoin d'aide.", "Je suis là pour ça, explique-moi ton problème."),
    ("Quel jour sommes-nous ?", "Je n'ai pas accès à la date actuelle, désolé."),
    ("Tu es un robot ?", "Je suis un programme informatique, oui, pas un humain."),
    ("Es-tu intelligent ?", "J'essaie de répondre du mieux possible, mais je reste un petit modèle."),
    ("Quelle est la couleur du ciel ?", "Le ciel est généralement bleu pendant la journée."),
    ("Où se trouve la tour Eiffel ?", "La tour Eiffel se trouve à Paris, en France."),
    ("Quel est le plus grand océan du monde ?", "Le plus grand océan du monde est l'océan Pacifique."),
    ("Comment fait-on du pain ?", "On mélange de la farine, de l'eau, du sel et de la levure, puis on laisse lever avant de cuire."),
    ("Peux-tu compter jusqu'à cinq ?", "Un, deux, trois, quatre, cinq."),
    ("Bonne nuit", "Bonne nuit, dors bien !"),
    ("Bon appétit", "Merci, bon appétit à toi aussi !"),
    ("Tu aimes la musique ?", "Je n'écoute pas de musique, mais j'aime en parler !"),
]


# ============================================================================
# Dataset d'instructions français (Hugging Face) — optionnel
# ============================================================================
#
# Par défaut: jpacifico/French-Alpaca-dataset-Instruct-55K (55184 exemples,
# format Alpaca instruction/input/output, licence Apache-2.0). Adapté à un
# modèle de cette taille — inutile de viser les datasets à 800k lignes type
# legmlai/openhermes-fr, votre modèle ne pourra de toute façon en absorber
# qu'une fraction ; sous-échantillonnez plutôt fortement (voir HF_SAMPLE_SIZE).

HF_DATASET_NAME = "jpacifico/French-Alpaca-dataset-Instruct-55K"
HF_SAMPLE_SIZE = 3000     # nombre d'exemples à garder (0 = désactivé)
HF_MAX_ANSWER_CHARS = 400  # ignore les réponses trop longues pour max_seq_len=256
SEED = 1337               # seed pour shuffle et reproductibilité


def load_hf_instruction_dataset(sample_size: int = HF_SAMPLE_SIZE) -> list[tuple[str, str]]:
    """Charge et sous-échantillonne un dataset d'instructions français depuis
    Hugging Face, au format Alpaca (instruction/input/output -> question/réponse).
    Retourne [] si `datasets` n'est pas installé ou si le téléchargement échoue
    (pas de connexion, dataset gated, etc.) — le script continue avec les seuls
    exemples locaux (SEED_DIALOGUES + dialogues_fr.jsonl) dans ce cas.
    """
    if sample_size <= 0:
        return []
    try:
        from datasets import load_dataset
    except ImportError:
        print("ℹ️  Package 'datasets' non installé — dataset HF ignoré "
              "(pip install datasets pour l'activer).")
        return []

    try:
        print(f"📥 Téléchargement de {HF_DATASET_NAME} (sous-échantillon de {sample_size})...")
        ds = load_dataset(HF_DATASET_NAME, split="train")
    except Exception as e:
        print(f"⚠️  Impossible de charger {HF_DATASET_NAME} ({e}) — ignoré.")
        return []

    ds = ds.shuffle(seed=SEED if "SEED" in globals() else 1337)
    pairs = []
    for row in ds:
        instruction = (row.get("instruction") or "").strip()
        extra_input = (row.get("input") or "").strip()
        answer = (row.get("output") or "").strip()
        if not instruction or not answer:
            continue
        if len(answer) > HF_MAX_ANSWER_CHARS:
            continue  # trop long pour max_seq_len=256 une fois tokenisé
        # Format Alpaca: si `input` est renseigné, on l'ajoute à la question
        # (ex: "Résume le texte suivant.\n\nTexte : ...").
        question = f"{instruction}\n\n{extra_input}" if extra_input else instruction
        pairs.append((question, answer))
        if len(pairs) >= sample_size:
            break

    print(f"✅ {len(pairs)} exemples retenus depuis {HF_DATASET_NAME} "
          f"(sur {sample_size} demandés, après filtrage par longueur).")
    return pairs


def load_extra_dialogues(script_dir: str) -> list[tuple[str, str]]:
    """Charge des exemples additionnels depuis dialogues_fr.jsonl s'il existe."""
    path = os.path.join(script_dir, "dialogues_fr.jsonl")
    if not os.path.exists(path):
        return []
    extra = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                extra.append((row["user"], row["assistant"]))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠️  Ligne {line_num} de dialogues_fr.jsonl ignorée ({e}).")
    if extra:
        print(f"📥 {len(extra)} exemples chargés depuis dialogues_fr.jsonl")
    return extra


# ============================================================================
# Construction des exemples tokenisés avec masquage de la loss sur la question
# ============================================================================

class ChatExample:
    __slots__ = ("input_ids", "target_ids")

    def __init__(self, input_ids: list[int], target_ids: list[int]):
        self.input_ids = input_ids
        self.target_ids = target_ids


def build_chat_examples(
    dialogues: list[tuple[str, str]],
    tokenizer: FrenchTokenizerWrapper,
    max_seq_len: int,
) -> list[ChatExample]:
    """Encode chaque paire (question, réponse) au même format que ChatFormatter
    (celui utilisé en inférence dans test_IA.py), et masque la loss (-1) sur
    tout ce qui précède la réponse de l'assistant : on n'entraîne le modèle
    qu'à produire la réponse, jamais à "prédire" la question de l'utilisateur.
    """
    examples = []
    skipped = 0

    for question, answer in dialogues:
        prefix_text = f"{ChatFormatter.USER_TAG}\n{question}\n{ChatFormatter.ASSISTANT_TAG}\n"
        full_text = prefix_text + answer + "\n<|endoftext|>"

        prefix_ids = tokenizer.encode(prefix_text)
        full_ids = tokenizer.encode(full_text)

        # Garde-fou : le préfixe doit être un vrai préfixe token-à-token de la
        # séquence complète (c'est le cas avec un tokenizer BPE byte-level tant
        # qu'on encode le même texte de départ à l'identique).
        if full_ids[: len(prefix_ids)] != prefix_ids:
            skipped += 1
            continue

        if len(full_ids) < 2 or len(full_ids) > max_seq_len:
            skipped += 1
            continue

        x = full_ids[:-1]
        y = full_ids[1:]

        # On masque (-1 = ignore_index) toutes les positions de `y` qui
        # correspondent à prédire un token du préfixe (tag + question), pour
        # ne rétropropager la loss que sur la réponse de l'assistant.
        mask_upto = max(0, len(prefix_ids) - 1)
        y = [-1] * mask_upto + y[mask_upto:]

        examples.append(ChatExample(x, y))

    if skipped:
        print(f"⚠️  {skipped} exemple(s) ignoré(s) (trop long ou tag mal aligné).")
    return examples


def collate_batch(batch: list[ChatExample], pad_id: int, device: str):
    """Pad un batch d'exemples de longueur variable. Les positions de padding
    reçoivent aussi target=-1 pour ne pas polluer la loss."""
    max_len = max(len(ex.input_ids) for ex in batch)
    x_batch, y_batch = [], []
    for ex in batch:
        pad_len = max_len - len(ex.input_ids)
        x_batch.append(ex.input_ids + [pad_id] * pad_len)
        y_batch.append(ex.target_ids + [-1] * pad_len)
    x = torch.tensor(x_batch, dtype=torch.long, device=device)
    y = torch.tensor(y_batch, dtype=torch.long, device=device)
    return x, y


def compute_loss(model: ModernLLM, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ModernLLM.forward() dans test_IA.py ne calcule pas la loss lui-même
    (contrairement à la version main.py) — on la calcule donc ici."""
    logits = model(x)  # (B, T, vocab_size)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        y.view(-1),
        ignore_index=-1,
    )
    return loss


# ============================================================================
# Point d'entrée
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, "best_model.pt")
    tokenizer_path = os.path.join(script_dir, "fr_bpe_tokenizer.json")
    output_path = os.path.join(script_dir, "chat_model.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print("🗨️  FINE-TUNING DIALOGUE (mode Utilisateur/Assistant)")
    print("=" * 65)
    print(f"💻 Périphérique : {device.upper()}")

    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer introuvable : {tokenizer_path}")
        sys.exit(1)
    tokenizer = FrenchTokenizerWrapper(tokenizer_path)
    print(f"✅ Tokenizer chargé (vocab={tokenizer.vocab_size:,})")

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint introuvable : {checkpoint_path}")
        print("💡 Lancez d'abord main.py pour entraîner un modèle de base.")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    saved_args = checkpoint.get("args") if isinstance(checkpoint, dict) else None

    if saved_args is not None and hasattr(saved_args, "vocab_size"):
        args = saved_args
        args.device = device
    else:
        emb_shape = state_dict["tok_embeddings.weight"].shape
        args = ModelArgs(vocab_size=emb_shape[0], dim=emb_shape[1], device=device)

    model = ModernLLM(args).to(device)
    model.load_state_dict(state_dict)
    print(f"✅ Modèle de base chargé depuis '{checkpoint_path}' "
          f"(vocab={args.vocab_size}, dim={args.dim}, layers={args.n_layers})")

    # --- Corpus de dialogues ---
    # Ordre: petit seed local + fichier JSONL local + dataset HF sous-échantillonné.
    dialogues = SEED_DIALOGUES + load_extra_dialogues(script_dir) + load_hf_instruction_dataset()
    random.shuffle(dialogues)
    examples = build_chat_examples(dialogues, tokenizer, max_seq_len=args.max_seq_len)
    print(f"📚 {len(examples)} exemples de dialogue prêts pour l'entraînement "
          f"({len(dialogues)} paires question/réponse au total).")

    if len(examples) < 5:
        print("❌ Pas assez d'exemples valides pour entraîner. Ajoutez des paires "
              "dans dialogues_fr.jsonl.")
        sys.exit(1)

    n_val = max(1, len(examples) // 10)
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]

    # --- Hyperparamètres de fine-tuning ---
    # LR nettement plus bas que le pré-entraînement : on ajuste le comportement
    # sans détruire ce que le modèle a déjà appris sur la langue française.
    # EPOCHS plus bas qu'avant (15 -> 6) car avec le dataset HF activé, on a
    # maintenant des milliers d'exemples au lieu de 30 : moins de passes
    # suffisent, et ça évite de sur-apprendre les 30 exemples seed.
    EPOCHS = 6
    BATCH_SIZE = 16
    LR = 5e-5
    PAD_ID = tokenizer.eos_token_id

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.01)

    def run_epoch(data: list[ChatExample], train: bool) -> float:
        model.train(train)
        random.shuffle(data) if train else None
        losses = []
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i + BATCH_SIZE]
            x, y = collate_batch(batch, PAD_ID, device)
            with torch.set_grad_enabled(train):
                loss = compute_loss(model, x, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            losses.append(loss.item())
        return sum(losses) / max(1, len(losses))

    print(f"\n🏋️  Fine-tuning pour {EPOCHS} epochs (lr={LR}, batch_size={BATCH_SIZE})...\n")
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(train_examples, train=True)
        val_loss = run_epoch(val_examples, train=False)
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"model_state_dict": model.state_dict(), "args": args,
                 "step": epoch, "val_loss": val_loss},
                output_path,
            )
            marker = "  💾 meilleur modèle sauvegardé"
        print(f"epoch {epoch:02d}/{EPOCHS} | train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f}{marker}")

    print(f"\n✅ Fine-tuning terminé. Meilleur val_loss={best_val_loss:.4f}")
    print(f"📂 Modèle sauvegardé dans '{output_path}'")
    print("\n👉 Pour l'utiliser, dans test_IA.py, changez :")
    print('     checkpoint_path = os.path.join(script_dir, "best_model.pt")')
    print("   en :")
    print('     checkpoint_path = os.path.join(script_dir, "chat_model.pt")')


if __name__ == "__main__":
    main()