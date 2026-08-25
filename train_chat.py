# ============================================================================
# train_opsiom.py — Script d'entraînement complet pour Google Colab
# Projet : Opsiom (LLM auto-régressif type LLaMA/Qwen)
#
# Ce script AJOUTE une pipeline d'entraînement à côté de test_IA_2.py / app.py,
# sans les modifier. Il redéfinit localement l'architecture (identique, pour
# rester 100% autonome dans une cellule Colab) et produit des checkpoints
# .pt strictement compatibles avec la détection automatique de dimensions de
# test_IA_2.py (clés: model_state_dict, args, val_loss, step).
#
# UTILISATION SUR COLAB :
#   1. Colle ce fichier entier dans une cellule (ou fais %%writefile puis
#      !python train_opsiom.py si tu préfères l'exécuter en subprocess).
#   2. Ajuste la section CONFIGURATION ci-dessous (taille du modèle, source du
#      tokenizer, dataset, hyperparamètres).
#   3. Exécute. Le script installe ses dépendances, monte Drive si demandé,
#      entraîne, sauvegarde, et pousse optionnellement sur le Hub HF.
# ============================================================================

import subprocess
import sys


def _pip_install(packages: list[str]) -> None:
    """Installe les paquets requis si absents (silencieux si déjà présents)."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *packages],
        check=False,
    )


_pip_install(["tokenizers", "datasets", "huggingface_hub", "requests"])

import os
import math
import time
import random
import inspect
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizers import Tokenizer


# ============================================================================
# 1. CONFIGURATION — à ajuster avant de lancer
# ============================================================================

class Config:
    # --- Choix de la taille du modèle Opsiom ---
    # Voir MODEL_PRESETS plus bas pour le détail des dimensions de chaque taille.
    # N'est utilisé que pour un tout premier entraînement de base (TRAINING_MODE
    # ="base") sans checkpoint existant — en mode "chat", on repart toujours du
    # checkpoint de base déjà entraîné (cf. CHECKPOINT_FILENAME).
    MODEL_SIZE = "Opsiom-Large"  # "Opsiom-Micro" | "Opsiom-Nano" | "Opsiom-Small" | "Opsiom-Medium" | "Opsiom-Large"

    # --- Mode d'entraînement ---
    # "chat" : fine-tuning conversationnel — repart du modèle de base déjà
    #          entraîné (CHECKPOINT_FILENAME) et l'entraîne sur des paires
    #          question/réponse au format USER_TAG/ASSISTANT_TAG, avec la loss
    #          masquée sur la question (on n'entraîne que la réponse). C'est le
    #          mode à utiliser pour obtenir un comportement de chatbot.
    # "base" : pré-entraînement classique next-token sur du texte brut continu
    #          (ce que faisait la version précédente du script).
    TRAINING_MODE = "base"  # "chat" | "base"

    # --- Google Drive ---
    # Dossier unique contenant déjà tes fichiers (tokenizer, .pt) — celui où
    # se trouvent tes checkpoints existants. Tout ce qui suit est résolu à
    # l'intérieur de ce dossier une fois Drive monté, donc plus besoin de
    # téléverser quoi que ce soit à chaque session.
    USE_GOOGLE_DRIVE = False
    DRIVE_MOUNT_POINT = "/content/drive"
    PROJECT_DIR = "/kaggle/working"

    # --- Tokenizer : priorité au repo Hugging Face, repli sur fichier local ---
    # Exemple de repo HF : "cecile/opsiom-fr-tokenizer" (doit contenir tokenizer.json)
    TOKENIZER_HF_REPO = "JuloEco/opsiom-fr-tokenizer"  # laisser vide pour ignorer et chercher dans PROJECT_DIR
    TOKENIZER_LOCAL_PATH = "fr_bpe_tokenizer.json"  # nom de fichier, résolu dans PROJECT_DIR

    # --- Tags de dialogue (mode "chat") ---
    # ⚠️ Ces tags doivent être IDENTIQUES à ceux utilisés par ChatFormatter dans
    # ton test_IA_2.py au moment de l'inférence, sinon la détection de fin de
    # tour ne fonctionnera plus (le modèle continuera à halluciner la suite).
    # Si tu changes ces valeurs, mets aussi à jour USER_TAG/ASSISTANT_TAG dans
    # test_IA_2.py en conséquence.
    USER_TAG = "<|Utilisateur|>"
    ASSISTANT_TAG = "<|Assistant|>"
    EOT_TOKEN = "<|endoftext|>"  # doit exister dans le tokenizer (token spécial)

    # --- Dataset(s) pour le mode "chat" (chargés automatiquement depuis le Hub HF) ---
    # Liste de datasets HF à concaténer. Par défaut : French-Alpaca + Alpaca-in-french,
    # deux traductions différentes du Alpaca original -> davantage de diversité de
    # formulation pour le même type de tâches (Q/R factuelles, instructions courtes).
    # Chaque entrée peut préciser ses propres noms de colonnes via instruction_col/
    # input_col/output_col si le dataset ne suit pas la convention instruction/input/
    # output — ex: tbboukhari/Alpaca-in-french utilise "saisir"/"sortir" (français).
    # ⚠️ Ces deux datasets restent du "instruction-following" façon Alpaca : peu
    # d'exemples de récit/conte long. Si tes échantillons de génération restent
    # faibles sur "raconte-moi une histoire" / "il était une fois" après avoir
    # ajouté ces datasets, c'est un signe qu'il faut du contenu narratif dédié —
    # voir CHAT_LOCAL_JSONL plus bas et la fonction build_story_jsonl_from_corpus()
    # pour transformer ta propre prose littéraire en exemples (prompt, histoire).
    CHAT_HF_DATASETS = [
        # sample_size=0 -> tout charger. Contrairement au corpus "base" (où
        # les sources web dépassent largement ce qu'un modèle de 25M peut
        # exploiter), les deux Alpaca ci-dessous sont des jeux BORNÉS (~55k et
        # ~52k lignes au total) — les charger en entier est raisonnable, pas
        # un problème d'échelle. Le vrai plafond de diversité, lui, ne bouge
        # pas : ce sont deux traductions du même Alpaca source, donc "tout
        # charger" apporte de la robustesse de formulation, pas plus de
        # couverture factuelle réelle.
        {"name": "jpacifico/French-Alpaca-dataset-Instruct-55K", "sample_size": 0},
        {
            "name": "tbboukhari/Alpaca-in-french", "sample_size": 0,
            # Colonnes confirmées via le diagnostic runtime (le rendu web de la
            # page HF masquait un espace en tête) : ' saisir' / ' sortir', pas
            # 'saisir' / 'sortir'.
            "instruction_col": "instruction", "input_col": " saisir", "output_col": " sortir",
        },
        # sample_size=0 ici aussi, mais par honnêteté : le sous-ensemble
        # français d'oasst1 ne contient que ~4 251 messages AU TOTAL (source
        # utilisateur+assistant confondus, cf. README du dataset) — donc
        # l'ancien plafond de 4000 ne limitait quasiment rien. Le vrai
        # plafond ici, c'est le dataset lui-même, pas notre config.
        {"loader": "oasst1", "name": "OpenAssistant/oasst1 (fr)", "sample_size": 0},
    ]
    CHAT_MAX_ANSWER_CHARS = 400      # ignore les réponses trop longues pour max_seq_len
    CHAT_LOCAL_JSONL = "dialogues_fr.jsonl"  # exemples additionnels optionnels, résolu dans PROJECT_DIR
    # Epochs ramenés de 10 à 6 : le corpus passe d'environ 44k à ~110k+
    # exemples (deux Alpaca en entier + tout oasst1-fr), donc à LR et nombre
    # d'epochs égal, le temps de calcul par run serait ~2.5x plus long pour un
    # gain marginal (le contenu Alpaca reste tout de même très redondant entre
    # les deux sources). Surveille val_loss : si elle continue de baisser à la
    # fin des 6 epochs, remonte ce chiffre plutôt que de le laisser tel quel.
    CHAT_EPOCHS = 6
    CHAT_BATCH_SIZE = 16
    CHAT_LR = 1e-4
    CHAT_VAL_FRACTION = 0.1
    CHAT_CHECKPOINT_FILENAME = "chat_model.pt"  # nom compatible avec test_IA_2.py

    # --- Dataset pour le mode "base" (pré-entraînement sur texte brut) ---
    # ⚠️ asi/wikitext_fr (ancienne valeur) ne charge plus : depuis `datasets`
    # >= 4.0.0, les datasets qui reposent sur un script Python de chargement
    # (comme celui-ci) sont bloqués définitivement — même trust_remote_code=True
    # a été supprimé, ce n'est pas contournable par une option. Remplacé par
    # wikimedia/wikipedia, dataset Parquet natif maintenu par Hugging Face
    # (donc pérenne), config "20231101.fr" pour le français. Champs identiques
    # (id/url/title/text), même colonne "text" qu'avant.
    DATASET_HF_NAME = "wikimedia/wikipedia"
    DATASET_HF_CONFIG = "20231101.fr"
    DATASET_TEXT_COLUMN = "text"   # nom de colonne texte dans le dataset HF
    DATASET_STREAMING = True            # utile pour les gros datasets sur Colab
    DATASET_MAX_CHARS = 15_000_000      # plafond de caractères chargés depuis le HF hub
    LOCAL_CORPUS_FALLBACK = "training_corpus.txt"  # repli local, résolu dans PROJECT_DIR

    # ⚠️ IMPORTANT : contrairement à ce que ce script faisait jusqu'ici, on
    # remixe ici TinyStories-French avec Wikipedia — exactement la recette de
    # main.py (celui qui a produit best_model.pt à l'origine). Sans ça, un
    # entraînement "base" repris via train_chat.py ne voit QUE du Wikipedia et
    # dérive vers un style encyclopédique, perdant le style "conte" acquis
    # pendant le pré-entraînement initial.
    TINYSTORIES_HF_NAME = "iproskurina/TinyStories-French"
    TINYSTORIES_COLUMN = "french-tinystories"
    TINYSTORIES_MAX_STORIES = 20000  # le dataset n'en contient qu'environ 20k, donc en pratique tout est utilisé

    # --- Hugging Face Hub (push optionnel en fin d'entraînement) ---
    PUSH_TO_HUB = False
    HUB_REPO_ID = "JuloEco/opsiom-fr-tokenizer"          # ex: "cecile/opsiom-nano-fr"
    HUB_PRIVATE = True

    # --- Hyperparamètres d'entraînement (mode "base") ---
    MAX_STEPS = 6000
    WARMUP_STEPS = 300
    BATCH_SIZE = 32
    MAX_LR = 3e-4
    MIN_LR = 3e-5
    GRAD_CLIP = 1.0
    EVAL_INTERVAL = 200
    EVAL_BATCHES = 20
    GEN_INTERVAL = 500
    VAL_FRACTION = 0.05

    # ⚠️ En reprise (checkpoint existant), le schedule LR repartait jusqu'ici
    # de zéro à CHAQUE run : warmup vers MAX_LR=3e-4 même sur un modèle déjà
    # entraîné (val_loss ~3.09). Ce pic fait sortir le modèle de son minimum
    # actuel sans que 6000 steps suffisent à y revenir (constaté : val_loss
    # reste élevée ~3.20-3.28 pendant tout le run, jamais de retour sous le
    # niveau de départ). En reprise, on utilise un pic beaucoup plus bas et un
    # warmup plus court — un ordre de grandeur cohérent avec du fine-tuning
    # continué plutôt qu'un entraînement from scratch.
    RESUME_LR_SCALE = 0.15    # MAX_LR effectif en reprise = MAX_LR * ce facteur (≈4.5e-5 ici)
    RESUME_WARMUP_STEPS = 50  # warmup plus court, moins de distance à parcourir
    SEED = 1337

    CHECKPOINT_FILENAME = "best_model.pt"  # nom compatible avec test_IA_2.py



# ============================================================================
# 2. PRESETS DE TAILLES DE MODÈLE
# ============================================================================
#
# Ces presets ne fixent pas `vocab_size` : il est écrasé dynamiquement par la
# taille réelle du tokenizer chargé (cf. section 4), comme dans test_IA_2.py.
# Les tailles de paramètres indiquées sont approximatives (poids liés
# tok_embeddings <-> lm_head, donc comptés une seule fois).

MODEL_PRESETS = {
    "Opsiom-Micro":  dict(dim=256, n_layers=6, n_heads=4, n_kv_heads=2, max_seq_len=192),    # ~5M params (vocab~16k)
    "Opsiom-Nano":   dict(dim=384, n_layers=8, n_heads=8, n_kv_heads=4, max_seq_len=256),    # ~15M params
    "Opsiom-Small":  dict(dim=576, n_layers=10, n_heads=9, n_kv_heads=3, max_seq_len=320),   # ~45M params
    "Opsiom-Medium": dict(dim=768, n_layers=12, n_heads=12, n_kv_heads=4, max_seq_len=384),  # ~95M params
    "Opsiom-Large":  dict(dim=1024, n_layers=16, n_heads=16, n_kv_heads=4, max_seq_len=512), # ~225M params
}
# ⚠️ Medium et Large sont des sauts de capacité importants (x2 puis x2.4 en
# paramètres par rapport à Small), pas juste des variantes cosmétiques :
# - VRAM (Colab) : Medium tient sans souci sur un T4 (16 Go) en pré-entraînement
#   "base" avec BATCH_SIZE=32 ; pour Large, réduis BATCH_SIZE (ex: 16) ou passe
#   sur A100 si tu restes à 32, sous peine d'OOM.
# - Ces tailles ne valent le coup QUE si le volume/diversité de données suit.
#   Le même problème déjà observé sur le fine-tuning chat (25M params, ~45k
#   exemples factuels type Alpaca, capitale de l'Espagne fausse) se reproduirait
#   en pire avec un modèle 2-9x plus gros mais entraîné sur le même corpus de
#   base (TinyStories-FR + ~15-25M caractères Wikipedia) : plus de capacité à
#   nourrir sans plus de données ne donne pas mécaniquement un meilleur modèle,
#   ça peut juste sous-utiliser la capacité ou légèrement overfitter plus vite.


@dataclass
class ModelArgs:
    """Configuration hyperparamétrique pour architecture Transformer type LLaMA/Qwen."""
    vocab_size: int = 16000
    dim: int = 384
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int | None = 4
    max_seq_len: int = 256
    dropout: float = 0.1
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def build_model_args(model_size: str, vocab_size: int) -> ModelArgs:
    if model_size not in MODEL_PRESETS:
        raise ValueError(
            f"Taille de modèle inconnue: '{model_size}'. "
            f"Choix possibles: {list(MODEL_PRESETS.keys())}"
        )
    preset = MODEL_PRESETS[model_size]
    return ModelArgs(vocab_size=vocab_size, **preset)


# ============================================================================
# 3. ARCHITECTURE — identique à test_IA_2.py (RMSNorm, RoPE, SwiGLU, GQA+SDPA)
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._norm(x.float()).type_as(x)
        return out * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "head_dim doit être pair pour RoPE"
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int, start_pos: int = 0):
        cos = self.cos_cached[start_pos:start_pos + seq_len].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[start_pos:start_pos + seq_len].to(dtype=x.dtype, device=x.device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    xq_rotated = (xq * cos) + (rotate_half(xq) * sin)
    xk_rotated = (xk * cos) + (rotate_half(xk) * sin)
    return xq_rotated, xk_rotated


class SwiGLUFeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = int(8 * args.dim / 3)
        multiple_of = 256
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w1(x))
        up = self.w3(x)
        return self.dropout(self.w2(gate * up))


class ModernCausalAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        assert args.n_heads % self.n_kv_heads == 0, "n_heads doit être divisible par n_kv_heads (GQA)"
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.dropout_p = args.dropout

        self.wq = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)
        self.resid_dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, kv_cache=None):
        B, T, C = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_emb(q, k, cos, sin)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            if past_k is not None:
                k = torch.cat((past_k, k), dim=2)
                v = torch.cat((past_v, v), dim=2)
            new_kv_cache = (k, v)
        else:
            new_kv_cache = None

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        is_causal = kv_cache is None or k.shape[2] == q.shape[2]
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        y = self.resid_dropout(self.wo(y))

        if kv_cache is not None:
            return y, new_kv_cache
        return y


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attn = ModernCausalAttention(args)
        self.ffn = SwiGLUFeedForward(args)
        self.norm1 = RMSNorm(args.dim, eps=args.norm_eps)
        self.norm2 = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, kv_cache=None):
        if kv_cache is not None:
            attn_out, new_kv_cache = self.attn(self.norm1(x), cos, sin, kv_cache=kv_cache)
            x = x + attn_out
            x = x + self.ffn(self.norm2(x))
            return x, new_kv_cache
        else:
            x = x + self.attn(self.norm1(x), cos, sin)
            x = x + self.ffn(self.norm2(x))
            return x


class ModernLLM(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.dropout = nn.Dropout(args.dropout)

        head_dim = args.dim // args.n_heads
        self.rope = RotaryEmbedding(head_dim, max_seq_len=args.max_seq_len, theta=args.rope_theta)

        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm_f = RMSNorm(args.dim, eps=args.norm_eps)

        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.tok_embeddings.weight = self.lm_head.weight  # Weight Tying

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("w2.weight") or name.endswith("wo.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * args.n_layers))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        unique_params = {id(p): p for p in self.parameters()}
        return sum(p.numel() for p in unique_params.values())

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_caches: list | None = None,
        start_pos: int = 0,
    ):
        B, T = tokens.shape
        assert start_pos + T <= self.args.max_seq_len, (
            f"Position {start_pos + T} > max_seq_len {self.args.max_seq_len}"
        )

        x = self.tok_embeddings(tokens)
        x = self.dropout(x)
        cos, sin = self.rope(x, seq_len=T, start_pos=start_pos)

        if kv_caches is not None:
            new_kv_caches = []
            for i, layer in enumerate(self.layers):
                x, layer_cache = layer(x, cos, sin, kv_cache=kv_caches[i])
                new_kv_caches.append(layer_cache)
        else:
            for layer in self.layers:
                x = layer(x, cos, sin)
            new_kv_caches = None

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        if kv_caches is not None:
            return logits, loss, new_kv_caches
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        tokenizer: "FrenchTokenizerWrapper",
        max_new_tokens: int = 60,
        temperature: float = 0.8,
        top_k: int | None = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.3,
    ) -> str:
        self.eval()
        device = next(self.parameters()).device

        token_ids = tokenizer.encode(prompt)
        tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
        max_prompt_len = self.args.max_seq_len - 1
        tokens = tokens[:, -max_prompt_len:]

        kv_caches = [(None, None) for _ in range(self.args.n_layers)]
        logits, _, kv_caches = self.forward(tokens, kv_caches=kv_caches, start_pos=0)
        logits = logits[:, -1, :]
        cur_pos = tokens.shape[1]

        for _ in range(max_new_tokens):
            if repetition_penalty != 1.0 and tokens.numel() > 0:
                unique_ids = torch.unique(tokens)
                prev_logits = logits[0, unique_ids]
                penalized = torch.where(
                    prev_logits > 0, prev_logits / repetition_penalty, prev_logits * repetition_penalty
                )
                logits[0, unique_ids] = penalized

            if temperature <= 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                scaled = logits / temperature
                if top_k is not None:
                    top_k_clamped = min(top_k, scaled.size(-1))
                    v, _ = torch.topk(scaled, top_k_clamped)
                    threshold = v[:, [-1]]
                    scaled = torch.where(scaled < threshold, torch.full_like(scaled, float("-inf")), scaled)
                probs = F.softmax(scaled, dim=-1)
                if top_p is not None and top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                    cum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cum - sorted_probs > top_p
                    sorted_probs[mask] = 0.0
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
                next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat((tokens, next_token), dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
            if cur_pos >= self.args.max_seq_len:
                break

            logits, _, kv_caches = self.forward(next_token, kv_caches=kv_caches, start_pos=cur_pos)
            logits = logits[:, -1, :]
            cur_pos += 1

        self.train()
        return tokenizer.decode(tokens[0].tolist())


# ============================================================================
# 4. TOKENIZER — chargement HF Hub prioritaire, repli fichier local
# ============================================================================

class FrenchTokenizerWrapper:
    """Charge un tokenizer BPE soit depuis un repo Hugging Face (via
    `Tokenizer.from_pretrained`, avec repli manuel via `huggingface_hub` si
    l'API `from_pretrained` n'est pas disponible dans la version installée
    de `tokenizers`), soit depuis un fichier .json local."""

    def __init__(self, hf_repo: str = "", local_path: str = "fr_bpe_tokenizer.json"):
        self._tok = self._load(hf_repo, local_path)

        eot_id = self._tok.token_to_id("<|endoftext|>")
        self.eos_token_id = eot_id if eot_id is not None else 0
        self.vocab_size = self._tok.get_vocab_size()

    def _load(self, hf_repo: str, local_path: str) -> Tokenizer:
        if hf_repo:
            try:
                print(f"📥 Chargement du tokenizer depuis le Hub HF: '{hf_repo}'...")
                return Tokenizer.from_pretrained(hf_repo)
            except Exception as e:
                print(f"⚠️  from_pretrained a échoué ({e}). Tentative via huggingface_hub...")
                try:
                    from huggingface_hub import hf_hub_download
                    downloaded_path = hf_hub_download(repo_id=hf_repo, filename="tokenizer.json")
                    return Tokenizer.from_file(downloaded_path)
                except Exception as e2:
                    print(f"⚠️  Échec du téléchargement HF ({e2}). Repli sur le fichier local.")

        if not os.path.exists(local_path):
            raise FileNotFoundError(
                f"Aucun tokenizer trouvé : ni via HF Hub ('{hf_repo}'), "
                f"ni localement ('{local_path}'). Entraîne d'abord un tokenizer "
                f"BPE (cf. main.py) ou renseigne TOKENIZER_HF_REPO."
            )
        print(f"📂 Chargement du tokenizer local: '{local_path}'")
        return Tokenizer.from_file(local_path)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)

    def save_pretrained_json(self, path: str) -> None:
        """Sauvegarde au format tokenizer.json, pour push_to_hub."""
        self._tok.save(path)


# ============================================================================
# 5. DATASET — texte HF Hub prioritaire, repli fichier local
# ============================================================================

def load_training_text_from_hf(
    dataset_name: str,
    dataset_config: str,
    text_column: str,
    max_chars: int,
    streaming: bool,
) -> str:
    from datasets import load_dataset

    print(f"📥 Téléchargement de '{dataset_name}' (config='{dataset_config}', streaming={streaming})...")
    kwargs = {"split": "train"}
    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config, streaming=streaming, **kwargs)
    else:
        ds = load_dataset(dataset_name, streaming=streaming, **kwargs)

    chunks, total_chars = [], 0
    for row in ds:
        text = (row.get(text_column) or "").strip()
        if not text:
            continue
        chunks.append(text)
        total_chars += len(text)
        if total_chars >= max_chars:
            break

    full_text = "\n<|endoftext|>\n".join(chunks)
    print(f"✅ {len(chunks):,} extraits chargés, {len(full_text):,} caractères au total.")
    return full_text


def load_tinystories_text(dataset_name: str, column: str, max_stories: int) -> str:
    """Charge TinyStories-French (mêmes paramètres que main.py à l'origine).
    Retourne "" en cas d'échec plutôt que de lever — l'appelant décide s'il
    continue avec Wikipedia seul ou pas."""
    from datasets import load_dataset

    print(f"📥 Téléchargement de '{dataset_name}'...")
    try:
        ds = load_dataset(dataset_name, split="train")
    except Exception as e:
        print(f"⚠️  Échec du chargement de '{dataset_name}' ({e}) — ignoré.")
        return ""

    col = column if column in ds.column_names else ds.column_names[0]
    texts = [t for t in ds[col] if t]
    texts = texts[:max_stories] if max_stories and max_stories < len(texts) else texts
    text = "\n<|endoftext|>\n".join(texts)
    print(f"✅ {len(texts):,} histoires chargées depuis '{dataset_name}' ({len(text):,} caractères).")
    return text


def load_training_text(cfg: "Config", local_path: str) -> str:
    """Charge le corpus pour le mode "base" : TinyStories-French + Wikipedia
    combinés (même recette que main.py, pour ne pas dériver vers un style
    purement encyclopédique lors d'un entraînement/reprise via ce script).
    Repli sur le fichier local dans le dossier projet si les deux sources HF
    échouent ou si DATASET_HF_NAME est vide — pensé pour éviter tout
    téléversement manuel une fois le dossier Drive en place."""
    tinystories_text = ""
    if getattr(cfg, "TINYSTORIES_HF_NAME", ""):
        tinystories_text = load_tinystories_text(
            cfg.TINYSTORIES_HF_NAME, cfg.TINYSTORIES_COLUMN, cfg.TINYSTORIES_MAX_STORIES,
        )

    wiki_text = ""
    if cfg.DATASET_HF_NAME:
        try:
            wiki_text = load_training_text_from_hf(
                cfg.DATASET_HF_NAME,
                cfg.DATASET_HF_CONFIG,
                cfg.DATASET_TEXT_COLUMN,
                cfg.DATASET_MAX_CHARS,
                cfg.DATASET_STREAMING,
            )
        except Exception as e:
            print(f"⚠️  Échec du chargement HF ({e}).")

    if tinystories_text and wiki_text:
        combined = tinystories_text + "\n<|endoftext|>\n" + wiki_text
        print(f"📚 Corpus combiné: {len(tinystories_text):,} caractères TinyStories-French + "
              f"{len(wiki_text):,} caractères Wikipedia = {len(combined):,} caractères au total.")
        return combined
    if tinystories_text or wiki_text:
        print("⚠️  Une seule des deux sources HF a pu être chargée — poursuite avec elle seule.")
        return tinystories_text or wiki_text

    print("⚠️  Aucune des deux sources HF n'a pu être chargée. Repli sur le fichier local.")
    if os.path.exists(local_path):
        print(f"📂 Chargement du corpus local: '{local_path}'")
        with open(local_path, "r", encoding="utf-8") as f:
            return f.read()

    raise FileNotFoundError(
        "Aucune source de données disponible : ni TinyStories/Wikipedia HF, ni fichier "
        f"local '{local_path}'.\n"
        "  → Si ce chemin est RELATIF (ex: './training_corpus.txt' au lieu de "
        "'/content/drive/MyDrive/...'), c'est que Google Drive n'a pas été monté "
        "correctement : relance ce script avec `%run train_opsiom.py` dans une "
        "cellule du notebook (pas `!python train_opsiom.py`), pour garder l'accès "
        "au canal interactif nécessaire à l'authentification Drive.\n"
        "  → Sinon, renseigne DATASET_HF_NAME/TINYSTORIES_HF_NAME ou dépose un "
        "fichier texte à cet emplacement dans ton dossier projet."
    )


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, token_ids: list[int], block_size: int):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx: int):
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1: idx + 1 + self.block_size]
        return x, y


# ============================================================================
# 5bis. DONNÉES DE DIALOGUE (mode "chat") — chargées depuis le Hub HF
# ============================================================================
#
# Format d'exemple pour chaque paire (question, réponse) :
#   {USER_TAG}\n{question}\n{ASSISTANT_TAG}\n{réponse}\n{EOT_TOKEN}
# La loss est masquée (-1 = ignore_index) sur tout ce qui précède la réponse
# de l'assistant : le modèle n'apprend qu'à produire la réponse, jamais à
# "prédire" la question de l'utilisateur — même logique que train_chat.py.

def load_chat_dataset_from_hf(
    dataset_name: str,
    sample_size: int,
    max_answer_chars: int,
    instruction_col: str = "instruction",
    input_col: str = "input",
    output_col: str = "output",
) -> list[tuple[str, str]]:
    """Charge UN dataset d'instructions français depuis le Hub HF, au format
    Alpaca (instruction/input/output -> question/réponse). Les noms de colonnes
    sont paramétrables car certains datasets (ex: tbboukhari/Alpaca-in-french,
    colonnes "saisir"/"sortir") ne suivent pas la convention anglaise. Retourne
    [] en cas d'échec (dataset introuvable, colonnes inattendues, etc.)."""
    from datasets import load_dataset

    print(f"📥 Téléchargement de '{dataset_name}' (sous-échantillon: {sample_size or 'tout'})...")
    try:
        ds = load_dataset(dataset_name, split="train")
    except Exception as e:
        print(f"⚠️  Échec du chargement de '{dataset_name}' ({e}) — ignoré.")
        return []
    ds = ds.shuffle(seed=1337)

    # Garde-fou : si les colonnes configurées n'existent pas dans le dataset
    # réellement chargé, on le dit explicitement plutôt que de retourner 0
    # paires en silence (ce qui s'est déjà produit : les noms de colonnes
    # affichés sur la page HF du dataset ne correspondent pas toujours
    # exactement à ceux que `datasets.load_dataset()` expose).
    missing = [c for c in (instruction_col, output_col) if c not in ds.column_names]
    if missing:
        print(
            f"⚠️  Colonnes {missing} introuvables dans '{dataset_name}'. "
            f"Colonnes réellement présentes : {ds.column_names}. "
            f"Corrige instruction_col/input_col/output_col dans Config.CHAT_HF_DATASETS "
            f"pour ce dataset — ignoré pour cette fois."
        )
        return []

    pairs = []
    for row in ds:
        instruction = (row.get(instruction_col) or "").strip()
        extra_input = (row.get(input_col) or "").strip()
        answer = (row.get(output_col) or "").strip()
        if not instruction or not answer:
            continue
        if len(answer) > max_answer_chars:
            continue
        question = f"{instruction}\n\n{extra_input}" if extra_input else instruction
        pairs.append((question, answer))
        if sample_size and len(pairs) >= sample_size:
            break

    print(f"✅ {len(pairs):,} paires question/réponse retenues depuis '{dataset_name}'.")
    return pairs


def load_oasst1_french(sample_size: int, max_answer_chars: int) -> list[tuple[str, str]]:
    """Charge le sous-ensemble FRANÇAIS d'OpenAssistant/oasst1 et reconstruit des
    paires (question, réponse) à partir de l'arbre de conversation.

    Schéma réel confirmé (table plate, PAS du format Alpaca) :
    ['message_id', 'parent_id', 'user_id', 'created_date', 'text', 'role',
     'lang', 'review_count', 'review_result', 'deleted', 'rank', 'synthetic',
     'model_name', 'detoxify', 'message_tree_id', 'tree_state', 'emojis', 'labels']
    role vaut "prompter" (utilisateur) ou "assistant". On reconstruit les paires
    en reliant chaque message assistant en français à son message parent
    (prompter, en français aussi) via parent_id/message_id.

    Intérêt par rapport aux datasets Alpaca déjà utilisés : du vrai dialogue
    conversationnel humain, pas uniquement des instructions factuelles courtes.
    """
    from datasets import load_dataset

    print(f"📥 Téléchargement de 'OpenAssistant/oasst1' (sous-échantillon: {sample_size or 'tout'})...")
    try:
        ds = load_dataset("OpenAssistant/oasst1", split="train")
    except Exception as e:
        print(f"⚠️  Échec du chargement de 'OpenAssistant/oasst1' ({e}) — ignoré.")
        return []

    by_id = {row["message_id"]: row for row in ds}

    pairs = []
    for row in ds:
        if row.get("role") != "assistant" or row.get("lang") != "fr" or row.get("deleted"):
            continue
        parent = by_id.get(row.get("parent_id"))
        if not parent or parent.get("role") != "prompter" or parent.get("lang") != "fr" or parent.get("deleted"):
            continue
        question = (parent.get("text") or "").strip()
        answer = (row.get("text") or "").strip()
        if not question or not answer or len(answer) > max_answer_chars:
            continue
        pairs.append((question, answer))
        if sample_size and len(pairs) >= sample_size:
            break

    print(f"✅ {len(pairs):,} paires question/réponse retenues depuis 'OpenAssistant/oasst1' (fr).")
    return pairs


def load_chat_datasets_from_hf(dataset_specs: list[dict], max_answer_chars: int) -> list[tuple[str, str]]:
    """Charge et concatène PLUSIEURS datasets HF (voir Config.CHAT_HF_DATASETS).
    Chaque entrée précise "loader": "alpaca" (par défaut, colonnes instruction/
    input/output paramétrables) ou "oasst1" (arbre de conversation, colonnes
    fixes). Un dataset qui échoue est simplement ignoré, pour ne pas bloquer
    l'entraînement si l'un d'eux est indisponible ou a été renommé/supprimé."""
    all_pairs: list[tuple[str, str]] = []
    for spec in dataset_specs:
        loader = spec.get("loader", "alpaca")
        if loader == "oasst1":
            pairs = load_oasst1_french(spec.get("sample_size", 0), max_answer_chars)
        else:
            pairs = load_chat_dataset_from_hf(
                spec["name"],
                spec.get("sample_size", 0),
                max_answer_chars,
                instruction_col=spec.get("instruction_col", "instruction"),
                input_col=spec.get("input_col", "input"),
                output_col=spec.get("output_col", "output"),
            )
        all_pairs.extend(pairs)
    print(f"📚 Total combiné: {len(all_pairs):,} paires question/réponse depuis {len(dataset_specs)} dataset(s).")
    return all_pairs


def build_story_jsonl_from_corpus(
    corpus_paths: list[str],
    output_jsonl_path: str,
    prompts: list[str] | None = None,
) -> None:
    """Convertit ta propre prose littéraire (fichiers .txt séparés par ligne
    vide = un texte par paragraphe/histoire) en exemples de dialogue JSONL
    compatibles avec CHAT_LOCAL_JSONL, en les associant à des prompts génériques
    de type "raconte-moi une histoire". Comble le vrai trou du fine-tuning
    actuel : French-Alpaca / Alpaca-cleaned-fr sont des Q/R factuelles courtes,
    quasi sans récit — d'où les complétions faibles sur "Raconte-moi une
    histoire" / "Il était une fois" malgré l'ajout d'un 2e dataset Alpaca.

    N'écrase jamais un fichier CHAT_LOCAL_JSONL existant : ajoute (append).
    """
    import json

    if prompts is None:
        prompts = [
            "Raconte-moi une histoire.",
            "Il était une fois",
            "Écris-moi un petit conte.",
            "Peux-tu inventer une histoire courte ?",
            "Raconte-moi quelque chose d'imaginaire.",
        ]

    texts: list[str] = []
    for path in corpus_paths:
        if not os.path.exists(path):
            print(f"⚠️  Corpus introuvable, ignoré : '{path}'")
            continue
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        # Un texte par bloc séparé par une ligne vide — adapte le séparateur si
        # ta prose est structurée différemment (ex: un fichier par histoire).
        for block in raw.split("\n\n"):
            block = block.strip()
            if len(block) >= 50:  # ignore les fragments trop courts pour être une vraie histoire
                texts.append(block)

    if not texts:
        print("⚠️  Aucun texte exploitable trouvé dans corpus_paths — rien à écrire.")
        return

    random.shuffle(texts)
    with open(output_jsonl_path, "a", encoding="utf-8") as f:
        for i, text in enumerate(texts):
            prompt = prompts[i % len(prompts)]
            f.write(json.dumps({"user": prompt, "assistant": text}, ensure_ascii=False) + "\n")

    print(f"✅ {len(texts):,} exemples narratifs ajoutés à '{output_jsonl_path}' "
          f"(à partir de {len(corpus_paths)} fichier(s) corpus).")


def load_local_jsonl_dialogues(path: str) -> list[tuple[str, str]]:
    """Charge des exemples additionnels depuis un fichier JSONL local (optionnel),
    une paire par ligne : {"user": "...", "assistant": "..."}."""
    if not os.path.exists(path):
        return []
    import json
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                pairs.append((row["user"], row["assistant"]))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠️  Ligne {line_num} de '{path}' ignorée ({e}).")
    if pairs:
        print(f"📥 {len(pairs)} exemples additionnels chargés depuis '{path}'.")
    return pairs


class ChatExample:
    __slots__ = ("input_ids", "target_ids")

    def __init__(self, input_ids: list[int], target_ids: list[int]):
        self.input_ids = input_ids
        self.target_ids = target_ids


def build_chat_examples(
    dialogues: list[tuple[str, str]],
    tokenizer: "FrenchTokenizerWrapper",
    max_seq_len: int,
    user_tag: str,
    assistant_tag: str,
    eot_token: str,
) -> list[ChatExample]:
    """Encode chaque paire (question, réponse) avec les tags de rôle, et masque
    la loss (-1) sur tout ce qui précède la réponse de l'assistant."""
    examples = []
    skipped = 0

    for question, answer in dialogues:
        prefix_text = f"{user_tag}\n{question}\n{assistant_tag}\n"
        full_text = prefix_text + answer + "\n" + eot_token

        prefix_ids = tokenizer.encode(prefix_text)
        full_ids = tokenizer.encode(full_text)

        # Garde-fou : le préfixe doit être un vrai préfixe token-à-token de la
        # séquence complète (vrai avec un tokenizer BPE byte-level tant qu'on
        # encode le même texte de départ à l'identique).
        if full_ids[: len(prefix_ids)] != prefix_ids:
            skipped += 1
            continue
        if len(full_ids) < 2 or len(full_ids) > max_seq_len:
            skipped += 1
            continue

        x = full_ids[:-1]
        y = full_ids[1:]
        mask_upto = max(0, len(prefix_ids) - 1)
        y = [-1] * mask_upto + y[mask_upto:]
        examples.append(ChatExample(x, y))

    if skipped:
        print(f"⚠️  {skipped} exemple(s) ignoré(s) (trop long ou tag mal aligné).")
    return examples


def collate_chat_batch(batch: list[ChatExample], pad_id: int, device: str):
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


# ============================================================================
# 6. GOOGLE DRIVE — montage + gestion des checkpoints
# ============================================================================

def maybe_mount_drive(cfg: "Config") -> str:
    """Monte Google Drive si demandé et renvoie le dossier projet à utiliser
    comme racine pour TOUS les fichiers (tokenizer, corpus, checkpoints) :
    `cfg.PROJECT_DIR` si Drive est monté avec succès, sinon le dossier local
    courant (utile si le script tourne hors Colab).

    IMPORTANT : ce script doit être exécuté DANS le noyau du notebook (ex. via
    `%run script.py` ou collé dans une cellule), PAS via `!python script.py`.
    Un `!python ...` lance un sous-processus qui n'a pas accès au canal
    interactif du notebook, ce dont `drive.mount()` a besoin pour
    l'authentification — le montage échouerait silencieusement dans ce cas."""
    if not cfg.USE_GOOGLE_DRIVE:
        return "."

    mydrive_path = os.path.join(cfg.DRIVE_MOUNT_POINT, "MyDrive")
    if os.path.isdir(mydrive_path):
        # Déjà monté (par cette exécution ou une cellule précédente) — inutile
        # de rappeler drive.mount(), qui pourrait redemander une confirmation.
        print(f"✅ Google Drive déjà monté sur '{cfg.DRIVE_MOUNT_POINT}'.")
    else:
        try:
            from google.colab import drive  # type: ignore
            print(f"💾 Montage de Google Drive sur '{cfg.DRIVE_MOUNT_POINT}'...")
            drive.mount(cfg.DRIVE_MOUNT_POINT)
        except ImportError:
            print("ℹ️  Pas dans Colab (module google.colab introuvable) — "
                  "utilisation du dossier local courant, Drive ignoré.")
            return "."
        except Exception as e:
            import traceback
            print(f"⚠️  Échec du montage Drive : {e}")
            traceback.print_exc()
            print("   ↳ Vérifie que ce script tourne bien DANS le notebook "
                  "(ex: `%run train_opsiom.py`) et pas via `!python train_opsiom.py`, "
                  "qui coupe l'accès au canal interactif nécessaire à l'authentification.")
            return "."

    if not os.path.isdir(cfg.PROJECT_DIR):
        print(f"⚠️  Dossier projet introuvable: '{cfg.PROJECT_DIR}' — création.")
        os.makedirs(cfg.PROJECT_DIR, exist_ok=True)
    print(f"✅ Dossier projet utilisé: '{cfg.PROJECT_DIR}'")
    return cfg.PROJECT_DIR


# ============================================================================
# 7. TRAINER — AdamW + Warmup/Cosine + AMP + Grad Clipping
# ============================================================================

class LLMTrainer:
    def __init__(self, model: ModernLLM, args: ModelArgs, dataset_text: str,
                 tokenizer: FrenchTokenizerWrapper, cfg: "Config", resumed: bool = False):
        self.model = model.to(args.device)
        self.args = args
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.resumed = resumed

        # En reprise : pic de LR et warmup réduits (voir commentaire sur
        # RESUME_LR_SCALE dans Config) pour ne pas percuter un checkpoint déjà
        # entraîné avec un LR pensé pour un démarrage from scratch.
        self.effective_max_lr = cfg.MAX_LR * cfg.RESUME_LR_SCALE if resumed else cfg.MAX_LR
        self.effective_warmup_steps = cfg.RESUME_WARMUP_STEPS if resumed else cfg.WARMUP_STEPS
        if resumed:
            print(f"   ↳ 🔧 Reprise détectée : LR de pic réduit à {self.effective_max_lr:.2e} "
                  f"(au lieu de {cfg.MAX_LR:.2e}), warmup réduit à {self.effective_warmup_steps} steps.")

        token_ids = tokenizer.encode(dataset_text)
        print(f"🔤 Corpus tokenisé: {len(token_ids):,} tokens.")
        if len(token_ids) < args.max_seq_len * 4:
            print("⚠️  Corpus très petit par rapport à max_seq_len — "
                  "envisage un dataset plus grand ou un max_seq_len plus court.")

        split_idx = int(len(token_ids) * (1 - cfg.VAL_FRACTION))
        train_ids = token_ids[:split_idx]
        val_ids = token_ids[split_idx:]

        self.train_dataset = TextDataset(train_ids, block_size=args.max_seq_len)
        self.val_dataset = (
            TextDataset(val_ids, block_size=args.max_seq_len)
            if len(val_ids) > args.max_seq_len + 1
            else self.train_dataset
        )

        decay_params, no_decay_params, seen = [], [], set()
        for name, p in self.model.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            (decay_params if p.dim() >= 2 else no_decay_params).append(p)

        optim_groups = [
            {"params": decay_params, "weight_decay": 0.1},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and args.device == "cuda"
        self.optimizer = torch.optim.AdamW(
            optim_groups, lr=self.effective_max_lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused,
        )

        self.amp_dtype = (
            torch.bfloat16 if (args.device == "cuda" and torch.cuda.is_bf16_supported())
            else torch.float16
        )
        self.scaler = torch.amp.GradScaler(enabled=(self.amp_dtype == torch.float16 and args.device == "cuda"))

    def get_lr(self, step: int) -> float:
        cfg = self.cfg
        max_lr = self.effective_max_lr
        warmup_steps = self.effective_warmup_steps
        if step < warmup_steps:
            return max_lr * (step + 1) / warmup_steps
        if step >= cfg.MAX_STEPS:
            return cfg.MIN_LR
        decay_ratio = min(max((step - warmup_steps) / max(1, (cfg.MAX_STEPS - warmup_steps)), 0.0), 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return cfg.MIN_LR + coeff * (max_lr - cfg.MIN_LR)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        self.model.train()
        x = x.to(self.args.device, non_blocking=True)
        y = y.to(self.args.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type="cuda" if self.args.device == "cuda" else "cpu",
            dtype=self.amp_dtype,
            enabled=(self.args.device == "cuda"),
        ):
            logits, loss = self.model(x, targets=y)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRAD_CLIP)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        losses = []
        n_batches = self.cfg.EVAL_BATCHES
        batch_size = self.cfg.BATCH_SIZE
        for _ in range(n_batches):
            n = min(batch_size, max(1, len(self.val_dataset)))
            idxs = [random.randint(0, len(self.val_dataset) - 1) for _ in range(n)]
            xb = torch.stack([self.val_dataset[i][0] for i in idxs])
            yb = torch.stack([self.val_dataset[i][1] for i in idxs])
            xb, yb = xb.to(self.args.device), yb.to(self.args.device)
            _, loss = self.model(xb, targets=yb)
            losses.append(loss.item())
        self.model.train()
        return sum(losses) / len(losses)


# ============================================================================
# 8. SAUVEGARDE — format 100% compatible avec test_IA_2.py
# ============================================================================

def save_checkpoint(model: ModernLLM, args: ModelArgs, step: int, val_loss: float, path: str) -> None:
    """Sauvegarde au format attendu par la détection automatique de
    test_IA_2.py : dict avec 'model_state_dict', 'args', 'val_loss', 'step'."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": args,
            "val_loss": val_loss,
            "step": step,
        },
        path,
    )


def maybe_push_to_hub(cfg: "Config", checkpoint_path: str, tokenizer: FrenchTokenizerWrapper) -> None:
    """Pousse le checkpoint .pt et le tokenizer.json sur un repo HF Hub perso.
    Nécessite d'être connecté (huggingface_hub.login() ou variable d'env HF_TOKEN)."""
    if not cfg.PUSH_TO_HUB:
        return
    if not cfg.HUB_REPO_ID:
        print("⚠️  PUSH_TO_HUB=True mais HUB_REPO_ID est vide — push ignoré.")
        return

    try:
        from huggingface_hub import HfApi, create_repo

        print(f"📤 Push vers le Hub HF: '{cfg.HUB_REPO_ID}' (private={cfg.HUB_PRIVATE})...")
        create_repo(cfg.HUB_REPO_ID, private=cfg.HUB_PRIVATE, exist_ok=True)

        tokenizer_tmp_path = "tokenizer.json"
        tokenizer.save_pretrained_json(tokenizer_tmp_path)

        api = HfApi()
        api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo=os.path.basename(checkpoint_path),
            repo_id=cfg.HUB_REPO_ID,
        )
        api.upload_file(
            path_or_fileobj=tokenizer_tmp_path,
            path_in_repo="tokenizer.json",
            repo_id=cfg.HUB_REPO_ID,
        )
        print(f"✅ Modèle et tokenizer poussés sur https://huggingface.co/{cfg.HUB_REPO_ID}")
    except Exception as e:
        print(f"⚠️  Échec du push vers le Hub HF ({e}). Le checkpoint reste disponible en local/Drive.")


# ============================================================================
# 9. POINT D'ENTRÉE
# ============================================================================

def setup_common(cfg: "Config"):
    """Étapes communes aux deux modes : montage Drive, tokenizer, résolution
    des chemins. Renvoie (project_dir, tokenizer, checkpoint_path_base)."""
    project_dir = maybe_mount_drive(cfg)
    checkpoint_path = os.path.join(project_dir, cfg.CHECKPOINT_FILENAME)
    tokenizer_local_path = os.path.join(project_dir, cfg.TOKENIZER_LOCAL_PATH)

    tokenizer = FrenchTokenizerWrapper(
        hf_repo=cfg.TOKENIZER_HF_REPO,
        local_path=tokenizer_local_path,
    )
    print(f"✅ Tokenizer prêt (vocab_size={tokenizer.vocab_size:,})")
    return project_dir, tokenizer, checkpoint_path


def load_or_init_base_model(cfg: "Config", tokenizer: "FrenchTokenizerWrapper", checkpoint_path: str):
    """Charge le modèle depuis checkpoint_path si présent (en utilisant SA
    config sauvegardée pour éviter tout mismatch de dimensions), sinon
    instancie un nouveau modèle selon le preset MODEL_SIZE."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resumed = False

    if os.path.exists(checkpoint_path):
        print(f"♻️  Checkpoint existant trouvé ('{checkpoint_path}')...")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        saved_args = ckpt.get("args") if isinstance(ckpt, dict) else None

        if saved_args is not None and hasattr(saved_args, "vocab_size"):
            args = saved_args
            args.device = device
            print(
                f"   ↳ Reprise avec la config DU CHECKPOINT (ignorant MODEL_SIZE='{cfg.MODEL_SIZE}'): "
                f"dim={args.dim}, layers={args.n_layers}, heads={args.n_heads}, kv_heads={args.n_kv_heads}, "
                f"max_seq_len={args.max_seq_len}"
            )
        else:
            args = build_model_args(cfg.MODEL_SIZE, vocab_size=tokenizer.vocab_size)
            print("   ⚠️  Le checkpoint ne contient pas d'objet 'args' — reprise avec le "
                  f"preset '{cfg.MODEL_SIZE}'. Si ça plante, le mismatch vient d'ici.")

        model = ModernLLM(args).to(args.device)
        model.load_state_dict(ckpt["model_state_dict"])
        resumed = True
        print(f"   ↳ step précédent: {ckpt.get('step', '?')}, val_loss précédent: {ckpt.get('val_loss', 'N/A')}")
    else:
        args = build_model_args(cfg.MODEL_SIZE, vocab_size=tokenizer.vocab_size)
        model = ModernLLM(args).to(args.device)

    n_params = model.num_params()
    print(
        f"✅ Modèle {'repris' if resumed else 'instancié'} sur '{args.device}' avec {n_params / 1e6:.2f}M paramètres "
        f"(dim={args.dim}, layers={args.n_layers}, heads={args.n_heads}, "
        f"kv_heads={args.n_kv_heads}, max_seq_len={args.max_seq_len})"
    )
    return model, args, resumed


def run_base_training(cfg: "Config") -> None:
    """Mode 'base' : pré-entraînement next-token sur texte brut continu."""
    project_dir, tokenizer, checkpoint_path = setup_common(cfg)
    corpus_local_path = os.path.join(project_dir, cfg.LOCAL_CORPUS_FALLBACK)

    model, args, resumed = load_or_init_base_model(cfg, tokenizer, checkpoint_path)

    dataset_text = load_training_text(cfg, local_path=corpus_local_path)

    trainer = LLMTrainer(model, args, dataset_text, tokenizer, cfg, resumed=resumed)
    print(f"📚 Fenêtres d'entraînement: {len(trainer.train_dataset):,} | validation: {len(trainer.val_dataset):,}")

    loader = torch.utils.data.DataLoader(
        trainer.train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True,
    )

    def infinite_loader(dl):
        while True:
            for batch in dl:
                yield batch

    data_iter = infinite_loader(loader)

    best_val_loss = float("inf")
    t_start = time.time()
    print(f"\n🏋️ Entraînement (base) pour {cfg.MAX_STEPS} steps...\n")

    for step in range(cfg.MAX_STEPS):
        lr = trainer.get_lr(step)
        for group in trainer.optimizer.param_groups:
            group["lr"] = lr

        x_batch, y_batch = next(data_iter)
        train_loss = trainer.train_step(x_batch, y_batch)

        if step % 20 == 0 or step == cfg.MAX_STEPS - 1:
            elapsed = time.time() - t_start
            print(f"step {step:04d}/{cfg.MAX_STEPS} | lr {lr:.2e} | train_loss {train_loss:.4f} | {elapsed:.0f}s")

        if step % cfg.EVAL_INTERVAL == 0 or step == cfg.MAX_STEPS - 1:
            val_loss = trainer.evaluate()
            print(f"   ↳ 📊 val_loss {val_loss:.4f} (best: {best_val_loss:.4f})")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, args, step, val_loss, checkpoint_path)
                print(f"   ↳ 💾 Nouveau meilleur modèle sauvegardé ('{checkpoint_path}')")

        if step % cfg.GEN_INTERVAL == 0 and step > 0:
            sample = model.generate(
                prompt="Il était une fois",
                tokenizer=tokenizer,
                max_new_tokens=40,
                temperature=0.8,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.3,
            )
            print(f"   ↳ 📝 Échantillon: {sample!r}\n")

    print(f"\n✅ Entraînement (base) terminé en {(time.time() - t_start) / 60:.1f} min.")
    print(f"📂 Meilleur checkpoint: '{checkpoint_path}' (val_loss={best_val_loss:.4f})")

    maybe_push_to_hub(cfg, checkpoint_path, tokenizer)


def run_chat_training(cfg: "Config") -> None:
    """Mode 'chat' : fine-tuning conversationnel. Repart OBLIGATOIREMENT du
    checkpoint de base déjà entraîné (cfg.CHECKPOINT_FILENAME), l'entraîne sur
    des paires question/réponse au format USER_TAG/ASSISTANT_TAG avec loss
    masquée sur la question, et sauvegarde dans un fichier séparé
    (cfg.CHAT_CHECKPOINT_FILENAME) pour ne jamais écraser le modèle de base."""
    project_dir, tokenizer, base_checkpoint_path = setup_common(cfg)
    chat_checkpoint_path = os.path.join(project_dir, cfg.CHAT_CHECKPOINT_FILENAME)
    dialogues_jsonl_path = os.path.join(project_dir, cfg.CHAT_LOCAL_JSONL)

    if not os.path.exists(base_checkpoint_path):
        raise FileNotFoundError(
            f"Le fine-tuning chat nécessite un modèle de base déjà entraîné "
            f"('{base_checkpoint_path}' introuvable). Lance d'abord un "
            f"entraînement avec TRAINING_MODE='base'."
        )

    # Repart toujours du checkpoint de base existant (jamais d'un chat_model.pt
    # partiel), pour un point de départ cohérent et reproductible.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(base_checkpoint_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args")
    if saved_args is None or not hasattr(saved_args, "vocab_size"):
        raise ValueError(
            f"Le checkpoint de base '{base_checkpoint_path}' ne contient pas "
            "d'objet 'args' exploitable — impossible de reconstruire l'architecture."
        )
    args = saved_args
    args.device = device
    model = ModernLLM(args).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    n_params = model.num_params()
    print(
        f"✅ Modèle de base chargé depuis '{base_checkpoint_path}' ({n_params / 1e6:.2f}M paramètres, "
        f"dim={args.dim}, layers={args.n_layers}, max_seq_len={args.max_seq_len})"
    )

    # --- Corpus de dialogues (datasets HF configurés + JSONL local optionnel) ---
    dialogues = load_chat_datasets_from_hf(
        cfg.CHAT_HF_DATASETS, cfg.CHAT_MAX_ANSWER_CHARS,
    ) + load_local_jsonl_dialogues(dialogues_jsonl_path)
    random.shuffle(dialogues)

    examples = build_chat_examples(
        dialogues, tokenizer, max_seq_len=args.max_seq_len,
        user_tag=cfg.USER_TAG, assistant_tag=cfg.ASSISTANT_TAG, eot_token=cfg.EOT_TOKEN,
    )
    print(f"📚 {len(examples)} exemples de dialogue prêts pour l'entraînement "
          f"({len(dialogues)} paires question/réponse au total).")
    if len(examples) < 5:
        raise RuntimeError(
            "Pas assez d'exemples de dialogue valides pour entraîner. "
            "Vérifie CHAT_HF_DATASETS et/ou ajoute des paires dans "
            f"'{dialogues_jsonl_path}'."
        )

    n_val = max(1, len(examples) // 10)
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.CHAT_LR, betas=(0.9, 0.95), weight_decay=0.01)
    pad_id = tokenizer.eos_token_id

    def run_epoch(data: list[ChatExample], train: bool) -> float:
        model.train(train)
        if train:
            random.shuffle(data)
        losses = []
        for i in range(0, len(data), cfg.CHAT_BATCH_SIZE):
            batch = data[i:i + cfg.CHAT_BATCH_SIZE]
            x, y = collate_chat_batch(batch, pad_id, device)
            with torch.set_grad_enabled(train):
                _, loss = model(x, targets=y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            losses.append(loss.item())
        return sum(losses) / max(1, len(losses))

    print(f"\n🏋️ Fine-tuning chat pour {cfg.CHAT_EPOCHS} epochs "
          f"(lr={cfg.CHAT_LR}, batch_size={cfg.CHAT_BATCH_SIZE})...\n")
    best_val_loss = float("inf")
    t_start = time.time()

    for epoch in range(1, cfg.CHAT_EPOCHS + 1):
        train_loss = run_epoch(train_examples, train=True)
        val_loss = run_epoch(val_examples, train=False)
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, args, epoch, val_loss, chat_checkpoint_path)
            marker = "  💾 meilleur modèle sauvegardé"
        print(f"epoch {epoch:02d}/{cfg.CHAT_EPOCHS} | train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f}{marker}")

        sample_prompt = f"{cfg.USER_TAG}\nBonjour, comment tu vas ?\n{cfg.ASSISTANT_TAG}\n"
        sample = model.generate(
            prompt=sample_prompt, tokenizer=tokenizer, max_new_tokens=40,
            temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.3,
        )
        print(f"   ↳ 📝 Échantillon: {sample[len(sample_prompt):]!r}\n")

    print(f"\n✅ Fine-tuning chat terminé en {(time.time() - t_start) / 60:.1f} min "
          f"(meilleur val_loss={best_val_loss:.4f})")
    print(f"📂 Checkpoint chat sauvegardé: '{chat_checkpoint_path}'")

    maybe_push_to_hub(cfg, chat_checkpoint_path, tokenizer)


def main():
    cfg = Config()
    torch.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)

    print("=" * 70)
    print(f"🚀 OPSIOM — Entraînement (mode: {cfg.TRAINING_MODE})")
    print("=" * 70)

    if cfg.TRAINING_MODE == "chat":
        run_chat_training(cfg)
        print("\n⚠️  RAPPEL IMPORTANT : pour que la génération s'arrête correctement")
        print(f"   à l'inférence, mets à jour dans test_IA_2.py :")
        print(f'     USER_TAG = "{cfg.USER_TAG}"')
        print(f'     ASSISTANT_TAG = "{cfg.ASSISTANT_TAG}"')
        print("   (dans la classe ChatFormatter), et charge 'chat_model.pt' au lieu de 'best_model.pt'.")
    elif cfg.TRAINING_MODE == "base":
        run_base_training(cfg)
    else:
        raise ValueError(f"TRAINING_MODE inconnu: '{cfg.TRAINING_MODE}' (attendu: 'base' ou 'chat')")


if __name__ == "__main__":
    main()
