# ============================================================================
# Mini-LLM style LLaMA/Qwen — script complet prêt pour une cellule Colab
# RMSNorm + RoPE + SwiGLU + Attention SDPA (FlashAttention) + KV-Cache
# ============================================================================

# --- Dépendances (décommente si tu es sur Colab, sinon `pip install` en local) ---
# %pip install -q datasets tokenizers torch

import os
import re
import math
import inspect
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ============================================================================
# Google Drive — sauvegarde directe du tokenizer et des checkpoints
# ============================================================================
# Sur Colab, monte le Drive et fait pointer TOKENIZER_CACHE_PATH / CHECKPOINT_PATH
# dedans, pour que le tokenizer entraîné et les poids du modèle survivent à la
# fermeture de la session (au lieu de finir dans le stockage éphémère de la VM).
# En dehors de Colab (exécution locale), on retombe simplement sur le
# répertoire courant.

DRIVE_SAVE_DIR = "/content/drive/MyDrive/mini_llm_fr"  # dossier créé automatiquement s'il n'existe pas


def _mount_drive_and_get_save_dir(save_dir: str = DRIVE_SAVE_DIR) -> str:
    """Monte Google Drive si on est sur Colab et renvoie le dossier de sauvegarde
    à utiliser pour le tokenizer et les checkpoints. Hors Colab, renvoie "."
    (répertoire courant) sans rien monter."""
    try:
        from google.colab import drive  # disponible uniquement sur Colab
    except ImportError:
        print("ℹ️ Pas sur Colab (module 'google.colab' introuvable) — "
              "sauvegarde en local dans le répertoire courant.")
        return "."

    print("📎 Montage de Google Drive...")
    drive.mount("/content/drive")
    os.makedirs(save_dir, exist_ok=True)
    print(f"✅ Google Drive monté — tokenizer et checkpoints sauvegardés dans {save_dir}")
    return save_dir


_SAVE_DIR = _mount_drive_and_get_save_dir()


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ModelArgs:
    """Configuration hyperparamétrique pour architecture Transformer type LLaMA/Qwen."""
    vocab_size: int = 16000   # Valeur par défaut — écrasée dynamiquement par la taille
                               # réelle du vocabulaire du tokenizer français entraîné plus bas
    dim: int = 448            # Dimension des embeddings — augmentée par rapport à la version
                               # gpt2 (384) car un vocab français compact (16k vs 50k) libère
                               # du budget de paramètres pour les couches Transformer
    n_layers: int = 8         # Nombre de blocs Transformer
    n_heads: int = 8          # Nombre de têtes d'attention (Query)
    n_kv_heads: int | None = 4  # Grouped-Query Attention (GQA): 4 têtes K/V pour 8 têtes Q
    max_seq_len: int = 256    # Fenêtre de contexte
    dropout: float = 0.1
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# Hyperparamètres d'entraînement — modifie-les librement
N_STORIES = 20000          # Plafond d'histoires TinyStories-French utilisées (le dataset n'en
                            # contient qu'environ 1000 au total, donc en pratique tout est utilisé)
TOKENIZER_VOCAB_SIZE = 16000   # Taille cible du vocabulaire du tokenizer BPE français
TOKENIZER_CACHE_PATH = os.path.join(_SAVE_DIR, "fr_bpe_tokenizer.json")
WIKI_CONFIG = "wikitext-72"    # Plus grande des deux configs d'asi/wikitext_fr (quality + good articles)
WIKI_MAX_CHARS = 25_000_000    # Plafond de caractères Wikipedia chargés (tokenizer + corpus LM)
# MAX_STEPS: 1000 steps à batch=32/block=256 ne couvre qu'~1 epoch sur le corpus
# (TinyStories-FR + Wikipedia ≈ 5-6M tokens). Pour un modèle de ~26M paramètres,
# c'est trop peu pour stabiliser les statistiques de sous-mots (cause principale
# des artefacts type "dçant", "garès êtreux"). On vise ~15-20 epochs sur les
# données disponibles plutôt qu'un budget de compute abstrait — ajustez à la
# hausse si votre val_loss continue de baisser à la fin de l'entraînement.
MAX_STEPS = 8000            # ~8-10 epochs sur le corpus combiné (était 1000)
WARMUP_STEPS = 400           # gardé à 5% de MAX_STEPS, comme avant
BATCH_SIZE = 32
MAX_LR = 3e-4
MIN_LR = 3e-5
EVAL_INTERVAL = 200        # Évaluation + sauvegarde du meilleur modèle tous les N steps
GEN_INTERVAL = 400         # Génération d'un échantillon de contrôle tous les N steps
CHECKPOINT_PATH = os.path.join(_SAVE_DIR, "best_model.pt")
SEED = 1337

# Reprend l'entraînement depuis CHECKPOINT_PATH s'il existe, au lieu de
# repartir de poids aléatoires. Pratique pour étendre un entraînement déjà
# fait (ex: vous aviez tourné 1000 steps, vous voulez continuer) sans perdre
# ce qui a déjà été appris. Le tokenizer/vocab doit être identique.
RESUME_FROM_CHECKPOINT = True


# ============================================================================
# Normalization
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (plus rapide et stable que LayerNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # (dim,) gain appris

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # (B, T, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calcul en float32 pour la stabilité numérique, recast dans le dtype d'origine
        out = self._norm(x.float()).type_as(x)  # (B, T, C)
        return out * self.weight  # (B, T, C) * (C,) broadcast


# ============================================================================
# Rotary Position Embeddings (RoPE)
# ============================================================================

class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) — précalcule cos/sin pour toutes les positions."""

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "head_dim doit être pair pour RoPE"
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))  # (dim/2,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len).float()  # (max_seq_len,)
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (max_seq_len, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)  # (max_seq_len, dim)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)  # (max_seq_len, dim)

    def forward(self, x: torch.Tensor, seq_len: int, start_pos: int = 0):
        # Positions [start_pos, start_pos + seq_len) — essentiel pour le décodage avec KV-Cache
        cos = self.cos_cached[start_pos:start_pos + seq_len].to(dtype=x.dtype, device=x.device)  # (T, head_dim)
        sin = self.sin_cached[start_pos:start_pos + seq_len].to(dtype=x.dtype, device=x.device)  # (T, head_dim)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Fait pivoter la moitié des dimensions: [-x2, x1] où x = [x1, x2]."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Applique la rotation RoPE sur les Query et Key tensors.

    xq, xk: (B, n_heads, T, head_dim) ; cos, sin: (T, head_dim)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    xq_rotated = (xq * cos) + (rotate_half(xq) * sin)  # (B, n_heads, T, head_dim)
    xk_rotated = (xk * cos) + (rotate_half(xk) * sin)  # (B, n_kv_heads, T, head_dim)
    return xq_rotated, xk_rotated


# ============================================================================
# SwiGLU FeedForward
# ============================================================================

class SwiGLUFeedForward(nn.Module):
    """Couche MLP SwiGLU (Gated Linear Unit avec SiLU) utilisée dans LLaMA/Qwen."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = int(8 * args.dim / 3)
        multiple_of = 256
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)  # down
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)  # up
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w1(x))     # (B, T, hidden_dim)
        up = self.w3(x)               # (B, T, hidden_dim)
        out = self.w2(gate * up)      # (B, T, dim)
        return self.dropout(out)


# ============================================================================
# Attention (SDPA / FlashAttention) avec support KV-Cache
# ============================================================================

class ModernCausalAttention(nn.Module):
    """Attention causale moderne avec PyTorch SDPA (FlashAttention / Scaled Dot-Product)."""

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

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)     # (B, n_heads, T, head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B, n_kv_heads, T, head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B, n_kv_heads, T, head_dim)

        q, k = apply_rotary_emb(q, k, cos, sin)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            if past_k is not None:
                k = torch.cat((past_k, k), dim=2)  # (B, n_kv_heads, T_past+T, head_dim)
                v = torch.cat((past_v, v), dim=2)
            new_kv_cache = (k, v)
        else:
            new_kv_cache = None

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)  # (B, n_heads, T_kv, head_dim)
            v = v.repeat_interleave(self.n_rep, dim=1)

        is_causal = kv_cache is None or k.shape[2] == q.shape[2]
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )  # (B, n_heads, T, head_dim)

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)  # (B, T, dim)
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


# ============================================================================
# Modèle complet
# ============================================================================

class ModernLLM(nn.Module):
    """Architecture complète GPT/LLaMA auto-régressive."""

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
        # Les poids liés (tok_embeddings == lm_head) ne sont comptés qu'une fois:
        # on déduplique par id() du tenseur avant de sommer.
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

        x = self.tok_embeddings(tokens)  # (B, T, dim)
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
        logits = self.lm_head(x)  # (B, T, vocab_size)

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

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        generated_ids: torch.Tensor | None = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Échantillonne le prochain token. logits: (1, vocab_size)."""

        # --- Pénalité de répétition (style HuggingFace) ---
        # Pour chaque token déjà généré, on divise son logit par la pénalité s'il est
        # positif (on le rend moins probable) ou on le multiplie s'il est négatif
        # (même effet: on pousse le score vers -inf). Casse les boucles de mots répétés.
        if repetition_penalty != 1.0 and generated_ids is not None and generated_ids.numel() > 0:
            unique_ids = torch.unique(generated_ids)
            prev_logits = logits[0, unique_ids]  # (n_unique,)
            penalized = torch.where(
                prev_logits > 0,
                prev_logits / repetition_penalty,
                prev_logits * repetition_penalty,
            )
            logits[0, unique_ids] = penalized

        if temperature <= 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)  # (1, 1)

        logits = logits / temperature

        if top_k is not None:
            top_k_clamped = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, top_k_clamped)
            threshold = v[:, [-1]]
            logits = torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)

        probs = F.softmax(logits, dim=-1)

        if top_p is not None and top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_mask = cumulative_probs - sorted_probs > top_p
            sorted_probs[sorted_mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)

        return torch.multinomial(probs, num_samples=1)  # (1, 1)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        tokenizer: "FrenchTokenizerWrapper",
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int | None = 40,
        repetition_penalty: float = 1.3,
    ):
        """Génération auto-régressive avec KV-Cache, temperature, top-k, top-p et
        pénalité de répétition. Le prompt est traité en une seule passe (prefill),
        puis chaque nouveau token n'est calculé qu'une seule fois (decode step)."""
        self.eval()
        device = next(self.parameters()).device

        token_ids = tokenizer.encode(prompt, allowed_special="all")
        tokens = torch.tensor([token_ids], dtype=torch.long, device=device)  # (1, T0)

        max_prompt_len = self.args.max_seq_len - 1
        tokens = tokens[:, -max_prompt_len:]

        kv_caches = [(None, None) for _ in range(self.args.n_layers)]

        logits, _, kv_caches = self.forward(tokens, kv_caches=kv_caches, start_pos=0)
        logits = logits[:, -1, :]
        cur_pos = tokens.shape[1]

        for _ in range(max_new_tokens):
            next_token = self._sample_next_token(
                logits, temperature, top_k, top_p,
                generated_ids=tokens, repetition_penalty=repetition_penalty,
            )
            tokens = torch.cat((tokens, next_token), dim=1)

            if next_token.item() == tokenizer.eot_token:
                break
            if cur_pos >= self.args.max_seq_len:
                break

            logits, _, kv_caches = self.forward(next_token, kv_caches=kv_caches, start_pos=cur_pos)
            logits = logits[:, -1, :]
            cur_pos += 1

        self.train()
        return tokenizer.decode(tokens[0].tolist())


# ============================================================================
# Dataset
# ============================================================================

class TextDataset(torch.utils.data.Dataset):
    """Découpe un long corpus tokenisé en fenêtres (input, target) décalées d'un token."""

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
# Tokenizer BPE français — vocabulaire entraîné sur asi/wikitext_fr (Hugging Face)
# ============================================================================

class FrenchTokenizerWrapper:
    """Adapte un `tokenizers.Tokenizer` (BPE byte-level) à l'interface utilisée
    dans le reste du script: `.encode()`, `.decode()`, `.eot_token`."""

    def __init__(self, tokenizer):
        self._tok = tokenizer
        eot_id = tokenizer.token_to_id("<|endoftext|>")
        assert eot_id is not None, "Le tokenizer doit contenir le token spécial <|endoftext|>"
        self.eot_token = eot_id
        self.vocab_size = tokenizer.get_vocab_size()

    def encode(self, text: str, allowed_special: str = "all") -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)


def _import_datasets_module():
    try:
        return __import__("datasets")
    except ImportError as e:
        raise ImportError(
            "Le package 'datasets' n'est pas installé. Installez-le avec `pip install datasets`."
        ) from e


# ============================================================================
# Filtre anti-LaTeX (corpus Wikipedia)
# ============================================================================
# Certains articles scientifiques d'asi/wikitext_fr / wikimedia/wikipedia
# contiennent des formules LaTeX brutes non rendues (\frac{}{}, T_{ij}, $...$,
# \begin{equation}...\end{equation}) injectées telles quelles comme texte
# français. Un petit modèle finit par mémoriser ces motifs comme "sortie
# plausible" et y retombe dès qu'il est incertain. On filtre ça à la source,
# une seule fois, avant que ces paragraphes ne servent à la fois à entraîner
# le tokenizer BPE et le corpus du modèle de langage.

_LATEX_COMMAND_RE = re.compile(r"\\(?:[a-zA-Z]+|[^a-zA-Z\s])")   # \frac, \alpha, \{, \\, ...
_LATEX_SCRIPT_RE = re.compile(r"[_^]\{[^{}]{0,80}\}")            # T_{ij}, x^{2}
_LATEX_INLINE_MATH_RE = re.compile(r"\${1,2}[^$\n]{1,200}\${1,2}")  # $...$ ou $$...$$
_LATEX_ENV_RE = re.compile(r"\\(?:begin|end)\{[a-zA-Z*]+\}")

# Part du paragraphe (en caractères, approximée) occupée par du balisage LaTeX
# brut au-delà de laquelle on considère le paragraphe entier comme pollué.
LATEX_DROP_THRESHOLD = 0.08


def _latex_pollution_ratio(text: str) -> float:
    """Estime la proportion d'un paragraphe qui ressemble à du LaTeX brut non
    rendu plutôt qu'à de la prose française."""
    if not text:
        return 0.0
    matches = (
        len(_LATEX_COMMAND_RE.findall(text))
        + len(_LATEX_SCRIPT_RE.findall(text))
        + len(_LATEX_INLINE_MATH_RE.findall(text))
        + len(_LATEX_ENV_RE.findall(text))
    )
    # Longueur moyenne approximative d'un motif LaTeX (\frac, _{ij}, ...) — sert
    # juste à obtenir un ratio, pas besoin d'un comptage de caractères exact.
    approx_chars = matches * 6
    return approx_chars / max(1, len(text))


def _strip_latex_noise(text: str) -> str:
    """Retire les fragments LaTeX isolés d'un paragraphe par ailleurs propre
    (ex: une formule ponctuelle au milieu de prose), en conservant le reste."""
    text = _LATEX_ENV_RE.sub(" ", text)
    text = _LATEX_INLINE_MATH_RE.sub(" ", text)
    text = _LATEX_SCRIPT_RE.sub(" ", text)
    text = _LATEX_COMMAND_RE.sub(" ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def filter_latex_pollution(paragraphs: list[str]) -> list[str]:
    """Filtre anti-LaTeX appliqué à des paragraphes Wikipedia bruts.
    - Paragraphe majoritairement formule (ratio > LATEX_DROP_THRESHOLD) -> supprimé.
    - Formule isolée dans un paragraphe sinon propre -> fragment retiré, prose gardée.
    - Si le nettoyage ne laisse presque rien d'exploitable, le paragraphe est supprimé.
    """
    cleaned = []
    dropped = 0
    for p in paragraphs:
        ratio = _latex_pollution_ratio(p)
        if ratio > LATEX_DROP_THRESHOLD:
            dropped += 1
            continue
        if ratio > 0:
            p = _strip_latex_noise(p)
            if len(p) < 20:
                dropped += 1
                continue
        cleaned.append(p)
    if dropped:
        print(f"🧹 Filtre anti-LaTeX: {dropped:,} paragraphe(s) pollué(s) retiré(s)/nettoyé(s) "
              f"sur {len(paragraphs):,} ({dropped / max(1, len(paragraphs)):.1%}).")
    return cleaned


def _load_wikitext_fr_via_datasets(max_chars: int) -> list[str]:
    """Tentative #1: `load_dataset` direct. Fonctionne uniquement si la version de
    `datasets` installée supporte encore les scripts de chargement personnalisés
    (dépréciés puis retirés dans les versions récentes de la librairie)."""
    datasets = _import_datasets_module()
    print(f"📥 Téléchargement de asi/wikitext_fr (config '{WIKI_CONFIG}') via load_dataset...")
    ds = datasets.load_dataset("asi/wikitext_fr", WIKI_CONFIG, split="train")
    paragraphs, total_chars = [], 0
    for row in ds:
        p = row["paragraph"].strip()
        if not p:
            continue
        paragraphs.append(p)
        total_chars += len(p)
        if total_chars >= max_chars:
            break
    return paragraphs


def _load_wikitext_fr_via_zip(max_chars: int) -> list[str]:
    """Tentative #2: contourne `load_dataset` en téléchargeant et extrayant
    directement l'archive de données brutes hébergée dans le dépôt HF (le script
    de chargement pointe vers `<config>/wiki.zip`, relatif à la racine du dépôt)."""
    from huggingface_hub import hf_hub_download
    import zipfile

    folder = "wikitext_72" if WIKI_CONFIG == "wikitext-72" else "wikitext_35"
    print(f"📥 Téléchargement direct de {folder}/wiki.zip depuis asi/wikitext_fr...")
    zip_path = hf_hub_download(repo_id="asi/wikitext_fr", repo_type="dataset", filename=f"{folder}/wiki.zip")
    extract_dir = zip_path + "_extracted"
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    train_file = None
    for root, _, files in os.walk(extract_dir):
        for fname in files:
            if "train" in fname.lower():
                train_file = os.path.join(root, fname)
                break
    if train_file is None:
        raise FileNotFoundError("Fichier d'entraînement introuvable dans l'archive extraite.")

    with open(train_file, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read(max_chars)
    return [p.strip() for p in raw.split("\n") if len(p.strip()) > 20]


def _load_wikimedia_wikipedia_fr(max_chars: int) -> list[str]:
    """Tentative #3 (repli fiable): `wikimedia/wikipedia` (config '20231101.fr') —
    corpus Wikipedia français équivalent, en Parquet natif, donc compatible avec
    toutes les versions récentes de `datasets`. Chargé en streaming pour ne
    télécharger que ce qui est nécessaire (le dataset complet fait plusieurs Go)."""
    try:
        from datasets import load_dataset
    except Exception:  # pragma: no cover - graceful fallback when `datasets` is not installed
        load_dataset = None
        import warnings

        warnings.warn(
            "Optional dependency 'datasets' is not available. Functions that rely on it will raise an error if used.\n"
            "Install it with: pip install datasets"
        )
    print("📥 Téléchargement (streaming) de wikimedia/wikipedia (fr) en repli...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.fr", split="train", streaming=True)
    paragraphs, total_chars = [], 0
    for row in ds:
        text = row["text"].strip()
        for p in text.split("\n\n"):
            p = p.strip()
            if len(p) < 50:  # ignore titres/fragments trop courts
                continue
            paragraphs.append(p)
            total_chars += len(p)
        if total_chars >= max_chars:
            break
    return paragraphs


def load_wikipedia_paragraphs(max_chars: int = WIKI_MAX_CHARS) -> list[str]:
    """Charge des paragraphes de Wikipédia en français, avec plusieurs niveaux de
    repli en cascade: asi/wikitext_fr (le corpus demandé, via deux stratégies de
    chargement différentes), puis wikimedia/wikipedia (fr) comme équivalent
    robuste, puis un texte français embarqué en tout dernier recours."""
    attempts = [
        (lambda: _load_wikitext_fr_via_datasets(max_chars), "asi/wikitext_fr (load_dataset)"),
        (lambda: _load_wikitext_fr_via_zip(max_chars), "asi/wikitext_fr (téléchargement direct)"),
        (lambda: _load_wikimedia_wikipedia_fr(max_chars), "wikimedia/wikipedia (fr)"),
    ]
    for loader, label in attempts:
        try:
            paragraphs = loader()
            print(f"✅ {len(paragraphs):,} paragraphes chargés depuis {label}.")
            paragraphs = filter_latex_pollution(paragraphs)
            return paragraphs
        except Exception as e:
            print(f"⚠️ Échec avec {label} ({e}).")
    print("↪️ Tous les téléchargements ont échoué — utilisation du texte de secours en français embarqué.")
    return [FALLBACK_TEXT_FR]


def build_or_load_french_tokenizer(
    paragraphs: list[str],
    vocab_size: int = TOKENIZER_VOCAB_SIZE,
    cache_path: str = TOKENIZER_CACHE_PATH,
) -> FrenchTokenizerWrapper:
    """Entraîne un tokenizer BPE byte-level (façon GPT-2, donc pas d'OOV possible)
    sur des paragraphes Wikipedia en français — vocabulaire authentiquement
    français, beaucoup plus compact que le vocab anglais de gpt2 (50257 tokens).
    Si un tokenizer entraîné est déjà en cache sur disque, il est rechargé direct."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder

    if os.path.exists(cache_path):
        print(f"📂 Tokenizer français rechargé depuis {cache_path}.")
        tok = Tokenizer.from_file(cache_path)
        return FrenchTokenizerWrapper(tok)

    print(f"🛠️ Entraînement d'un tokenizer BPE français (vocab_size={vocab_size}) "
          f"sur {len(paragraphs):,} paragraphes...")
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=["<|endoftext|>"])
    tokenizer.train_from_iterator(paragraphs, trainer=trainer)
    # Enregistre <|endoftext|> comme token spécial "atomique": sans ça, encode()
    # découperait la chaîne littérale en octets au lieu de la reconnaître d'un bloc.
    tokenizer.add_special_tokens(["<|endoftext|>"])
    tokenizer.save(cache_path)
    print(f"✅ Tokenizer entraîné (vocabulaire réel: {tokenizer.get_vocab_size()} tokens) "
          f"et sauvegardé dans {cache_path}.")
    return FrenchTokenizerWrapper(tokenizer)


# ============================================================================
# Chargement du dataset — TinyStories-French, avec repli sur un texte français
# embarqué si le téléchargement échoue (pas de réseau, dataset gated, etc.)
# ============================================================================

FALLBACK_TEXT_FR = """
Il était une fois un petit renard curieux qui vivait à l'orée d'une forêt tranquille.
Chaque matin, il sortait de son terrier pour explorer les sentiers couverts de mousse.
Un jour, il rencontra une chouette sage perchée sur une branche basse.
La chouette lui dit: si tu veux comprendre la forêt, il faut d'abord apprendre à écouter le silence.
Le petit renard s'assit et ferma les yeux. Il entendit le vent dans les feuilles, le ruisseau au loin, et les oiseaux qui chantaient.
Depuis ce jour, il revenait souvent voir la chouette pour apprendre de nouvelles histoires.
Un lapin nommé Noisette vivait aussi dans cette forêt. Il aimait collectionner de petits cailloux ronds.
Chaque cailloux avait une couleur différente, et Noisette les rangeait soigneusement dans un panier tressé.
Un matin de printemps, la rivière déborda légèrement à cause de la fonte des neiges.
Le renard et le lapin décidèrent de construire un petit pont avec des branches pour aider leurs amis à traverser.
Ensemble, ils travaillèrent toute la journée, portant des bâtons et les attachant avec des lianes solides.
Quand le pont fut terminé, tous les animaux de la forêt vinrent le remercier avec des fleurs et des fruits.
La chouette sage regarda la scène depuis son arbre et sourit: la coopération est la plus belle des forces.
Le soir venu, les étoiles apparurent une à une dans le ciel violet, et la forêt s'endormit doucement.
Le lendemain, le petit renard raconta cette aventure à tous ses amis, encore et encore, avec des étoiles dans les yeux.
""" * 40  # répété pour donner un corpus d'entraînement de taille suffisante


def _fix_mojibake(text: str) -> str:
    """Corrige les artefacts d'encodage du type 'Ã©' -> 'é' présents dans certaines
    lignes du dataset (texte UTF-8 mal redécodé en Latin-1 à un moment de la pipeline)."""
    if "Ã©" in text or "Ã¨" in text or "â€™" in text:
        try:
            return text.encode("latin1").decode("utf8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text
    return text


def load_training_text(n_stories: int = N_STORIES) -> str:
    """Charge TinyStories-French (traduction française de TinyStories) depuis Hugging
    Face; retombe sur un texte français embarqué si le téléchargement échoue."""
    try:
        from datasets import load_dataset
        print(f"📥 Téléchargement de TinyStories-French...")
        # Le dataset ne contient qu'environ 1000 histoires (bien moins que l'original
        # anglais) — on charge tout le split puis on tronque si n_stories est plus petit.
        ds = load_dataset("iproskurina/TinyStories-French", split="train")
        column = "french-tinystories" if "french-tinystories" in ds.column_names else ds.column_names[0]
        texts = [t for t in ds[column] if t and t.strip()]
        texts = texts[:n_stories] if n_stories < len(texts) else texts
        texts = [_fix_mojibake(t) for t in texts]
        dataset_text = "\n<|endoftext|>\n".join(texts)
        print(f"✅ Dataset chargé: {len(texts)} histoires en français, {len(dataset_text):,} caractères.")
        if len(texts) < 1500:
            print("ℹ️ Corpus restreint (~1000 histoires) — le modèle reverra plusieurs fois "
                  "les mêmes textes sur 1000 steps, ce qui reste adapté à un modèle de cette taille.")
        return dataset_text
    except Exception as e:
        print(f"⚠️ Impossible de charger TinyStories-French ({e}).")
        print("↪️ Utilisation du texte de secours en français (corpus embarqué).")
        return FALLBACK_TEXT_FR


# ============================================================================
# Trainer
# ============================================================================

class LLMTrainer:
    """Gestionnaire d'entraînement haut débit pour LLM."""

    def __init__(self, model: ModernLLM, args: ModelArgs, dataset_text: str, tokenizer: FrenchTokenizerWrapper, val_fraction: float = 0.05):
        self.model = model.to(args.device)
        self.args = args

        self.tokenizer = tokenizer
        token_ids = self.tokenizer.encode(dataset_text, allowed_special="all")
        print(f"🔤 Corpus tokenisé: {len(token_ids):,} tokens.")

        split_idx = int(len(token_ids) * (1 - val_fraction))
        train_ids = token_ids[:split_idx]
        val_ids = token_ids[split_idx:]

        self.train_dataset = TextDataset(train_ids, block_size=args.max_seq_len)
        self.val_dataset = TextDataset(val_ids, block_size=args.max_seq_len) if len(val_ids) > args.max_seq_len + 1 else self.train_dataset

        decay_params, no_decay_params = [], []
        seen = set()
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
            optim_groups, lr=MAX_LR, betas=(0.9, 0.95), eps=1e-8, fused=use_fused,
        )

        self.amp_dtype = torch.bfloat16 if (args.device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        self.scaler = torch.amp.GradScaler(enabled=(self.amp_dtype == torch.float16))
        self.grad_clip = 1.0

    def get_lr(self, step: int, max_steps: int, warmup_steps: int, max_lr: float, min_lr: float) -> float:
        if step < warmup_steps:
            return max_lr * (step + 1) / warmup_steps
        if step >= max_steps:
            return min_lr
        decay_ratio = min(max((step - warmup_steps) / max(1, (max_steps - warmup_steps)), 0.0), 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

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
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()

    @torch.no_grad()
    def evaluate(self, n_batches: int = 20, batch_size: int = BATCH_SIZE) -> float:
        """Perte moyenne sur des batchs aléatoires du split de validation."""
        self.model.eval()
        losses = []
        for _ in range(n_batches):
            idxs = [random.randint(0, len(self.val_dataset) - 1) for _ in range(min(batch_size, max(1, len(self.val_dataset))))]
            xb = torch.stack([self.val_dataset[i][0] for i in idxs])
            yb = torch.stack([self.val_dataset[i][1] for i in idxs])
            xb, yb = xb.to(self.args.device), yb.to(self.args.device)
            _, loss = self.model(xb, targets=yb)
            losses.append(loss.item())
        self.model.train()
        return sum(losses) / len(losses)


# ============================================================================
# Point d'entrée — entraînement complet
# ============================================================================

if __name__ == "__main__":
    torch.manual_seed(SEED)
    random.seed(SEED)

    # --- 1. Tokenizer français + articles Wikipedia (asi/wikitext_fr) ---
    # On charge les paragraphes une seule fois: ils servent à la fois à entraîner
    # le vocabulaire BPE et à enrichir le corpus d'entraînement du modèle de langage.
    wiki_paragraphs = load_wikipedia_paragraphs()
    tokenizer = build_or_load_french_tokenizer(wiki_paragraphs, vocab_size=TOKENIZER_VOCAB_SIZE)

    # --- 2. Modèle — vocab_size fixé sur la taille réelle du tokenizer entraîné ---
    print("\n🚀 Initialisation du modèle...")
    args = ModelArgs(vocab_size=tokenizer.vocab_size)
    model = ModernLLM(args).to(args.device)
    n_params = model.num_params()
    print(f"✅ Modèle instancié sur '{args.device}' avec {n_params / 1e6:.2f}M paramètres "
          f"(vocab={args.vocab_size}, dim={args.dim}, n_layers={args.n_layers}).")

    if RESUME_FROM_CHECKPOINT and os.path.exists(CHECKPOINT_PATH):
        print(f"♻️  Reprise depuis '{CHECKPOINT_PATH}' (au lieu de repartir de poids aléatoires)...")
        _ckpt = torch.load(CHECKPOINT_PATH, map_location=args.device, weights_only=False)
        model.load_state_dict(_ckpt["model_state_dict"])
        print(f"   ↳ poids chargés (step précédent: {_ckpt.get('step', '?')}, "
              f"val_loss précédent: {_ckpt.get('val_loss', float('nan')):.4f})")

    # --- 3. Corpus d'entraînement: TinyStories-French + articles Wikipedia ---
    tinystories_text = load_training_text(N_STORIES)
    wiki_text = "\n<|endoftext|>\n".join(wiki_paragraphs)
    dataset_text = tinystories_text + "\n<|endoftext|>\n" + wiki_text
    print(f"📚 Corpus combiné: {len(tinystories_text):,} caractères TinyStories-French + "
          f"{len(wiki_text):,} caractères Wikipedia.")

    trainer = LLMTrainer(model, args, dataset_text, tokenizer=tokenizer)
    print(f"📚 Fenêtres d'entraînement: {len(trainer.train_dataset):,} | validation: {len(trainer.val_dataset):,}")

    loader = torch.utils.data.DataLoader(
        trainer.train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    )

    def infinite_loader(dl):
        while True:
            for batch in dl:
                yield batch

    data_iter = infinite_loader(loader)

    best_val_loss = float("inf")
    print(f"\n🏋️ Entraînement pour {MAX_STEPS} steps...\n")

    for step in range(MAX_STEPS):
        lr = trainer.get_lr(step, MAX_STEPS, WARMUP_STEPS, MAX_LR, MIN_LR)
        for group in trainer.optimizer.param_groups:
            group["lr"] = lr

        x_batch, y_batch = next(data_iter)
        train_loss = trainer.train_step(x_batch, y_batch)

        if step % 20 == 0 or step == MAX_STEPS - 1:
            print(f"step {step:04d}/{MAX_STEPS} | lr {lr:.2e} | train_loss {train_loss:.4f}")

        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
            val_loss = trainer.evaluate()
            print(f"   ↳ 📊 val_loss {val_loss:.4f} (best: {best_val_loss:.4f})")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {"model_state_dict": model.state_dict(), "args": args, "step": step, "val_loss": val_loss},
                    CHECKPOINT_PATH,
                )
                print(f"   ↳ 💾 Nouveau meilleur modèle sauvegardé ({CHECKPOINT_PATH})")

        if step % GEN_INTERVAL == 0 and step > 0:
            sample = model.generate(
                prompt="Il était une fois",
                tokenizer=trainer.tokenizer,
                max_new_tokens=40,
                temperature=0.8,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.3,
            )
            print(f"   ↳ 📝 Échantillon: {sample!r}\n")

    print("\n✅ Entraînement terminé.")

    # --- Rechargement du meilleur modèle et génération finale ---
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"📂 Meilleur modèle rechargé (step {checkpoint['step']}, val_loss {checkpoint['val_loss']:.4f})")

    print("\n🧪 Génération finale (temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.3):")
    final_sample = model.generate(
        prompt="Il était une fois",
        tokenizer=trainer.tokenizer,
        max_new_tokens=80,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.3,
    )
    print(final_sample)
