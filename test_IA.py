import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

try:
    from tokenizers import Tokenizer
except ImportError:
    print("❌ Le package 'tokenizers' est requis pour recharger le tokenizer BPE français.")
    print("👉 Installez-le via : pip install tokenizers")
    sys.exit(1)


@dataclass
class ModelArgs:
    """Configuration hyperparamétrique pour architecture Transformer type LLaMA/Qwen."""
    vocab_size: int = 16000
    dim: int = 384
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int | None = 4
    max_seq_len: int = 256
    dropout: float = 0.0
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class FrenchTokenizerWrapper:
    """Wrapper pour charger et utiliser le tokenizer BPE personnalisé sauvegardé pendant l'entraînement."""

    def __init__(self, json_path: str):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Le fichier tokenizer '{json_path}' est introuvable !")

        self._tok = Tokenizer.from_file(json_path)

        # Récupération de l'ID pour le token de fin de texte / de séquence
        eot_id = self._tok.token_to_id("<|endoftext|>")
        self.eos_token_id = eot_id if eot_id is not None else 0
        self.vocab_size = self._tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        """Encode une chaîne de texte en liste d'identifiants de tokens."""
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        """Décode une liste d'identifiants de tokens en chaîne de texte."""
        return self._tok.decode(ids, skip_special_tokens=False)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

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
    """Rotary Position Embeddings (RoPE)."""

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
    """Pivote la moitié des dimensions pour RoPE: [-x2, x1]."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Applique la rotation RoPE sur Query et Key."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    xq_rotated = (xq * cos) + (rotate_half(xq) * sin)
    xk_rotated = (xk * cos) + (rotate_half(xk) * sin)
    return xq_rotated, xk_rotated


class SwiGLUFeedForward(nn.Module):
    """Couche MLP SwiGLU (Gated Linear Unit avec SiLU)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = int(8 * args.dim / 3)
        multiple_of = 256
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w1(x))
        up = self.w3(x)
        return self.w2(gate * up)


class ModernCausalAttention(nn.Module):
    """Attention causale moderne avec PyTorch SDPA et KV-Cache."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        assert args.n_heads % self.n_kv_heads == 0, "n_heads doit être divisible par n_kv_heads"
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)

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
            dropout_p=0.0,
            is_causal=is_causal,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        y = self.wo(y)

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
    """Architecture LLM auto-régressive moderne."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        head_dim = args.dim // args.n_heads
        self.rope = RotaryEmbedding(head_dim, max_seq_len=args.max_seq_len, theta=args.rope_theta)

        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm_f = RMSNorm(args.dim, eps=args.norm_eps)

        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.tok_embeddings.weight = self.lm_head.weight  # Weight Tying

    def forward(self, tokens: torch.Tensor, kv_caches: list | None = None, start_pos: int = 0):
        B, T = tokens.shape
        x = self.tok_embeddings(tokens)
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

        if kv_caches is not None:
            return logits, new_kv_caches
        return logits


class ChatFormatter:
    """Formate l'historique de conversation avec des rôles explicites et gère
    le tronquage pour respecter la fenêtre de contexte (max_seq_len) du modèle."""

    USER_TAG = "[Utilisateur]"
    ASSISTANT_TAG = "[Assistant]"

    # Séquences qui signalent que le modèle a fini son tour et commence à
    # halluciner la réplique suivante (ce qu'un modèle non fine-tuné pour le
    # dialogue fait très souvent). Dès qu'une de ces séquences apparaît dans
    # le texte généré, on coupe la génération avant elle.
    STOP_SEQUENCES = [USER_TAG, ASSISTANT_TAG, "<|endoftext|>"]

    def __init__(self, tokenizer: "FrenchTokenizerWrapper"):
        self.tokenizer = tokenizer

    def _turn_text(self, role: str, content: str) -> str:
        tag = self.USER_TAG if role == "user" else self.ASSISTANT_TAG
        return f"{tag}\n{content}\n"

    def build_prompt(self, history: list[dict], max_ctx_tokens: int) -> str:
        """Construit le prompt en incluant le plus d'historique récent possible
        sans dépasser `max_ctx_tokens` (on réserve toujours le tour courant).

        history: liste de {"role": "user"|"assistant", "content": str}, le
        dernier élément doit être le message utilisateur courant.
        """
        if not history:
            return f"{self.ASSISTANT_TAG}\n"

        *past_turns, current_user_turn = history
        suffix = self._turn_text("user", current_user_turn["content"]) + f"{self.ASSISTANT_TAG}\n"
        suffix_len = len(self.tokenizer.encode(suffix))

        # On ajoute les tours passés du plus récent au plus ancien tant que ça
        # tient dans le budget de tokens restant.
        kept_chunks: list[str] = []
        budget = max_ctx_tokens - suffix_len
        for turn in reversed(past_turns):
            chunk = self._turn_text(turn["role"], turn["content"])
            n_tokens = len(self.tokenizer.encode(chunk))
            if n_tokens > budget:
                break
            kept_chunks.append(chunk)
            budget -= n_tokens

        kept_chunks.reverse()
        return "".join(kept_chunks) + suffix

    def find_stop(self, text: str) -> int | None:
        """Retourne l'index de la première séquence d'arrêt trouvée dans `text`,
        ou None si aucune n'est présente."""
        indices = [text.find(s) for s in self.STOP_SEQUENCES if s in text]
        return min(indices) if indices else None


class StreamingInferenceEngine:
    """Moteur de génération en streaming, avec décodage UTF-8 correct et
    arrêt sur séquence de rôle (pour ne pas halluciner le tour suivant)."""

    def __init__(self, model: ModernLLM, tokenizer: FrenchTokenizerWrapper):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.formatter = ChatFormatter(tokenizer)

    def sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        generated_ids: torch.Tensor | None = None,
        repetition_penalty: float = 1.2,
    ) -> torch.Tensor:
        """Échantillonnage de token avec pénalité de répétition, top-k et top-p."""
        if repetition_penalty != 1.0 and generated_ids is not None and generated_ids.numel() > 0:
            unique_ids = torch.unique(generated_ids)
            valid_unique_ids = unique_ids[unique_ids < logits.size(-1)]
            if valid_unique_ids.numel() > 0:
                prev_logits = logits[0, valid_unique_ids]
                penalized = torch.where(
                    prev_logits > 0,
                    prev_logits / repetition_penalty,
                    prev_logits * repetition_penalty,
                )
                logits[0, valid_unique_ids] = penalized

        if temperature <= 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        logits = logits / temperature

        if top_k is not None and top_k > 0:
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

        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def stream_generate_from_history(
        self,
        history: list[dict],
        max_new_tokens: int = 80,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.3,
    ):
        """Génère la réponse de l'assistant pour un historique de conversation.

        - Construit le prompt avec rôles via ChatFormatter (tronque l'historique
          pour respecter max_seq_len).
        - Décode l'intégralité des tokens de la RÉPONSE à chaque étape (jamais un
          seul token isolé) pour ne jamais casser un caractère UTF-8 multi-octets.
        - S'arrête dès qu'une séquence de rôle apparaît (le modèle commence sinon
          à halluciner "[Utilisateur]" et à continuer la conversation lui-même).

        Renvoie (yield) des fragments de texte au fur et à mesure, prêts à
        afficher. Le texte final complet est aussi accessible via `self.last_response`.
        """
        self.model.eval()
        max_seq_len = self.model.args.max_seq_len

        # On réserve de la place pour les tokens à générer dans le budget de contexte.
        max_prompt_tokens = max(1, max_seq_len - max_new_tokens - 1)
        prompt = self.formatter.build_prompt(history, max_ctx_tokens=max_prompt_tokens)

        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            self.last_response = ""
            return

        prompt_len = len(prompt_ids)
        tokens = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        tokens = tokens[:, -(max_seq_len - 1):]  # garde-fou final

        kv_caches = [(None, None) for _ in range(self.model.args.n_layers)]
        logits, kv_caches = self.model(tokens, kv_caches=kv_caches, start_pos=0)
        logits = logits[:, -1, :]
        cur_pos = tokens.shape[1]

        response_ids: list[int] = []
        emitted_text = ""      # texte déjà "yield" (décodé depuis response_ids)
        stop_hold = ""         # buffer de retenue au cas où une séquence d'arrêt
                                # serait en train de se former à cheval sur 2 tokens
        max_stop_len = max(len(s) for s in self.formatter.STOP_SEQUENCES)
        stopped_on_sequence = False

        for _ in range(max_new_tokens):
            if cur_pos >= max_seq_len:
                break

            next_token = self.sample_next_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generated_ids=tokens,
                repetition_penalty=repetition_penalty,
            )
            token_val = next_token.item()

            if token_val == self.tokenizer.eos_token_id:
                break

            response_ids.append(token_val)

            # Décodage de TOUTE la réponse générée jusqu'ici (jamais un seul token
            # isolé) : ça évite déjà la quasi-totalité des '�'. Il reste un cas
            # limite -- le tout dernier caractère peut être coupé en plein milieu
            # d'une séquence UTF-8 multi-octets tant que le token suivant (qui la
            # complète) n'a pas encore été généré. On détecte ça via les '\ufffd'
            # (caractère de remplacement) en fin de texte et on les retient tant
            # qu'ils ne sont pas résolus par un futur token.
            full_response_text = self.tokenizer.decode(response_ids)
            stable_text = full_response_text.rstrip("\ufffd")
            new_chunk = stop_hold + stable_text[len(emitted_text):]
            emitted_text = stable_text

            stop_idx = self.formatter.find_stop(new_chunk)
            if stop_idx is not None:
                # Le modèle commence à halluciner le tour suivant : on coupe ici
                # et on ne remet jamais stop_hold en circulation ensuite.
                safe_part = new_chunk[:stop_idx]
                if safe_part:
                    yield safe_part
                stop_hold = ""
                stopped_on_sequence = True
                break

            # On retient les derniers caractères qui pourraient être le début
            # d'une séquence d'arrêt tant qu'on n'est pas sûr que ce n'en est pas une.
            if len(new_chunk) > max_stop_len:
                to_yield = new_chunk[: len(new_chunk) - (max_stop_len - 1)]
                stop_hold = new_chunk[len(new_chunk) - (max_stop_len - 1):]
                if to_yield:
                    yield to_yield
            else:
                stop_hold = new_chunk

            tokens = torch.cat((tokens, next_token), dim=1)
            logits, kv_caches = self.model(next_token, kv_caches=kv_caches, start_pos=cur_pos)
            logits = logits[:, -1, :]
            cur_pos += 1

        # Fin de boucle sans avoir trouvé de séquence d'arrêt : on vide le
        # buffer de retenue restant (il ne contient alors pas de séquence d'arrêt).
        if stop_hold and not stopped_on_sequence:
            yield stop_hold

        self.last_response = self.tokenizer.decode(response_ids).strip()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, "chat_model.pt")
    tokenizer_path = os.path.join(script_dir, "fr_bpe_tokenizer.json")

    if not os.path.exists(checkpoint_path) and os.path.exists("best_model.pt"):
        checkpoint_path = os.path.abspath("best_model.pt")

    if not os.path.exists(tokenizer_path) and os.path.exists("fr_bpe_tokenizer.json"):
        tokenizer_path = os.path.abspath("fr_bpe_tokenizer.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 65)
    print("🤖 INTERACTIVE STREAMING LLM INFERENCE (French BPE Tokenizer)")
    print("=" * 65)
    print(f"💻 Peripherique utilise : {device.upper()}")
    print(f"📂 Répertoire du script : {script_dir}")
    print(f"📂 Fichier modèle : {checkpoint_path}")
    print(f"📂 Fichier tokenizer : {tokenizer_path}")

    # Chargement du tokenizer BPE français sur-mesure
    try:
        print(f"\n🔍 Chargement du tokenizer BPE ('{os.path.basename(tokenizer_path)}')...")
        tokenizer = FrenchTokenizerWrapper(tokenizer_path)
        print(f"✅ Tokenizer BPE chargé avec succès (Taille du vocabulaire: {tokenizer.vocab_size:,})")
    except Exception as e:
        print(f"❌ ERREUR lors du chargement du tokenizer BPE : {e}")
        print("💡 Assurez-vous d'avoir bien exécuté l'entraînement pour générer 'fr_bpe_tokenizer.json'.")
        sys.exit(1)

    # Chargement et détection automatique des dimensions du fichier .pt
    if os.path.exists(checkpoint_path):
        print(f"\n✅ Poids trouves ! Chargement depuis '{checkpoint_path}'...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            val_loss = checkpoint.get("val_loss", "N/A")
            step = checkpoint.get("step", "N/A")
            saved_args = checkpoint.get("args", None)
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
            val_loss, step, saved_args = "N/A", "N/A", None
        else:
            state_dict = checkpoint
            val_loss, step, saved_args = "N/A", "N/A", None

        if saved_args is not None and hasattr(saved_args, "vocab_size"):
            args = saved_args
            args.device = device
            print(f"📐 Configuration lue du checkpoint : vocab={args.vocab_size}, dim={args.dim}, layers={args.n_layers}, heads={args.n_heads}")
        else:
            emb_shape = state_dict["tok_embeddings.weight"].shape
            vocab_size_ckpt, dim_ckpt = emb_shape[0], emb_shape[1]

            layer_indices = set()
            for key in state_dict.keys():
                if key.startswith("layers."):
                    idx = int(key.split(".")[1])
                    layer_indices.add(idx)
            n_layers_ckpt = len(layer_indices) if layer_indices else 8

            wk_shape = state_dict["layers.0.attn.wk.weight"].shape
            n_heads_ckpt = 8
            head_dim = dim_ckpt // n_heads_ckpt
            n_kv_heads_ckpt = wk_shape[0] // head_dim

            args = ModelArgs(
                vocab_size=vocab_size_ckpt,
                dim=dim_ckpt,
                n_layers=n_layers_ckpt,
                n_heads=n_heads_ckpt,
                n_kv_heads=n_kv_heads_ckpt,
                device=device,
            )
            print(f"🔍 Dimensions déduites du fichier .pt : vocab={vocab_size_ckpt}, dim={dim_ckpt}, layers={n_layers_ckpt}, heads={n_heads_ckpt}, kv_heads={n_kv_heads_ckpt}")

        model = ModernLLM(args).to(device)
        model.load_state_dict(state_dict)
        print(f"✅ Modèle charge avec succès ! (Étape: {step}, Validation Loss: {val_loss})")
    else:
        print(f"\n⚠️ ATTENTION : Fichier '{checkpoint_path}' INTROUVABLE !")
        print("💡 Le modèle utilise des poids initialises aléatoirement.")
        print("👉 Lancez d'abord votre script d'entraînement !\n")
        args = ModelArgs(vocab_size=tokenizer.vocab_size, device=device)
        model = ModernLLM(args).to(device)

    engine = StreamingInferenceEngine(model, tokenizer)
    history: list[dict] = []  # [{"role": "user"|"assistant", "content": str}, ...]

    print("\n" + "-" * 65)
    print("💬 Entrez votre message ci-dessous.")
    print("   Commandes : 'exit'/'quit' pour quitter, 'reset' pour effacer l'historique.")
    print("-" * 65)
    print("⚠️  Ce modèle n'a PAS été fine-tuné pour le dialogue (seulement entraîné")
    print("    sur des contes et des articles Wikipedia). Les tags de rôle et l'arrêt")
    print("    de génération rendent la sortie propre et lisible, mais le contenu des")
    print("    réponses restera de la complétion de texte, pas un vrai raisonnement")
    print("    d'assistant. Voir la note en fin de script pour y remédier.")
    print("-" * 65 + "\n")

    while True:
        try:
            prompt = input("\n👉 Vous: ").strip()
            if not prompt:
                continue
            if prompt.lower() in ["exit", "quit", "q"]:
                print("\n👋 Au revoir !")
                break
            if prompt.lower() == "reset":
                history.clear()
                print("🧹 Historique effacé.")
                continue

            history.append({"role": "user", "content": prompt})

            print("🤖 Assistant: ", end="", flush=True)
            for chunk in engine.stream_generate_from_history(
                history=history,
                max_new_tokens=80,
                temperature=0.7,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.3,
            ):
                print(chunk, end="", flush=True)
            print()

            response_text = engine.last_response
            if not response_text:
                # Le modèle n'a rien produit d'utilisable (ex: EOS immédiat) :
                # on retire le message utilisateur pour ne pas polluer l'historique.
                history.pop()
            else:
                history.append({"role": "assistant", "content": response_text})

        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Interruption par l'utilisateur. Au revoir !")
            break





def send_to_app_server(
    text: str, 
    server_url: str = "http://localhost:5000",
    conversation_id: str = "default",
    word_by_word: bool = True
) -> bool:
    """
    Envoie le texte vers le serveur app.py.
    
    Args:
        text: Texte à envoyer
        server_url: URL du serveur Flask
        conversation_id: Identifiant de la conversation
        word_by_word: Si True, envoie mot par mot. Si False, envoie en une fois.
    
    Returns:
        True si l'envoi a réussi, False sinon
    """
    import requests
    import time
    
    try:
        if word_by_word:
            # Envoyer mot par mot
            words = text.split()
            for word in words:
                data = {
                    "conversation_id": conversation_id,
                    "word": word,
                    "timestamp": time.time()
                }
                response = requests.post(
                    f"{server_url}/receive_word",
                    json=data,
                    timeout=1.0
                )
                if response.status_code != 200:
                    print(f"Erreur lors de l'envoi du mot '{word}': {response.status_code}")
                    return False
                time.sleep(0.05)  # Petite pause pour simuler le streaming
            
            # Envoyer un signal de fin
            end_data = {
                "conversation_id": conversation_id,
                "word": "",
                "end_of_message": True,
                "timestamp": time.time()
            }
            requests.post(
                f"{server_url}/receive_word",
                json=end_data,
                timeout=1.0
            )
        else:
            # Envoyer en une fois
            data = {
                "conversation_id": conversation_id,
                "chunk": text,
                "timestamp": time.time()
            }
            response = requests.post(
                f"{server_url}/receive_chunk",
                json=data,
                timeout=1.0
            )
            if response.status_code != 200:
                print(f"Erreur lors de l'envoi du chunk: {response.status_code}")
                return False
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Erreur de connexion au serveur: {e}")
        return False





