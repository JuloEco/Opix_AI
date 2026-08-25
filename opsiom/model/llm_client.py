# ============================================================================
# model/llm_client.py — Interface LLM commune pour RAG et Reasoning
#
# Conformément à l'architecture décrite dans le README (section 2 et 11.1),
# les modules `rag/` et `reasoning/` ne doivent JAMAIS dépendre directement
# des internals du modèle (ModernLLM, tokenizer, etc.). Ils appellent une
# interface `LLMClient` minimale, ce qui permet de remplacer le modèle, son
# hébergement, ou sa version, sans toucher au code de RAG/Reasoning.
#
# Deux implémentations sont fournies :
#   - LocalOpsiomClient : charge un checkpoint .pt (format compatible
#     test_IA_2.py / train_opsiom.py / main.py : model_state_dict / args /
#     val_loss / step) et génère en mémoire, dans le même processus.
#   - APIOpsiomClient : appelle l'API Flask existante (POST /api/chat), utile
#     quand RAG/Reasoning tournent dans un service séparé du modèle.
#
# ----------------------------------------------------------------------------
# ⚠️ CORRECTIF — incohérence entre train_opsiom.py et main.py
# ----------------------------------------------------------------------------
# Le projet fait cohabiter DEUX scripts qui redéfinissent chacun localement
# `ModernLLM` / `ModelArgs` / `FrenchTokenizerWrapper` (pour rester autonomes
# en cellule Colab/Kaggle), avec des interfaces DIFFÉRENTES et INCOMPATIBLES :
#
#   train_opsiom.py :
#       FrenchTokenizerWrapper(hf_repo="...", local_path="...")   # kwargs
#       tokenizer.eos_token_id
#       tokenizer.encode(text)                                    # 1 argument
#
#   main.py (script de pré-entraînement à grande échelle) :
#       FrenchTokenizerWrapper(tokenizer_obj)                      # positionnel,
#                                                                   # un objet
#                                                                   # tokenizers.Tokenizer
#                                                                   # déjà chargé
#       tokenizer.eot_token
#       tokenizer.encode(text, allowed_special="all")              # kwarg en plus
#
# `ModernLLM.generate()` référence l'un OU l'autre de ces deux noms d'attribut
# selon le fichier dont il provient — donc selon que le checkpoint chargé ici
# a été produit par train_opsiom.py ou par main.py, l'ancienne version de ce
# client (qui appelait toujours `FrenchTokenizerWrapper(hf_repo=..., local_path=...)`
# et ne définissait jamais `.eot_token`) plantait dans la moitié des cas.
#
# `_import_architecture_module()` et `_build_tokenizer()` ci-dessous détectent
# l'interface réellement disponible et normalisent le résultat pour que
# LocalOpsiomClient fonctionne avec les DEUX. Voir aussi INTEGRATION.md
# (section "Incohérence connue entre train_opsiom.py et main.py") pour la
# vraie solution à terme : extraire une seule définition partagée dans
# `model/architecture.py` (déjà anticipé comme chemin d'import prioritaire
# ci-dessous), pour ne plus avoir à faire cohabiter deux versions.
#
# Autre point corrigé au passage : `ModelArgs` a des valeurs par défaut
# différentes selon le script (ex: `dim` par défaut) — ce client ne s'appuie
# JAMAIS sur ces valeurs par défaut. Il reconstruit systématiquement
# `ModelArgs` depuis `ckpt["args"]`, l'objet réellement sauvegardé dans le
# checkpoint au moment de l'entraînement (voir `__post_init__` ci-dessous),
# donc le `dim` par défaut d'un script ou de l'autre n'a aucune incidence.
# ============================================================================

from __future__ import annotations

import abc
import os
from dataclasses import dataclass


def _import_architecture_module():
    """Localise le module qui définit `ModernLLM` / `FrenchTokenizerWrapper`.

    Ordre de priorité :
      1. `model.architecture` — module dédié recommandé (README section 11.1),
         à créer en factorisant train_opsiom.py et main.py si ce n'est pas
         déjà fait ; s'il existe, il n'y a plus d'ambiguïté d'interface.
      2. `train_opsiom` — interface `FrenchTokenizerWrapper(hf_repo=, local_path=)`.
      3. `main` — interface `FrenchTokenizerWrapper(tokenizer_obj)`.
    """
    try:
        import model.architecture as arch_module  # type: ignore

        return arch_module
    except ImportError:
        pass
    try:
        import train_opsiom as arch_module  # type: ignore

        return arch_module
    except ImportError:
        pass
    import main as arch_module  # type: ignore

    return arch_module


def _normalize_tokenizer(tok):
    """Fait cohabiter les deux noms d'attribut de fin de séquence rencontrés
    dans le projet (`eos_token_id` côté train_opsiom.py, `eot_token` côté
    main.py) en ajoutant l'alias manquant, et rend `.encode()` tolérant à un
    éventuel kwarg `allowed_special` (présent côté main.py, absent côté
    train_opsiom.py) pour que le reste du code (ModernLLM.generate(), et tout
    appelant de LLMClient) fonctionne quelle que soit l'origine du wrapper."""
    if hasattr(tok, "eos_token_id") and not hasattr(tok, "eot_token"):
        tok.eot_token = tok.eos_token_id
    elif hasattr(tok, "eot_token") and not hasattr(tok, "eos_token_id"):
        tok.eos_token_id = tok.eot_token

    _original_encode = tok.encode

    def _encode_compat(text: str, **kwargs):
        try:
            return _original_encode(text, **kwargs)
        except TypeError:
            # Version train_opsiom.py : encode(text) sans kwarg.
            return _original_encode(text)

    tok.encode = _encode_compat
    return tok


def _build_tokenizer(arch_module, hf_repo: str, local_path: str):
    """Instancie `FrenchTokenizerWrapper` avec la bonne interface, quelle que
    soit celle exposée par `arch_module` (voir note en tête de fichier)."""
    Wrapper = arch_module.FrenchTokenizerWrapper

    try:
        # Interface train_opsiom.py.
        tok = Wrapper(hf_repo=hf_repo, local_path=local_path)
    except TypeError:
        # Interface main.py : attend un objet `tokenizers.Tokenizer` déjà
        # chargé, pas des chemins/repo en kwargs.
        from tokenizers import Tokenizer

        raw = None
        if hf_repo:
            try:
                raw = Tokenizer.from_pretrained(hf_repo)
            except Exception:
                raw = None
        if raw is None:
            if not os.path.exists(local_path):
                raise FileNotFoundError(
                    f"Tokenizer introuvable : ni via le Hub HF ('{hf_repo}'), ni "
                    f"localement ('{local_path}'). Vérifie les chemins passés à "
                    f"LocalOpsiomClient."
                )
            raw = Tokenizer.from_file(local_path)
        tok = Wrapper(raw)

    return _normalize_tokenizer(tok)


class LLMClient(abc.ABC):
    """Interface minimale attendue par rag/ et reasoning/.

    Toute implémentation doit savoir transformer un prompt "brut" (déjà
    formaté avec les tags de dialogue si besoin) en texte généré.
    """

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_k: int | None = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.3,
        stop: list[str] | None = None,
    ) -> str:
        """Génère une complétion texte à partir d'un prompt déjà formaté."""
        raise NotImplementedError

    def chat(self, user_message: str, **gen_kwargs) -> str:
        """Emballe `user_message` avec les tags de dialogue configurés puis
        génère. Les implémentations concrètes définissent USER_TAG/ASSISTANT_TAG."""
        prompt = self.format_chat_prompt(user_message)
        raw = self.generate(prompt, **gen_kwargs)
        return self._strip_prompt(raw, prompt)

    # -- Aides communes, redéfinissables --------------------------------
    USER_TAG = "<|Utilisateur|>"
    ASSISTANT_TAG = "<|Assistant|>"

    def format_chat_prompt(self, user_message: str) -> str:
        return f"{self.USER_TAG}\n{user_message}\n{self.ASSISTANT_TAG}\n"

    @staticmethod
    def _strip_prompt(full_text: str, prompt: str) -> str:
        if full_text.startswith(prompt):
            return full_text[len(prompt):].strip()
        return full_text.strip()


@dataclass
class LocalOpsiomClient(LLMClient):
    """Charge le modèle et le tokenizer directement en mémoire.

    Fonctionne que le checkpoint et l'environnement d'exécution proviennent
    de train_opsiom.py ou de main.py (voir la note d'en-tête du fichier) :
    l'architecture et le tokenizer sont importés dynamiquement via
    `_import_architecture_module()`, et le tokenizer est normalisé via
    `_build_tokenizer()` / `_normalize_tokenizer()`.
    """

    checkpoint_path: str
    tokenizer_hf_repo: str = "JuloEco/opsiom-fr-tokenizer"
    tokenizer_local_path: str = "fr_bpe_tokenizer.json"
    device: str | None = None
    user_tag: str = "<|Utilisateur|>"
    assistant_tag: str = "<|Assistant|>"

    def __post_init__(self):
        import torch

        arch_module = _import_architecture_module()
        ModernLLM = arch_module.ModernLLM

        self._device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = _build_tokenizer(arch_module, self.tokenizer_hf_repo, self.tokenizer_local_path)

        ckpt = torch.load(self.checkpoint_path, map_location=self._device, weights_only=False)
        args = ckpt.get("args") if isinstance(ckpt, dict) else None
        if args is None or not hasattr(args, "vocab_size"):
            # Ne JAMAIS retomber sur un ModelArgs par défaut ici : les valeurs
            # par défaut diffèrent entre train_opsiom.py et main.py (ex: dim),
            # ce qui provoquerait un mismatch silencieux de dimensions au
            # chargement du state_dict. On préfère un échec explicite.
            raise ValueError(
                f"Le checkpoint '{self.checkpoint_path}' ne contient pas d'objet "
                "'args' exploitable — impossible de reconstruire l'architecture "
                "sans risquer un mismatch de dimensions."
            )
        args.device = self._device
        self.model = ModernLLM(args).to(self._device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.USER_TAG = self.user_tag
        self.ASSISTANT_TAG = self.assistant_tag

        print(
            f"✅ LocalOpsiomClient prêt (checkpoint='{self.checkpoint_path}', "
            f"step={ckpt.get('step', '?')}, val_loss={ckpt.get('val_loss', 'N/A')})"
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_k: int | None = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.3,
        stop: list[str] | None = None,
    ) -> str:
        text = self.model.generate(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        if stop:
            for s in stop:
                idx = text.find(s, len(prompt))
                if idx != -1:
                    text = text[:idx]
        return text


@dataclass
class APIOpsiomClient(LLMClient):
    """Appelle l'API Flask existante (README section 11 : POST /api/chat).

    Pratique quand `rag/` et `reasoning/` tournent dans un service séparé du
    modèle (ex: reasoning appelle RAG et le modèle via HTTP plutôt que de
    tout charger en mémoire).
    """

    base_url: str
    timeout_s: float = 60.0
    api_key: str | None = None
    user_tag: str = "<|Utilisateur|>"
    assistant_tag: str = "<|Assistant|>"

    def __post_init__(self):
        self.USER_TAG = self.user_tag
        self.ASSISTANT_TAG = self.assistant_tag

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_k: int | None = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.3,
        stop: list[str] | None = None,
    ) -> str:
        import requests

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "message": prompt,
            "raw_prompt": True,  # signale à l'API de ne pas ré-emballer avec les tags
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        }
        resp = requests.post(
            f"{self.base_url.rstrip('/')}/api/chat",
            json=payload,
            headers=headers,
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        text = resp.json()["response"]
        if stop:
            for s in stop:
                idx = text.find(s)
                if idx != -1:
                    text = text[:idx]
        return text
