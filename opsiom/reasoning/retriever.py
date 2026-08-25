# ============================================================================
# rag/retriever.py — Chunking, indexation, et recherche vectorielle (top-k)
#
# Implémente le flux du README section 7.3 (Chunking) et le début du flux de
# la section 7.1 :
#
#   Question → Query rewriting → Vector search → Top 20
#
# (Le re-ranking Top 20 → Top 3-5 vit dans rag/reranker.py, orchestré par
# rag/pipeline.py.)
# ============================================================================

from __future__ import annotations

import glob
import os
import re

from .database import Chunk, VectorDatabase
from .embeddings import EmbeddingModel


def _split_into_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def chunk_text(
    text: str,
    source: str,
    max_chars: int = 800,
    overlap_chars: int = 120,
) -> list[Chunk]:
    """Découpe un texte en chunks d'environ `max_chars` caractères, avec un
    recouvrement pour ne pas couper une idée pile à la frontière (README
    7.3 : Document → Section → Paragraphes → Chunks).

    Les paragraphes sont d'abord regroupés jusqu'à la limite de taille ; un
    paragraphe plus long que `max_chars` est découpé avec chevauchement.
    """
    paragraphs = _split_into_paragraphs(text)
    chunks: list[Chunk] = []
    buffer = ""
    chunk_idx = 0

    def flush():
        nonlocal buffer, chunk_idx
        if buffer.strip():
            chunks.append(
                Chunk(
                    text=buffer.strip(),
                    source=source,
                    chunk_id=f"{source}::{chunk_idx}",
                )
            )
            chunk_idx += 1
        buffer = ""

    for para in paragraphs:
        if len(para) > max_chars:
            flush()
            start = 0
            while start < len(para):
                end = start + max_chars
                chunks.append(
                    Chunk(text=para[start:end].strip(), source=source, chunk_id=f"{source}::{chunk_idx}")
                )
                chunk_idx += 1
                start = end - overlap_chars
            continue

        if len(buffer) + len(para) + 2 > max_chars:
            flush()
        buffer = f"{buffer}\n\n{para}" if buffer else para

    flush()
    return chunks


def load_documents_from_dir(root_dir: str, extensions: tuple[str, ...] = (".md", ".txt")) -> dict[str, str]:
    """Charge tous les fichiers texte/markdown d'un dossier (ex: `learncode/`
    cité en exemple dans le README section 7). Renvoie {chemin_relatif: contenu}."""
    docs: dict[str, str] = {}
    for ext in extensions:
        for path in glob.glob(os.path.join(root_dir, "**", f"*{ext}"), recursive=True):
            rel_path = os.path.relpath(path, root_dir)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                docs[rel_path] = f.read()
    return docs


class Retriever:
    """Construit et interroge la base vectorielle."""

    def __init__(self, embedding_model: EmbeddingModel | None = None, db: VectorDatabase | None = None):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.db = db or VectorDatabase(dim=self.embedding_model.dim)

    def index_documents(
        self,
        documents: dict[str, str],
        max_chars: int = 800,
        overlap_chars: int = 120,
        batch_size: int = 64,
    ) -> int:
        """Chunk + embed + ajoute à l'index. Renvoie le nombre de chunks ajoutés."""
        all_chunks: list[Chunk] = []
        for source, text in documents.items():
            all_chunks.extend(chunk_text(text, source=source, max_chars=max_chars, overlap_chars=overlap_chars))

        if not all_chunks:
            print("⚠️  Aucun chunk à indexer.")
            return 0

        vectors = self.embedding_model.encode([c.text for c in all_chunks], batch_size=batch_size)
        self.db.add(vectors, all_chunks)
        print(f"✅ {len(all_chunks):,} chunks indexés depuis {len(documents)} document(s).")
        return len(all_chunks)

    def index_directory(self, root_dir: str, **kwargs) -> int:
        docs = load_documents_from_dir(root_dir)
        return self.index_documents(docs, **kwargs)

    def search(self, query: str, top_k: int = 20) -> list[tuple[Chunk, float]]:
        """Recherche vectorielle brute (avant re-ranking)."""
        query_vector = self.embedding_model.encode_query(query)
        return self.db.search(query_vector, top_k=top_k)

    def save(self, dir_path: str) -> None:
        self.db.save(dir_path)

    @classmethod
    def load(cls, dir_path: str, embedding_model: EmbeddingModel | None = None) -> "Retriever":
        embedding_model = embedding_model or EmbeddingModel()
        db = VectorDatabase.load(dir_path)
        return cls(embedding_model=embedding_model, db=db)


def rewrite_query(raw_question: str, llm_client=None) -> str:
    """Query rewriting (README 7.1). Par défaut, renvoie la question telle
    quelle (nettoyée). Si un `llm_client` (voir model/llm_client.py) est
    fourni, on lui demande de reformuler la question en une requête de
    recherche autonome — utile par exemple pour résoudre les références
    ("et pour la version précédente ?") à partir de l'historique de
    conversation. Reste optionnel pour ne pas coupler RAG et modèle."""
    cleaned = raw_question.strip()
    if llm_client is None:
        return cleaned

    prompt = llm_client.format_chat_prompt(
        "Reformule la question suivante en une requête de recherche courte, "
        "autonome et sans ambiguïté, sans répondre à la question elle-même.\n\n"
        f"Question : {cleaned}\nRequête de recherche :"
    )
    rewritten = llm_client.generate(prompt, max_new_tokens=40, temperature=0.3)
    rewritten = llm_client._strip_prompt(rewritten, prompt).strip().strip('"')
    return rewritten or cleaned
