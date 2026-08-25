# ============================================================================
# rag/pipeline.py — Orchestrateur RAG de bout en bout (README section 7)
#
#   Question → Query rewriting → Vector search → Top 20 → Re-ranking →
#   Top 3-5 → Contexte → Opsiom → Réponse
# ============================================================================

from __future__ import annotations

from dataclasses import dataclass

from .database import Chunk
from .reranker import Reranker
from .retriever import Retriever, rewrite_query

CONTEXT_PROMPT_TEMPLATE = """Tu es Opsiom, un assistant qui répond en te basant UNIQUEMENT sur le \
contexte fourni ci-dessous. Si le contexte ne contient pas la réponse, dis-le \
clairement plutôt que d'inventer.

Contexte :
{context}

Question : {question}
Réponse (cite tes sources entre crochets, ex: [source.md]) :"""


@dataclass
class RAGResult:
    answer: str
    context_chunks: list[Chunk]
    sources: list[str]


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker | None = None,
        llm_client=None,
        top_k_search: int = 20,
        top_k_final: int = 5,
    ):
        self.retriever = retriever
        self.reranker = reranker or Reranker()
        self.llm_client = llm_client
        self.top_k_search = top_k_search
        self.top_k_final = top_k_final

    def retrieve_context(self, question: str) -> list[Chunk]:
        """Étapes 1 à 5 du flux README (jusqu'au "Contexte"), sans appeler
        le modèle génératif — utile pour l'endpoint /api/rag qui peut vouloir
        renvoyer juste les passages pertinents (README, exemple section 7)."""
        rewritten = rewrite_query(question, llm_client=self.llm_client)
        candidates = self.retriever.search(rewritten, top_k=self.top_k_search)
        reranked = self.reranker.rerank(question, candidates, top_k=self.top_k_final)
        return [chunk for chunk, _ in reranked]

    @staticmethod
    def format_context(chunks: list[Chunk]) -> str:
        parts = []
        for chunk in chunks:
            header = f"[{chunk.source}]"
            if chunk.section:
                header += f" ({chunk.section})"
            parts.append(f"{header}\n{chunk.text}")
        return "\n\n---\n\n".join(parts)

    def answer(self, question: str, **gen_kwargs) -> RAGResult:
        """Flux complet, y compris la génération par Opsiom (README, section
        7 : "Le modèle peut alors répondre même si cette information
        n'existait pas dans ses poids")."""
        if self.llm_client is None:
            raise RuntimeError(
                "RAGPipeline.answer() nécessite un llm_client (voir model/llm_client.py). "
                "Utilise retrieve_context() seul si tu veux juste les passages pertinents."
            )

        chunks = self.retrieve_context(question)
        if not chunks:
            fallback = "Je n'ai trouvé aucun document pertinent pour répondre à cette question."
            return RAGResult(answer=fallback, context_chunks=[], sources=[])

        context = self.format_context(chunks)
        prompt = self.llm_client.format_chat_prompt(
            CONTEXT_PROMPT_TEMPLATE.format(context=context, question=question)
        )
        raw = self.llm_client.generate(prompt, **gen_kwargs)
        answer = self.llm_client._strip_prompt(raw, prompt)

        sources = sorted({chunk.source for chunk in chunks})
        return RAGResult(answer=answer, context_chunks=chunks, sources=sources)
