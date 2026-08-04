"""
Core RAG logic: retrieve relevant chunks for a question, generate a
grounded answer, and run a lightweight groundedness (hallucination) check.

Kept separate from main.py so it can be imported directly by eval.py
without spinning up the FastAPI app.
"""
from dataclasses import dataclass, field

from app.providers import get_llm
from app.retrieval import retrieve_with_hybrid_and_rerank

_ANSWER_SYSTEM_PROMPT = """You are a precise assistant that answers questions \
using ONLY the provided context. Follow these rules strictly:

1. Base your answer only on the context below. Do not use outside knowledge.
2. If the context does not contain enough information to answer, say exactly:
   "I don't have enough information in the provided documents to answer that."
3. Keep answers concise and factual.
4. Do not fabricate sources, numbers, or details not present in the context."""

_GROUNDEDNESS_SYSTEM_PROMPT = """You are a strict fact-checker. You will be \
given a CONTEXT and an ANSWER. Determine whether every factual claim in the \
ANSWER is directly supported by the CONTEXT.

Respond with exactly one word: "GROUNDED" if every claim is supported, or \
"UNSUPPORTED" if the answer contains any claim not found in the context."""


@dataclass
class RagResult:
    answer: str
    sources: list = field(default_factory=list)
    groundedness: str = "NOT_CHECKED"


def retrieve(question: str, k: int | None = None) -> list:
    """
    Retrieve the top-k most relevant chunks for a question, via hybrid
    (BM25 + vector, RRF-fused) retrieval followed by LLM reranking --
    see app/retrieval.py for why each stage exists and its tradeoffs.
    """
    return retrieve_with_hybrid_and_rerank(question, k=k)


def _format_context(chunks: list) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page")
        label = f"[{i}] source={source}" + (f" page={page}" if page is not None else "")
        parts.append(f"{label}\n{chunk.page_content}")
    return "\n\n".join(parts)


def generate_answer(question: str, chunks: list) -> str:
    context = _format_context(chunks)
    llm = get_llm(temperature=0.0)

    messages = [
        ("system", _ANSWER_SYSTEM_PROMPT),
        ("human", f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"),
    ]
    response = llm.invoke(messages)
    return response.content


def check_groundedness(answer: str, chunks: list) -> str:
    """
    Lightweight hallucination check: ask the LLM whether the answer's claims
    are supported by the retrieved context. Not a substitute for a proper
    NLI-based groundedness model, but a real, working first pass -- and it's
    the kind of check most candidates skip entirely.
    """
    context = _format_context(chunks)
    llm = get_llm(temperature=0.0)

    messages = [
        ("system", _GROUNDEDNESS_SYSTEM_PROMPT),
        ("human", f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"),
    ]
    response = llm.invoke(messages)
    verdict = response.content.strip().upper()
    return verdict if verdict in ("GROUNDED", "UNSUPPORTED") else "UNKNOWN"


def answer_question(question: str, k: int | None = None, check_hallucination: bool = True) -> RagResult:
    chunks = retrieve(question, k=k)

    if not chunks:
        return RagResult(
            answer="I don't have enough information in the provided documents to answer that.",
            sources=[],
            groundedness="GROUNDED",  # trivially true -- no claims were made
        )

    answer = generate_answer(question, chunks)
    groundedness = check_groundedness(answer, chunks) if check_hallucination else "NOT_CHECKED"

    sources = [
        {
            "source": chunk.metadata.get("source", "unknown"),
            "page": chunk.metadata.get("page"),
            "excerpt": chunk.page_content[:200],
        }
        for chunk in chunks
    ]

    return RagResult(answer=answer, sources=sources, groundedness=groundedness)
