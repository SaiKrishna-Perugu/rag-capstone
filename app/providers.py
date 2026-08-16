"""
Model provider factory. Every place in the codebase that needs an LLM or
an embeddings model calls get_llm()/get_embeddings() from here instead of
constructing ChatGroq/FastEmbedEmbeddings directly -- that's what makes
switching providers a one-line config change (MODEL_PROVIDER in .env)
instead of a code change scattered across rag.py, agent.py, and eval*.py.

Supported providers:
  - "groq"     (default) -- uses GROQ_API_KEY, free tier Llama 3.3 70B.
                Embeddings via FastEmbed (local ONNX, no API key needed).
  - "vertexai" -- uses GCP_PROJECT_ID / GCP_LOCATION, config.VERTEX_CHAT_MODEL /
                  VERTEX_EMBEDDING_MODEL. Requires `gcloud auth application-default
                  login` locally, or a service account when deployed on GCP.
"""
import logging
from functools import lru_cache

from app import config, cost, metrics

logger = logging.getLogger(__name__)


class _CostTrackingLLM:
    """Thin proxy recording token usage and estimated cost per call.

    Wrapping here rather than at each call site keeps cost accounting in
    the one place that already knows which model is in use, and means a
    new caller cannot forget to instrument itself. Same delegation pattern
    as `_RagasCompatibleEmbeddings` in eval_ragas.py.

    Both invocation paths the app actually uses are covered: `invoke()`
    (retrieval rerank, generation, groundedness, agent grading/rewriting)
    and `astream()` (`/ask-stream`). Anything else falls through
    `__getattr__` to the real client untouched -- and would go unmeasured,
    so add an override here if a new path appears.
    """

    def __init__(self, wrapped, model: str, stage: str = "llm"):
        self._wrapped = wrapped
        self._model = model
        self._stage_name = stage

    def _record(self, message) -> None:
        # usage_metadata is LangChain's provider-agnostic shape. Absent on
        # some providers/streaming modes -- treated as "unknown", not zero
        # tokens, by simply not recording: a missing measurement should not
        # masquerade as a free call.
        usage = getattr(message, "usage_metadata", None)
        if not usage:
            return
        try:
            in_tok = int(usage.get("input_tokens", 0))
            out_tok = int(usage.get("output_tokens", 0))
        except (AttributeError, TypeError, ValueError):
            return
        usd = cost.add_usage(self._model, in_tok, out_tok, stage=self._stage_name)
        metrics.record_llm_usage(self._model, in_tok, out_tok, usd)

    def invoke(self, *args, **kwargs):
        result = self._wrapped.invoke(*args, **kwargs)
        self._record(result)
        return result

    async def astream(self, *args, **kwargs):
        # Accumulate chunks so the final merged message carries the usage
        # totals, while still yielding each chunk through untouched -- the
        # streaming UX must not change to get measured.
        merged = None
        async for chunk in self._wrapped.astream(*args, **kwargs):
            merged = chunk if merged is None else merged + chunk
            yield chunk
        if merged is not None:
            self._record(merged)

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


@lru_cache(maxsize=16)
def get_llm(temperature: float = 0.0, stage: str = "llm"):
    """`stage` labels this client's calls in the per-request cost breakdown
    (see app/cost.py) -- "rerank", "generate", "groundedness". It is a
    parameter rather than a chained builder so that tests patching
    `get_llm` keep receiving their configured mock directly, which is the
    mocking convention used throughout tests/."""
    if config.MODEL_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return _CostTrackingLLM(
            ChatGroq(
                model=config.GROQ_CHAT_MODEL,
                api_key=config.GROQ_API_KEY,
                temperature=temperature,
                max_retries=config.LLM_MAX_RETRIES,
                timeout=config.LLM_REQUEST_TIMEOUT,
            ),
            config.GROQ_CHAT_MODEL,
            stage,
        )

    if config.MODEL_PROVIDER == "vertexai":
        from langchain_google_vertexai import ChatVertexAI
        return _CostTrackingLLM(
            ChatVertexAI(
                model_name=config.VERTEX_CHAT_MODEL,
                project=config.GCP_PROJECT_ID,
                location=config.GCP_LOCATION,
                temperature=temperature,
                max_retries=config.LLM_MAX_RETRIES,
                timeout=config.LLM_REQUEST_TIMEOUT,
            ),
            config.VERTEX_CHAT_MODEL,
            stage,
        )

    raise ValueError(f"Unknown MODEL_PROVIDER: {config.MODEL_PROVIDER}")


@lru_cache(maxsize=1)
def get_embeddings():
    if config.MODEL_PROVIDER == "groq":
        # Groq doesn't offer an embeddings API, so we use FastEmbed --
        # lightweight local ONNX-based embeddings. No API key, no torch.
        # The model (~33MB) is downloaded once on first run to a local cache.
        from langchain_community.embeddings import FastEmbedEmbeddings
        return FastEmbedEmbeddings(
            model_name=config.GROQ_EMBEDDING_MODEL,
            cache_dir=config.FASTEMBED_CACHE_PATH,
        )

    if config.MODEL_PROVIDER == "vertexai":
        from langchain_google_vertexai import VertexAIEmbeddings
        return VertexAIEmbeddings(
            model_name=config.VERTEX_EMBEDDING_MODEL,
            project=config.GCP_PROJECT_ID,
            location=config.GCP_LOCATION,
        )

    raise ValueError(f"Unknown MODEL_PROVIDER: {config.MODEL_PROVIDER}")
