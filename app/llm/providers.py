"""
Model provider factory. Every place in the codebase that needs an LLM or
an embeddings model calls get_llm()/get_embeddings() from here instead of
constructing ChatGroq/FastEmbedEmbeddings directly -- that's what makes
switching providers a one-line config change (MODEL_PROVIDER in .env)
instead of a code change scattered across rag.py, agent.py, and eval*.py.

Supported providers:
  - "groq"     (default) -- uses GROQ_API_KEY, free tier gpt-oss-20b.
                Embeddings via FastEmbed (local ONNX, no API key needed).
  - "vertexai" -- uses GCP_PROJECT_ID / GCP_LOCATION, config.VERTEX_CHAT_MODEL /
                  VERTEX_EMBEDDING_MODEL. Requires `gcloud auth application-default
                  login` locally, or a service account when deployed on GCP.

get_llm() returns a client wrapped in two proxies, both of which exist
because this is the one place that already knows which provider is in play:

  _ResilientLLM(_CostTrackingLLM(real client))

_CostTrackingLLM records tokens and estimated spend (app/llm/cost.py).
_ResilientLLM applies the circuit breaker (app/llm/circuit.py) and, when
LLM_FALLBACK_PROVIDER is configured, routes to the other provider while the
primary's circuit is open -- the payoff for having built this abstraction
in the first place, since an outage no longer needs a manual redeploy with
a different MODEL_PROVIDER. Cost tracking sits *inside* failover so a call
served by the fallback is priced against the model that actually ran.

**Embeddings never fail over, and that is not an oversight.** The pgvector
store is built in one provider's embedding space -- 768-dim
text-embedding-005 for Vertex AI, 384-dim FastEmbed for Groq (see
config.EMBEDDING_DIMENSION). Embedding a query with the other provider
would either be rejected outright by pgvector on the dimension mismatch or,
where dimensions happen to agree, silently return nonsense neighbours. A
degraded chat provider is recoverable; a corrupted or unusable retrieval
path is not. So get_embeddings() stays pinned to MODEL_PROVIDER, and a
failover run keeps retrieving with the primary's embeddings while
generating with the fallback's chat model.
"""
import logging
from functools import lru_cache

from app import config, metrics
from app.llm import circuit, cost

logger = logging.getLogger(__name__)

_VALID_PROVIDERS = ("groq", "vertexai")


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


class _CostTrackingEmbeddings:
    """Thin proxy recording estimated embedding spend, same pattern as
    _CostTrackingLLM above and wrapping get_embeddings() for the same reason:
    before this, embedding calls -- the ones /upload's ingestion path makes,
    at a volume the caller controls -- were invisible to both per-request
    cost reporting and DAILY_BUDGET_USD (see llm/budget.py; add_usage() is
    its only feed).

    Unlike chat completions, LangChain's Embeddings interface returns no
    usage_metadata -- there is no token count to read off the response. Cost
    here is therefore a character-count estimate (~4 chars/token, the
    common rough ratio for English text), not an exact one -- consistent
    with cost.py's documented posture that these figures are for relative
    budgeting, not accounting; correct it via RAG_PRICE_<MODEL>_IN like any
    other entry once real Cloud Billing numbers are available. FastEmbed
    (Groq mode) prices at zero regardless, since it runs locally with no
    per-call cost to attribute.
    """

    _CHARS_PER_TOKEN = 4

    def __init__(self, wrapped, model: str):
        self._wrapped = wrapped
        self._model = model

    def _record(self, texts: list[str]) -> None:
        chars = sum(len(t) for t in texts)
        tokens = chars // self._CHARS_PER_TOKEN
        usd = cost.add_usage(self._model, tokens, 0, stage="embedding")
        metrics.record_llm_usage(self._model, tokens, 0, usd)

    def embed_documents(self, texts, *args, **kwargs):
        result = self._wrapped.embed_documents(texts, *args, **kwargs)
        self._record(texts)
        return result

    def embed_query(self, text, *args, **kwargs):
        result = self._wrapped.embed_query(text, *args, **kwargs)
        self._record([text])
        return result

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def _build_raw_client(provider: str, temperature: float):
    """Construct a provider's chat client. Returns (client, model_name) --
    the model name is needed separately for cost attribution, which must
    follow whichever provider actually served the call."""
    if provider == "groq":
        from langchain_groq import ChatGroq
        return (
            ChatGroq(
                model=config.GROQ_CHAT_MODEL,
                api_key=config.GROQ_API_KEY,
                temperature=temperature,
                max_retries=config.LLM_MAX_RETRIES,
                timeout=config.LLM_REQUEST_TIMEOUT,
            ),
            config.GROQ_CHAT_MODEL,
        )

    if provider == "vertexai":
        from langchain_google_vertexai import ChatVertexAI
        return (
            ChatVertexAI(
                model_name=config.VERTEX_CHAT_MODEL,
                project=config.GCP_PROJECT_ID,
                location=config.GCP_LOCATION,
                temperature=temperature,
                max_retries=config.LLM_MAX_RETRIES,
                timeout=config.LLM_REQUEST_TIMEOUT,
            ),
            config.VERTEX_CHAT_MODEL,
        )

    raise ValueError(f"Unknown MODEL_PROVIDER: {provider}")


def _build_client(provider: str, temperature: float, stage: str) -> "_CostTrackingLLM":
    raw, model = _build_raw_client(provider, temperature)
    return _CostTrackingLLM(raw, model, stage)


def _resolve_fallback_provider() -> str | None:
    """The provider to fail over to, or None if failover is off/unusable.

    Misconfiguration disables failover with a warning rather than raising.
    Failover is a resilience add-on; refusing to start the whole service
    over a bad value for it would be a worse failure than running without
    it -- and it would turn a typo in an optional env var into an outage.
    """
    name = config.LLM_FALLBACK_PROVIDER
    if not name:
        return None

    if name not in _VALID_PROVIDERS:
        logger.warning(
            f"LLM_FALLBACK_PROVIDER={name!r} is not one of {_VALID_PROVIDERS}; "
            "failover disabled."
        )
        return None

    if name == config.MODEL_PROVIDER:
        logger.warning(
            f"LLM_FALLBACK_PROVIDER={name!r} matches MODEL_PROVIDER; failing over "
            "to the same provider would achieve nothing. Failover disabled."
        )
        return None

    # Credentials are checked here, not on first use: a fallback that turns
    # out to be unusable mid-outage is the worst possible moment to find
    # out, and this way the warning lands in the logs at startup instead.
    if name == "groq" and not config.GROQ_API_KEY:
        logger.warning("LLM_FALLBACK_PROVIDER=groq but GROQ_API_KEY is unset; failover disabled.")
        return None
    if name == "vertexai" and not config.GCP_PROJECT_ID:
        logger.warning("LLM_FALLBACK_PROVIDER=vertexai but GCP_PROJECT_ID is unset; failover disabled.")
        return None

    return name


class _ResilientLLM:
    """Circuit breaker + optional cross-provider failover around a client.

    Three outcomes per call:

    * Circuit closed (or probing) and the call succeeds -- pass straight
      through, reset the breaker.
    * Circuit closed and the call fails -- count the failure, then serve
      from the fallback if one is configured, else re-raise.
    * Circuit open -- skip the primary entirely. Serve from the fallback,
      or raise CircuitOpenError immediately. This is the fail-fast case
      the breaker exists for: no retries, no 60s timeout, no held thread.

    Anything other than invoke()/astream() falls through __getattr__ to the
    primary untouched -- same caveat as _CostTrackingLLM, a new invocation
    path would bypass both the breaker and cost tracking.
    """

    def __init__(self, primary, primary_provider: str, fallback_provider: str | None,
                 temperature: float, stage: str):
        self._primary = primary
        self._primary_provider = primary_provider
        self._fallback_provider = fallback_provider
        self._temperature = temperature
        self._stage = stage
        self._fallback = None

    def _get_fallback(self):
        """Built on first need, not up front: most deployments never fail
        over, and constructing the second client eagerly would authenticate
        against a provider that may never be called."""
        if self._fallback_provider is None:
            return None
        if self._fallback is None:
            try:
                self._fallback = _build_client(
                    self._fallback_provider, self._temperature, self._stage
                )
            except Exception as exc:
                # Don't let this mask the real problem -- the caller is
                # already handling a primary failure, and that exception is
                # the more useful one to propagate.
                logger.error(
                    f"Fallback provider {self._fallback_provider!r} could not be "
                    f"constructed: {exc}",
                    exc_info=True,
                )
                return None
        return self._fallback

    def _on_primary_failure(self, breaker, exc: Exception) -> None:
        logger.warning(
            f"LLM call to {self._primary_provider!r} failed "
            f"({type(exc).__name__}: {exc})"
        )
        if breaker.record_failure():
            metrics.record_circuit_opened(self._primary_provider)

    def _fallback_for_open_circuit(self):
        """Fallback client to use while the circuit is open, or raise."""
        fallback = self._get_fallback()
        if fallback is None:
            raise circuit.CircuitOpenError(
                f"Circuit for provider {self._primary_provider!r} is open; "
                "failing fast instead of calling it."
            )
        metrics.record_llm_failover(self._primary_provider, self._fallback_provider)
        return fallback

    def invoke(self, *args, **kwargs):
        breaker = circuit.get_breaker(self._primary_provider)

        if not breaker.allow_request():
            return self._fallback_for_open_circuit().invoke(*args, **kwargs)

        try:
            result = self._primary.invoke(*args, **kwargs)
        except Exception as exc:
            self._on_primary_failure(breaker, exc)
            fallback = self._get_fallback()
            if fallback is None:
                raise
            metrics.record_llm_failover(self._primary_provider, self._fallback_provider)
            return fallback.invoke(*args, **kwargs)

        breaker.record_success()
        return result

    async def astream(self, *args, **kwargs):
        breaker = circuit.get_breaker(self._primary_provider)

        if not breaker.allow_request():
            async for chunk in self._fallback_for_open_circuit().astream(*args, **kwargs):
                yield chunk
            return

        yielded_any = False
        try:
            async for chunk in self._primary.astream(*args, **kwargs):
                yielded_any = True
                yield chunk
        except Exception as exc:
            self._on_primary_failure(breaker, exc)
            # Once tokens have reached the client there is no transparent
            # recovery: the fallback would start the answer from the top and
            # the reader would watch the response restart mid-sentence.
            # Only a failure before the first token can be re-routed
            # invisibly. After that, re-raise and let streaming.py emit its
            # SSE error payload.
            if yielded_any:
                raise
            fallback = self._get_fallback()
            if fallback is None:
                raise
            metrics.record_llm_failover(self._primary_provider, self._fallback_provider)
            async for chunk in fallback.astream(*args, **kwargs):
                yield chunk
            return

        breaker.record_success()

    def __getattr__(self, name):
        return getattr(self._primary, name)


@lru_cache(maxsize=16)
def get_llm(temperature: float = 0.0, stage: str = "llm"):
    """`stage` labels this client's calls in the per-request cost breakdown
    (see app/llm/cost.py) -- "rerank", "generate", "groundedness". It is a
    parameter rather than a chained builder so that tests patching
    `get_llm` keep receiving their configured mock directly, which is the
    mocking convention used throughout tests/.

    The primary client is built eagerly here, so an unknown MODEL_PROVIDER
    still surfaces as a ValueError from this call rather than being
    deferred to the first invoke().
    """
    primary = _build_client(config.MODEL_PROVIDER, temperature, stage)
    return _ResilientLLM(
        primary,
        config.MODEL_PROVIDER,
        _resolve_fallback_provider(),
        temperature,
        stage,
    )


@lru_cache(maxsize=1)
def get_embeddings():
    if config.MODEL_PROVIDER == "groq":
        # Groq doesn't offer an embeddings API, so we use FastEmbed --
        # lightweight local ONNX-based embeddings. No API key, no torch.
        # The model (~33MB) is downloaded once on first run to a local cache.
        from langchain_community.embeddings import FastEmbedEmbeddings
        raw = FastEmbedEmbeddings(
            model_name=config.GROQ_EMBEDDING_MODEL,
            cache_dir=config.FASTEMBED_CACHE_PATH,
        )
        return _CostTrackingEmbeddings(raw, config.GROQ_EMBEDDING_MODEL)

    if config.MODEL_PROVIDER == "vertexai":
        from langchain_google_vertexai import VertexAIEmbeddings
        raw = VertexAIEmbeddings(
            model_name=config.VERTEX_EMBEDDING_MODEL,
            project=config.GCP_PROJECT_ID,
            location=config.GCP_LOCATION,
        )
        return _CostTrackingEmbeddings(raw, config.VERTEX_EMBEDDING_MODEL)

    raise ValueError(f"Unknown MODEL_PROVIDER: {config.MODEL_PROVIDER}")
