"""
LangSmith trace roots, and the redaction every traced run passes through.

LangChain and LangGraph trace their own calls once LANGSMITH_TRACING is on
(config.py), but only an enclosing @traceable run groups those calls into one
trace per request. Until this module existed only /ask-agentic had one:
/ask-stream -- the UI's default mode -- reached LangSmith as unconnected
ChatVertexAI runs, with no sign of retrieval, the cache, or which request
they belonged to.

Every root is bound to one Client, and nested LangChain/LangGraph runs
inherit their parent's client (langchain_core's callback manager passes
``run_tree.client`` to the tracer), so that client's hide_inputs/hide_outputs
hooks see every run in the tree -- including LangGraph's own node runs, which
record the whole AgentState. That matters because ``session_id`` there is the
visitor's X-Session-Id: the value that scopes, and so grants access to, their
uploads. It is replaced with a marker that still shows whether one was set.

Prompts and retrieved passages are deliberately NOT redacted -- they are the
reason to trace at all. Enabling LANGSMITH_TRACING in production therefore
sends visitors' uploaded text to LangSmith; see CLAUDE.md.
"""

import inspect
from functools import lru_cache

from langsmith import Client, traceable

from app import config

REDACTED = "[redacted]"
_PRIVATE_KEYS = frozenset({"session_id", "doc_session_id"})


def redact(value):
    """Replace session identifiers at any depth; leave everything else alone."""
    if isinstance(value, dict):
        return {
            k: (REDACTED if k in _PRIVATE_KEYS and v else redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list | tuple):
        return [redact(v) for v in value]
    return value


@lru_cache(maxsize=1)
def client() -> Client:
    return Client(hide_inputs=redact, hide_outputs=redact)


def traced(name: str, run_type: str = "chain", **kwargs):
    """@traceable bound to the redacting client.

    The client is only built when tracing is on: constructing one without an
    API key logs LangSmithMissingAPIKeyWarning, which is noise in local runs
    and tests, where tracing is off and traceable is a pass-through anyway.
    """
    if config.LANGSMITH_TRACING:
        kwargs["client"] = client()
    return traceable(name=name, run_type=run_type, **kwargs)


def _endpoint_inputs(inputs: dict) -> dict:
    # The Request object is not serialisable and carries every header,
    # X-Session-Id included. The parsed body is what a trace reader needs.
    return {k: v for k, v in inputs.items() if k != "request"}


def traced_endpoint(name: str):
    """traced() for a FastAPI route handler.

    traceable adds a keyword-only ``config`` parameter to the wrapper's
    signature, and FastAPI builds a route's request contract from that
    signature -- it would publish ``config`` as a new query parameter.
    Restoring the handler's own signature keeps the API unchanged.
    """
    def decorate(fn):
        wrapped = traced(name, process_inputs=_endpoint_inputs)(fn)
        wrapped.__signature__ = inspect.signature(fn)
        return wrapped
    return decorate
