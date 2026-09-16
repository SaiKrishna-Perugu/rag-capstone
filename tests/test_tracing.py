"""One LangSmith trace per request, with the visitor's session tag stripped.

Before app/tracing.py only /ask-agentic produced a request-shaped trace; the
UI's default /ask-stream reached LangSmith as unconnected ChatVertexAI runs,
two per request, with nothing tying them together. And the agentic trace
recorded ``session_id`` -- the visitor's X-Session-Id, the value that scopes
their uploads -- in its inputs and in every LangGraph node's state.

These tests drive the real tracing machinery offline: the client's
send-side methods are patched, so what they capture is exactly what would
have left the process, after hide_inputs/hide_outputs have run.
"""
import asyncio
from typing import TypedDict
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tracers.langchain import wait_for_all_tracers
from langgraph.graph import END, StateGraph
from langsmith import Client
from langsmith.run_helpers import tracing_context

from app import tracing

_SECRET = "3f0c9a2e-visitor-tag"


def test_redact_replaces_session_ids_at_any_depth():
    out = tracing.redact({
        "question": "refund window?",
        "session_id": _SECRET,
        "state": {"doc_session_id": _SECRET, "chunks": [{"session_id": _SECRET}]},
    })
    assert _SECRET not in str(out)
    assert out["question"] == "refund window?"
    assert out["session_id"] == tracing.REDACTED


def test_redact_keeps_an_absent_session_distinguishable():
    """None stays None: whether the visitor had uploads matters when reading a trace."""
    assert tracing.redact({"session_id": None}) == {"session_id": None}


@pytest.fixture
def sent():
    """Everything the redacting client would have transmitted."""
    captured = []
    with patch.object(Client, "_create_run", lambda self, run, **kw: captured.append(("create", run))), \
         patch.object(Client, "_update_run", lambda self, run, **kw: captured.append(("update", run))):
        client = Client(
            api_key="test", api_url="http://127.0.0.1:9", auto_batch_tracing=False,
            hide_inputs=tracing.redact, hide_outputs=tracing.redact,
        )
        with patch("app.tracing.config.LANGSMITH_TRACING", True), \
             patch("app.tracing.client", return_value=client):
            yield captured


def _created(sent):
    return [run for kind, run in sent if kind == "create"]


def test_stream_root_contains_threaded_and_streamed_calls(sent):
    """/ask-stream runs retrieval and the groundedness check through
    asyncio.to_thread and streams generation from inside the generator. All
    of it must land under the one root, not as separate traces."""
    llm = FakeListChatModel(responses=["retrieved", "answer text"])

    def in_worker_thread(question):
        return llm.invoke(question).content

    @tracing.traced("ask-stream")
    async def stream(question, doc_session_id=None):
        await asyncio.to_thread(in_worker_thread, question)
        async for chunk in llm.astream(question):
            yield chunk.content

    async def run():
        with tracing_context(enabled=True):
            return [x async for x in stream("q", doc_session_id=_SECRET)]

    asyncio.run(run())
    wait_for_all_tracers()

    runs = _created(sent)
    roots = [r for r in runs if not r.get("parent_run_id")]
    assert [r["name"] for r in roots] == ["ask-stream"]
    children = [r for r in runs if r.get("parent_run_id")]
    assert len(children) == 2, [r["name"] for r in runs]
    assert all(str(r["parent_run_id"]) == str(roots[0]["id"]) for r in children)
    assert _SECRET not in str(sent)


class _State(TypedDict):
    question: str
    session_id: str | None
    answer: str


def test_langgraph_node_runs_are_redacted_too(sent):
    """LangGraph traces its nodes itself, recording the whole state -- so
    redacting only our own decorators' inputs would still leak the tag."""
    llm = FakeListChatModel(responses=["an answer"])

    def generate(state):
        return {**state, "answer": llm.invoke(state["question"]).content}

    graph = StateGraph(_State)
    graph.add_node("generate", generate)
    graph.set_entry_point("generate")
    graph.add_edge("generate", END)
    compiled = graph.compile()

    @tracing.traced("agentic_rag.run")
    def run(question, session_id=None):
        return compiled.invoke({"question": question, "session_id": session_id, "answer": ""})

    with tracing_context(enabled=True):
        run("q", session_id=_SECRET)
    wait_for_all_tracers()

    names = {r["name"] for r in _created(sent)}
    assert "generate" in names, names  # LangGraph's own node run was captured
    assert _SECRET not in str(sent)


def _wrapper_chain(fn):
    while fn is not None:
        yield fn
        fn = getattr(fn, "__wrapped__", None)


def test_every_answer_path_opens_a_trace():
    """Dropping any of these decorators silently reverts that mode to
    loose, unattributable LLM runs -- which is how /ask-stream shipped."""
    from app import main
    from app.api import streaming
    from app.retrieval import agent, rag

    for fn in (streaming.stream_answer, rag.answer_question, agent.run_agentic_rag,
               main.ask, main.ask_agentic):
        assert any(hasattr(f, "__langsmith_traceable__") for f in _wrapper_chain(fn)), fn.__name__


def test_tracing_adds_no_parameters_to_the_api():
    """traceable adds a keyword-only `config` parameter to what it wraps;
    FastAPI would publish that as a query parameter on the route."""
    from app.main import app

    schema = app.openapi()
    for path in ("/ask", "/ask-agentic"):
        params = {p["name"] for p in schema["paths"][path]["post"].get("parameters", [])}
        assert "config" not in params, (path, params)


def test_the_suite_itself_never_traces():
    """A developer .env with LANGSMITH_TRACING=true must not turn this mocked
    suite into real LangSmith traffic; conftest.py switches it off."""
    from langsmith.utils import tracing_is_enabled

    assert tracing_is_enabled() is False
