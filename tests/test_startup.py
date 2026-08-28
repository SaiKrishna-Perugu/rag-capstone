"""Startup provider warmup.

The measurement behind this: on the live service `/health` answered in
0.35s while the very next `/ask` took 20.6s, and every request after that
was under 0.7s. The latency read like a container cold start and was
actually lazy initialisation -- `get_embeddings()` builds its client,
acquires credentials and opens a connection on first use.

Note these tests drive TestClient as a context manager. The rest of the
suite uses the bare `client` fixture, which does NOT run lifespan, so
warmup never fires there and no test makes a real provider call.
"""
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import config
from app.main import _warm_providers, app


def test_warmup_thread_starts_when_enabled(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_STARTUP_WARMUP", True)
    with patch("app.main.database.init_db"), \
         patch("app.main.threading.Thread") as thread:
        with TestClient(app):
            pass
    assert thread.call_count == 1
    assert thread.call_args.kwargs["target"] is _warm_providers
    # Daemon, or a hung provider call would hold shutdown open.
    assert thread.call_args.kwargs["daemon"] is True
    thread.return_value.start.assert_called_once()


def test_warmup_is_skipped_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_STARTUP_WARMUP", False)
    with patch("app.main.database.init_db"), \
         patch("app.main.threading.Thread") as thread:
        with TestClient(app):
            pass
    assert thread.call_count == 0


def test_warmup_calls_embeddings_and_builds_the_llm():
    with patch("app.llm.providers.get_embeddings") as emb, \
         patch("app.llm.providers.get_llm") as llm:
        _warm_providers()

    emb.return_value.embed_query.assert_called_once()
    llm.assert_called_once()


def test_warmup_fails_open_when_the_provider_is_unreachable():
    """Warmup is off the request path, so an unreachable provider must
    degrade to the old lazy behaviour rather than stop the service starting.
    Both halves are guarded independently: a broken embeddings client must
    not skip the LLM construction that follows it."""
    with patch("app.llm.providers.get_embeddings", side_effect=RuntimeError("no creds")), \
         patch("app.llm.providers.get_llm", side_effect=RuntimeError("no creds")) as llm:
        _warm_providers()  # must not raise

    llm.assert_called_once()
