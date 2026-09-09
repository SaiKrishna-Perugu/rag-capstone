from unittest.mock import MagicMock, patch


def _chunk(text: str = "The sky is blue."):
    chunk = MagicMock()
    chunk.page_content = text
    chunk.metadata = {"source": "sky.txt"}
    return chunk


# NOTE: these patch `app.retrieval.rag.get_llm`, not `app.llm.providers.get_llm` -- rag.py
# binds the name at import time, so patching the source module would leave the
# real function in place. They also deliberately skip the `mock_groundedness`
# fixture, which replaces `check_groundedness` itself: a test using it would
# assert against the mock rather than the function under test.

def test_check_groundedness_supported():
    # This tests that our check_groundedness function works and can parse LLM output
    from app.retrieval.rag import check_groundedness

    with patch("app.retrieval.rag.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "GROUNDED"
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        result = check_groundedness("The sky is blue.", [_chunk()])
        assert result == "GROUNDED"

def test_check_groundedness_llm_failure_returns_not_checked():
    # A transient failure on the *verification* call must not propagate --
    # the caller already has a usable answer.
    from app.retrieval.rag import check_groundedness

    with patch("app.retrieval.rag.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("upstream LLM 503")
        mock_get_llm.return_value = mock_llm

        assert check_groundedness("The sky is blue.", [_chunk()]) == "NOT_CHECKED"

def test_answer_question_survives_groundedness_failure(mock_retrieval, mock_llm_answer):
    # The end-to-end guarantee: a dead groundedness check degrades the verdict,
    # it does not lose the answer.
    from app.retrieval.rag import answer_question

    with patch("app.retrieval.rag.get_llm", side_effect=RuntimeError("upstream LLM 503")):
        result = answer_question("What is the refund policy?")

    assert result.answer == "The refund policy is 30 days."
    assert result.groundedness == "NOT_CHECKED"

def test_generate_answer(mock_llm_answer):
    from app.retrieval.rag import generate_answer
    mock_chunk = MagicMock()
    mock_chunk.page_content = "Context"
    
    ans = generate_answer("Question?", [mock_chunk])
    assert ans == "The refund policy is 30 days."


# --- Groundedness sampling ------------------------------------------------
# The check is a whole extra LLM call over the same context. Sampling trades
# per-request verdicts for aggregate signal; "SKIPPED" must stay distinct
# from "NOT_CHECKED", which means the check ran and failed.

def test_groundedness_runs_on_every_request_by_default():
    """Default rate is 1.0 -- this ships inert, no behaviour change."""
    from app import config
    from app.retrieval import rag

    with patch.object(config, "GROUNDEDNESS_SAMPLE_RATE", 1.0):
        with patch("app.retrieval.rag.retrieve", return_value=[_chunk()]):
            with patch("app.retrieval.rag.generate_answer", return_value="Blue."):
                with patch("app.retrieval.rag.check_groundedness", return_value="GROUNDED") as chk:
                    result = rag.answer_question("colour?")

    chk.assert_called_once()
    assert result.groundedness == "GROUNDED"


def test_rate_zero_skips_the_check_without_calling_the_llm():
    from app import config
    from app.retrieval import rag

    with patch.object(config, "GROUNDEDNESS_SAMPLE_RATE", 0.0):
        with patch("app.retrieval.rag.retrieve", return_value=[_chunk()]):
            with patch("app.retrieval.rag.generate_answer", return_value="Blue."):
                with patch("app.retrieval.rag.check_groundedness") as chk:
                    result = rag.answer_question("colour?")

    chk.assert_not_called()
    assert result.groundedness == "SKIPPED"


def test_skipped_is_distinct_from_not_checked():
    """check_hallucination=False means 'never asked for'; SKIPPED means sampled out."""
    from app import config
    from app.retrieval import rag

    with patch.object(config, "GROUNDEDNESS_SAMPLE_RATE", 0.0):
        with patch("app.retrieval.rag.retrieve", return_value=[_chunk()]):
            with patch("app.retrieval.rag.generate_answer", return_value="Blue."):
                with patch("app.retrieval.rag.check_groundedness"):
                    opted_out = rag.answer_question("colour?", check_hallucination=False)

    assert opted_out.groundedness == "NOT_CHECKED"


def test_partial_rate_uses_the_sampler():
    from app import config
    from app.retrieval import rag

    with patch.object(config, "GROUNDEDNESS_SAMPLE_RATE", 0.5):
        # random() < 0.5 -> checked; >= 0.5 -> skipped.
        with patch("app.retrieval.rag.random.random", return_value=0.9):
            assert rag._should_check_groundedness() is False
        with patch("app.retrieval.rag.random.random", return_value=0.1):
            assert rag._should_check_groundedness() is True


# ---------------------------------------------------------------------------
# Citation dedup, and the parity that keeps it true on all three answer paths.
#
# Dedup was added in 1207d69 to rag.answer_question() and agent.node_generate()
# and missed on api/streaming.py, so /ask and /ask-agentic showed one card per
# document while /ask-stream still showed six identical cards for a six-chunk
# resume -- the exact symptom the dedup was written for, and one the
# whole-document retrieval path makes the norm rather than the exception.
# ---------------------------------------------------------------------------

def _chunk_with(source: str, page, text: str):
    chunk = MagicMock()
    chunk.page_content = text
    chunk.metadata = {"source": source, "page": page}
    return chunk


def test_build_sources_dedupes_by_source_and_page():
    from app.retrieval.rag import build_sources
    chunks = [
        _chunk_with("resume.pdf", None, "first chunk"),
        _chunk_with("resume.pdf", None, "second chunk"),
        _chunk_with("resume.pdf", 2, "page two"),
        _chunk_with("policy.txt", None, "other doc"),
    ]
    sources = build_sources(chunks)
    assert [(s["source"], s["page"]) for s in sources] == [
        ("resume.pdf", None), ("resume.pdf", 2), ("policy.txt", None),
    ]


def test_build_sources_keeps_the_first_chunk_of_each_key():
    """First wins, so the excerpt is the highest-ranked chunk on the retrieval
    path and the opening of the document on the whole-document path."""
    from app.retrieval.rag import build_sources
    sources = build_sources([
        _chunk_with("a.md", None, "highest ranked"),
        _chunk_with("a.md", None, "lower ranked"),
    ])
    assert len(sources) == 1
    assert sources[0]["excerpt"] == "highest ranked"


def test_all_three_answer_paths_share_one_build_sources():
    """Structural parity, not a behavioural echo: assert the SAME function
    object backs every path. A future re-inlined copy fails here rather than
    quietly reintroducing the drift this test exists to prevent."""
    from app.api import streaming
    from app.retrieval import agent, rag
    assert agent.build_sources is rag.build_sources
    assert streaming.build_sources is rag.build_sources
