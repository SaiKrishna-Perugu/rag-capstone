"""Tests for the whole-document retrieval path (rag.retrieve()).

The path exists because chunk selection was measured getting a real question
wrong: "explain all the projects that I built" against an uploaded resume
returned 1 of 3 projects, because the full-text half of hybrid retrieval
matched "built" in the resume's EXPERIENCE bullets and displaced two project
chunks. These tests pin the decision boundary and the fallback, not the
retrieval quality that motivated it.
"""
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


def _row(content: str, source: str = "docs/uploads/s1/resume.txt", session: str = "s1"):
    return {"source": source, "content": content, "metadata": {}, "session_id": session}


def _curated(text: str = "curated passage"):
    return Document(page_content=text, metadata={"source": "docs/faq.md", "_session_id": None})


def test_small_upload_is_passed_whole_and_curated_is_still_retrieved():
    """The visitor's chunks all reach generation, AND the curated corpus is
    still searched -- dropping the second half would trade one recall bug for
    a quieter one."""
    from app.retrieval import rag

    rows = [_row("chunk one"), _row("chunk two"), _row("chunk three")]
    with patch("app.retrieval.hybrid.database.get_session_chunks", return_value=rows), \
         patch("app.retrieval.rag.retrieve_with_hybrid_and_rerank", return_value=[_curated()]) as mock_hybrid:
        chunks = rag.retrieve("explain all the projects that i built", session_id="s1")

    assert [c.page_content for c in chunks] == [
        "chunk one", "chunk two", "chunk three", "curated passage",
    ]
    # Curated retrieval must be scoped to the corpus, not the session --
    # otherwise the visitor's chunks are fetched twice and duplicated.
    assert mock_hybrid.call_args.kwargs["session_id"] is None


def test_oversized_upload_falls_back_to_hybrid_retrieval():
    """get_session_chunks() returns None over budget; retrieve() must then
    behave exactly as it did before this path existed."""
    from app.retrieval import rag

    with patch("app.retrieval.hybrid.database.get_session_chunks", return_value=None), \
         patch("app.retrieval.rag.retrieve_with_hybrid_and_rerank", return_value=[_curated()]) as mock_hybrid:
        chunks = rag.retrieve("a question", session_id="s1")

    assert [c.page_content for c in chunks] == ["curated passage"]
    assert mock_hybrid.call_args.kwargs["session_id"] == "s1"


def test_no_session_never_touches_the_whole_document_path():
    """A visitor with no X-Session-Id must not trigger a database call for
    session chunks -- the curated-only path is the common case."""
    from app.retrieval import rag

    with patch("app.retrieval.hybrid.database.get_session_chunks") as mock_get, \
         patch("app.retrieval.rag.retrieve_with_hybrid_and_rerank", return_value=[_curated()]):
        rag.retrieve("a question", session_id=None)

    mock_get.assert_not_called()


def test_whole_document_chunks_are_marked_private():
    """_session_id drives used_private_docs, which keeps the answer out of the
    shared semantic cache. Losing it here would leak one visitor's resume into
    another's cached answer."""
    from app.retrieval.hybrid import session_documents

    with patch("app.retrieval.hybrid.database.get_session_chunks", return_value=[_row("mine")]):
        docs = session_documents("s1")

    assert docs[0].metadata["_session_id"] == "s1"
    assert docs[0].metadata["source"] == "docs/uploads/s1/resume.txt"


def test_sources_are_deduped_per_document():
    """A six-chunk resume must render as one citation, not six identical ones."""
    from app.retrieval import rag

    chunks = [
        Document(page_content=f"part {i}", metadata={"source": "resume.txt", "_session_id": "s1"})
        for i in range(6)
    ]
    with patch("app.retrieval.rag.retrieve", return_value=chunks), \
         patch("app.retrieval.rag.generate_answer", return_value="an answer"), \
         patch("app.retrieval.rag.check_groundedness", return_value="GROUNDED"):
        result = rag.answer_question("q", session_id="s1")

    assert len(result.sources) == 1
    assert result.sources[0]["source"] == "resume.txt"
    # First chunk wins, so the excerpt is the top of the document.
    assert result.sources[0]["excerpt"] == "part 0"
    assert result.used_private_docs is True


def test_distinct_pages_of_one_pdf_stay_separate():
    """Dedup is per (source, page): a two-page PDF is two citations, because
    the page number is what makes a citation checkable."""
    from app.retrieval import rag

    chunks = [
        Document(page_content="page one", metadata={"source": "r.pdf", "page": 1}),
        Document(page_content="page two", metadata={"source": "r.pdf", "page": 2}),
        Document(page_content="page one again", metadata={"source": "r.pdf", "page": 1}),
    ]
    with patch("app.retrieval.rag.retrieve", return_value=chunks), \
         patch("app.retrieval.rag.generate_answer", return_value="a"), \
         patch("app.retrieval.rag.check_groundedness", return_value="GROUNDED"):
        result = rag.answer_question("q")

    assert [s["page"] for s in result.sources] == [1, 2]


def test_get_session_chunks_refuses_over_budget(monkeypatch):
    """The size guard is a separate aggregate query, so an oversized session
    costs one SUM rather than transferring every row and discarding it."""
    from app.db import database

    cur = MagicMock()
    cur.fetchone.return_value = {"total": 50_000}
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    ctx = MagicMock()
    ctx.__enter__.return_value = conn
    monkeypatch.setattr(database, "get_conn", lambda: ctx)

    assert database.get_session_chunks("s1", max_chars=12_000) is None
    # One aggregate query only -- the row fetch must not have run.
    assert cur.execute.call_count == 1


def test_get_session_chunks_returns_none_for_empty_session(monkeypatch):
    """None, not [], so retrieve() can tell 'too big' from 'nothing uploaded'
    -- both mean 'go and retrieve', but only one is a budget decision."""
    from app.db import database

    cur = MagicMock()
    cur.fetchone.return_value = {"total": 0}
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    ctx = MagicMock()
    ctx.__enter__.return_value = conn
    monkeypatch.setattr(database, "get_conn", lambda: ctx)

    assert database.get_session_chunks("s1", max_chars=12_000) is None


def test_whole_document_path_disabled_by_zero_budget():
    from app.db import database

    assert database.get_session_chunks("s1", max_chars=0) is None


def test_database_error_falls_back_instead_of_failing_the_request():
    """The whole-document lookup runs BEFORE hybrid_search(), so without a
    fail-open it would turn any database blip into a 500 raised from a new
    call site -- ahead of the error the retrieval path already reports.
    Matches cache.py's posture: an optimisation must not cost an answer."""
    from app.retrieval import rag

    with patch(
        "app.retrieval.hybrid.database.get_session_chunks",
        side_effect=RuntimeError("pool exhausted"),
    ), patch("app.retrieval.rag.retrieve_with_hybrid_and_rerank", return_value=[_curated()]):
        chunks = rag.retrieve("a question", session_id="s1")

    assert [c.page_content for c in chunks] == ["curated passage"]


def test_reranker_sees_the_whole_chunk(monkeypatch):
    """The candidate digest was truncated at 500 chars while CHUNK_SIZE is
    800, so the reranker judged every candidate with its tail invisible."""
    from app import config
    from app.retrieval import hybrid

    monkeypatch.setattr(config, "RERANKER_PROVIDER", "llm")
    long_chunk = "A" * (config.CHUNK_SIZE - 1) + "Z"
    doc = Document(page_content=long_chunk, metadata={"source": "s.txt"})
    captured = {}

    def _capture(messages):
        captured["prompt"] = messages[-1][1]
        resp = MagicMock()
        resp.content = "[1, 2]"
        return resp

    llm = MagicMock()
    llm.invoke.side_effect = _capture
    with patch("app.retrieval.hybrid.get_llm", return_value=llm):
        hybrid.rerank("q", [doc, Document(page_content="other", metadata={})], top_k=1)

    assert "Z" in captured["prompt"], "reranker never saw the end of the chunk"
