from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(autouse=True)
def _disable_rate_limit():
    """Turn slowapi off for the whole suite.

    RATE_LIMIT is 10/minute in production, and slowapi keys on client IP --
    which is one shared value for every TestClient request. Without this, the
    suite starts returning 429 partway through and the failures land on
    whichever tests happen to run last, not on whatever is actually broken.
    Rate limiting is production behaviour worth having; it is not something
    unit tests should be fighting.
    """
    from app.main import limiter
    limiter.enabled = False
    yield
    limiter.enabled = True


@pytest.fixture
def client():
    """Test client for FastAPI app."""
    return TestClient(app)

@pytest.fixture
def mock_cache():
    """Mocks the semantic cache to always return None (cache miss)."""
    with patch("app.cache.get_cached_answer", return_value=None) as mock_get:
        with patch("app.cache.set_cached_answer") as mock_set:
            yield mock_get, mock_set

@pytest.fixture
def mock_retrieval():
    """Mocks retrieval to return dummy documents."""
    mock_doc = MagicMock()
    mock_doc.page_content = "This is a mock document about the refund policy. It is 30 days."
    mock_doc.metadata = {"source": "mock.pdf"}
    
    with patch("app.rag.retrieve_with_hybrid_and_rerank", return_value=[mock_doc]) as mock:
        yield mock

@pytest.fixture
def mock_llm_answer():
    """Mocks generate_answer to return a fixed string."""
    with patch("app.rag.generate_answer", return_value="The refund policy is 30 days.") as mock:
        yield mock

@pytest.fixture
def mock_groundedness():
    """Mocks groundedness check to return GROUNDED."""
    with patch("app.rag.check_groundedness", return_value="GROUNDED") as mock:
        yield mock
