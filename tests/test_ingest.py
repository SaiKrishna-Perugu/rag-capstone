from unittest.mock import patch


def test_ingest_process():
    """Verify ingest module imports and has the expected interface."""
    from app import ingest

    with patch("app.ingest.database") as mock_database:
        mock_database.get_manifest.return_value = {}
        # We aren't doing a full integration test here, just checking it imports
        assert hasattr(ingest, "run")


def test_ingest_chunk_documents():
    """Verify chunk_documents splits text correctly."""
    from unittest.mock import MagicMock

    from app.ingest import chunk_documents

    doc = MagicMock()
    doc.page_content = "Hello world. " * 200  # long enough to split
    doc.metadata = {"source": "test.txt"}

    chunks = chunk_documents([doc])
    assert len(chunks) >= 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 800 + 50  # chunk_size + tolerance
