from unittest.mock import patch


def test_ingest_process():
    from app import ingest
    
    with patch("app.ingest._get_vector_store"):
        with patch("app.ingest._load_manifest") as mock_manifest:
            mock_manifest.return_value = {}
            # We aren't doing a full integration test here, just checking it imports
            assert hasattr(ingest, "run")
