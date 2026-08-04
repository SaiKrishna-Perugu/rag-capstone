from unittest.mock import patch, MagicMock
from pathlib import Path

def test_ingest_process():
    from app import ingest
    
    with patch("app.ingest._get_vector_store") as mock_vs:
        with patch("app.ingest._load_manifest") as mock_manifest:
            mock_manifest.return_value = {}
            # We aren't doing a full integration test here, just checking it imports
            assert hasattr(ingest, "run")
