import os
from unittest.mock import patch

import pytest


def test_config_defaults():
    # Import config and test some default values are set
    from app import config
    assert config.MODEL_PROVIDER in ["groq", "vertexai"]
    assert isinstance(config.CHUNK_SIZE, int)
    assert isinstance(config.CORS_ORIGINS, list)

@patch.dict(os.environ, {"MODEL_PROVIDER": "invalid"}, clear=True)
def test_invalid_model_provider():
    # Verify that providers.py raises ValueError on invalid provider
    from app import config
    from app.providers import get_llm
    config.MODEL_PROVIDER = "invalid"
    with pytest.raises(ValueError, match="Unknown MODEL_PROVIDER: invalid"):
        get_llm.cache_clear()
        get_llm()

def test_fastembed_cache_path_used(monkeypatch):
    # get_embeddings() must pass config.FASTEMBED_CACHE_PATH through to
    # FastEmbedEmbeddings -- this is what lets the Dockerfile bake the model
    # into the image at a known path instead of every cold container
    # instance redownloading it from Hugging Face at runtime (see Dockerfile
    # comments). A silent drop of this kwarg would reintroduce that bug.
    from app import config
    from app.providers import get_embeddings

    monkeypatch.setattr(config, "MODEL_PROVIDER", "groq")
    monkeypatch.setattr(config, "FASTEMBED_CACHE_PATH", "/tmp/custom_fastembed_cache")
    get_embeddings.cache_clear()

    try:
        with patch("langchain_community.embeddings.FastEmbedEmbeddings") as mock_cls:
            get_embeddings()
            _, kwargs = mock_cls.call_args
            assert kwargs["cache_dir"] == "/tmp/custom_fastembed_cache"
    finally:
        get_embeddings.cache_clear()
