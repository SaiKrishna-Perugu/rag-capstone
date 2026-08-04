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
