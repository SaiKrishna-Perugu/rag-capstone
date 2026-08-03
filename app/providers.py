"""
Model provider factory. Every place in the codebase that needs an LLM or
an embeddings model calls get_llm()/get_embeddings() from here instead of
constructing ChatGroq/FastEmbedEmbeddings directly -- that's what makes
switching providers a one-line config change (MODEL_PROVIDER in .env)
instead of a code change scattered across rag.py, agent.py, and eval*.py.

Supported providers:
  - "groq"     (default) -- uses GROQ_API_KEY, free tier Llama 3.3 70B.
                Embeddings via FastEmbed (local ONNX, no API key needed).
  - "vertexai" -- uses GCP_PROJECT_ID / GCP_LOCATION, config.VERTEX_CHAT_MODEL /
                  VERTEX_EMBEDDING_MODEL. Requires `gcloud auth application-default
                  login` locally, or a service account when deployed on GCP.
"""
from functools import lru_cache

from app import config


@lru_cache(maxsize=8)
def get_llm(temperature: float = 0.0):
    if config.MODEL_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=config.GROQ_CHAT_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=temperature,
        )

    if config.MODEL_PROVIDER == "vertexai":
        from langchain_google_vertexai import ChatVertexAI
        return ChatVertexAI(
            model_name=config.VERTEX_CHAT_MODEL,
            project=config.GCP_PROJECT_ID,
            location=config.GCP_LOCATION,
            temperature=temperature,
        )

    raise ValueError(f"Unknown MODEL_PROVIDER: {config.MODEL_PROVIDER}")


@lru_cache(maxsize=1)
def get_embeddings():
    if config.MODEL_PROVIDER == "groq":
        # Groq doesn't offer an embeddings API, so we use FastEmbed --
        # lightweight local ONNX-based embeddings. No API key, no torch.
        # The model (~33MB) is downloaded once on first run to a local cache.
        from langchain_community.embeddings import FastEmbedEmbeddings
        return FastEmbedEmbeddings(model_name=config.GROQ_EMBEDDING_MODEL)

    if config.MODEL_PROVIDER == "vertexai":
        from langchain_google_vertexai import VertexAIEmbeddings
        return VertexAIEmbeddings(
            model_name=config.VERTEX_EMBEDDING_MODEL,
            project=config.GCP_PROJECT_ID,
            location=config.GCP_LOCATION,
        )

    raise ValueError(f"Unknown MODEL_PROVIDER: {config.MODEL_PROVIDER}")
