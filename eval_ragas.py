"""
RAGAS-based evaluation harness -- replaces the custom LLM-as-judge in
eval.py with RAGAS's standardized, widely-used RAG metrics:

  - Faithfulness:            does the answer's claims actually come from the
                              retrieved context? (RAGAS's own hallucination check)
  - ResponseRelevancy:       does the answer actually address the question?
  - LLMContextPrecisionWithoutReference: are the retrieved chunks relevant
                              to the question (i.e. was retrieval precise)?
  - LLMContextRecall:        did retrieval surface what was needed to answer
                              correctly? (requires a reference answer)

Why RAGAS instead of (or alongside) the custom judge in eval.py: these are
named, standardized metrics that anyone in this space recognizes -- "we
scored 0.87 faithfulness" means something specific and comparable, whereas
a custom PASS/FAIL judge is bespoke to this project. Keep eval.py too --
having both is a legitimate talking point ("I built a lightweight custom
check first, then adopted the standard framework").

Run:
    python eval_ragas.py

Edit EVAL_SET below with real questions/reference answers based on your
own docs/ files before running -- same placeholders-must-be-replaced rule
as eval.py.

--------------------------------------------------------------------------
COMPATIBILITY NOTE (read this if imports fail on your machine):
As of the versions pinned in requirements.txt, RAGAS unconditionally
imports a legacy `langchain_community.chat_models.vertexai` shim at
import time, even if you never touch Vertex AI. Recent `langchain-community`
releases removed that shim as part of their ongoing deprecation ("community
is being sunset"), which breaks `import ragas` with:
    ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
This is a real, currently-open upstream compatibility gap between two
fast-moving libraries -- not a mistake in this codebase. The shim below
works around it without touching your installed packages: if the real
module is missing, we register a stand-in in sys.modules pointing at
`langchain-google-vertexai`'s ChatVertexAI (already in requirements.txt)
before ragas is imported anywhere. If you upgrade ragas later and this
starts failing differently, that means upstream finally fixed it and this
whole shim block can be deleted.
--------------------------------------------------------------------------
"""
import sys
import types

# --- Compatibility shim: must run BEFORE any `import ragas` -----------------
try:
    import langchain_community.chat_models.vertexai  # noqa: F401
except ModuleNotFoundError:
    from langchain_google_vertexai import ChatVertexAI

    _shim = types.ModuleType("langchain_community.chat_models.vertexai")
    _shim.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _shim
# -----------------------------------------------------------------------------

import json
from pathlib import Path

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig

from app.providers import get_embeddings, get_llm
from app.rag import generate_answer, retrieve

# ---------------------------------------------------------------------------
# Same placeholder pattern as eval.py -- replace with real Q&A pairs based
# on your ingested documents. `reference` is required for LLMContextRecall
# (it needs a known-correct answer to check whether retrieval found what
# was necessary), so don't skip it.
# ---------------------------------------------------------------------------
EVAL_SET = [
    {
        "question": "How long is the refund window?",
        "reference": "Customers may request a full refund within 30 days of purchase, provided the product is unused and in its original packaging.",
    },
    {
        "question": "Are digital products refundable?",
        "reference": "Digital products are non-refundable once downloaded.",
    },
    {
        "question": "How long does standard shipping take?",
        "reference": "Standard shipping takes 5-8 business days within the country of purchase.",
    },
    {
        "question": "What does the hardware warranty cover?",
        "reference": "All hardware products come with a 1-year limited warranty covering manufacturing defects. The warranty does not cover accidental damage, water damage, or unauthorized repairs.",
    },
    {
        "question": "How do I reset my password?",
        "reference": "Go to Settings > Security > Reset Password. A reset link will be emailed to your registered address, valid for 30 minutes.",
    },
    {
        "question": "What payment methods are accepted?",
        "reference": "Visa, Mastercard, American Express, and PayPal. Bank transfers are supported for enterprise accounts only.",
    },
    {
        "question": "When is the scheduled maintenance window?",
        "reference": "The platform is unavailable for scheduled maintenance on the first Sunday of every month, from 2:00 AM to 4:00 AM UTC.",
    },
    # Negative test case
    {
        "question": "What is the CEO's name?",
        "reference": "The documents do not contain information about the CEO's name.",
    },
]


class _RagasCompatibleEmbeddings:
    """Thin proxy around whatever app.providers.get_embeddings() returns,
    fixing a real RAGAS bug confirmed by reading ragas/embeddings/base.py:
    LangchainEmbeddingsWrapper.embed_query()/embed_documents() call the
    real embedding first (succeeds), then do
    `getattr(self.embeddings, "model", None)` to build a telemetry event
    -- and pydantic-validates that as a string. FastEmbedEmbeddings uses
    `.model` for the *loaded model object* (the string name lives in
    `.model_name` instead), so that validation always raises, discarding
    the already-successful embedding and crashing every ResponseRelevancy
    computation. Confirmed 100% reproducible, not intermittent -- every
    single sample failed with the same ValidationError before this fix.

    Only exposes `.model` as a plain string for RAGAS's telemetry to find;
    everything else (embed_query, embed_documents, ...) delegates to the
    real embeddings object untouched via __getattr__.
    """

    def __init__(self, wrapped):
        self._wrapped = wrapped
        self.model = getattr(wrapped, "model_name", None) or "unknown"

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def build_dataset(eval_set: list) -> EvaluationDataset:
    samples = []
    for case in eval_set:
        chunks = retrieve(case["question"])
        contexts = [c.page_content for c in chunks]
        answer = generate_answer(case["question"], chunks)

        samples.append(
            SingleTurnSample(
                user_input=case["question"],
                retrieved_contexts=contexts,
                response=answer,
                reference=case["reference"],
            )
        )
    return EvaluationDataset(samples=samples)


def run_ragas_eval() -> dict:
    dataset = build_dataset(EVAL_SET)

    llm = LangchainLLMWrapper(get_llm(temperature=0.0))
    embeddings = LangchainEmbeddingsWrapper(_RagasCompatibleEmbeddings(get_embeddings()))

    metrics = [
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=embeddings),
        LLMContextPrecisionWithoutReference(llm=llm),
        LLMContextRecall(llm=llm),
    ]

    # max_workers is held low because RAGAS's default 16-way concurrency
    # overwhelms rate limits; the wall-clock cost is worth the reliability.
    #
    # timeout=300 (was 90): RAGAS degrades a timed-out job to NaN rather
    # than failing loudly, and check_thresholds.py correctly treats NaN as
    # a failure -- so a too-short timeout looks exactly like a quality
    # collapse. That is what happened on the first Vertex run:
    # context_precision came back NaN with every one of its jobs logging
    # TimeoutError, while faithfulness and context_recall both scored 1.0.
    # LLMContextPrecisionWithoutReference is the most timeout-prone metric
    # here because it makes one LLM call per retrieved context (TOP_K=4),
    # so its jobs do roughly 4x the work of the others within the same
    # per-job budget.
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        show_progress=False,
        run_config=RunConfig(max_workers=3, timeout=300),
    )
    return result


if __name__ == "__main__":
    # RAGAS/pandas output can include non-ASCII characters (e.g. "->").
    # Windows' default console encoding (cp1252) can't print those and
    # raises UnicodeEncodeError -- confirmed via a live run that otherwise
    # completed successfully and crashed only at this print step. Linux
    # CI runners default to UTF-8 already, so this is a no-op there.
    sys.stdout.reconfigure(encoding="utf-8")

    if any(case["question"].startswith("REPLACE ME") for case in EVAL_SET):
        print(
            "eval_ragas.py: EVAL_SET still contains placeholder questions.\n"
            "Edit EVAL_SET with real questions/reference answers based on "
            "your docs/ files before running.",
            file=sys.stderr,
        )
        sys.exit(1)

    result = run_ragas_eval()

    df = result.to_pandas()
    print("\n=== RAGAS Eval Summary ===")
    print(df.to_string(index=False))

    scores = {
        col: round(float(df[col].mean()), 3)
        for col in df.columns
        if df[col].dtype.kind in "fc"  # float/complex columns = metric scores
    }
    print("\nMean scores:")
    for name, score in scores.items():
        print(f"  {name}: {score}")

    out_path = Path("ragas_results.json")
    out_path.write_text(json.dumps(
        {"mean_scores": scores, "per_sample": df.to_dict(orient="records")},
        indent=2, default=str,
    ))
    print(f"\nFull results written to {out_path}")
