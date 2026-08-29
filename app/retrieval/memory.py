"""
Conversation Memory for RAG.

Handles tracking chat history by session_id and contextualizing follow-up
questions (e.g. rewriting "How much does it cost?" to "How much does the
Pro plan cost?" based on previous turns) before they hit the vector store.

Backed by Firestore -- one document per session_id in the
FIRESTORE_COLLECTION collection, holding the last 5 {question, answer}
turns and an `expires_at` timestamp for Firestore's native TTL (a
server-side policy configured once via `gcloud firestore fields ttls
update`, see README -- Python only sets the field). This replaces a
plain in-process dict, which meant every Cloud Run instance had its own
divergent chat history and a restart lost it entirely -- the same
stateless-instance problem Phase 1 fixed for the vector store.

Firestore is optional: if GCP_PROJECT_ID is unset and FIRESTORE_EMULATOR_HOST
isn't either, this module fails open at the client-construction level and
every call below behaves exactly as if no history existed -- so
MODEL_PROVIDER=groq deployments keep working with zero GCP setup. Real
Firestore errors (transient outage, permission issue) are also caught,
logged, and degrade to "no history" -- the same fail-open posture as
app/retrieval/cache.py and app/retrieval/rag.py's check_groundedness().
"""

import logging
import os
from datetime import UTC, datetime, timedelta

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app import config
from app.llm.providers import get_llm

logger = logging.getLogger(__name__)

_MAX_TURNS = 5

_CONTEXTUALIZE_PROMPT = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

_client = None


def _get_client():
    """Lazily construct the Firestore client. Returns None if Firestore
    isn't configured (no GCP project, no emulator) -- callers treat that
    as "no history available" rather than an error. Deliberately not
    memoized on failure (unlike success): a transient construction error
    should get retried on the next call, not permanently disable memory
    for the rest of the process."""
    global _client
    if _client is not None:
        return _client

    if not config.GCP_PROJECT_ID and not os.getenv("FIRESTORE_EMULATOR_HOST"):
        return None

    from google.cloud import firestore
    _client = firestore.Client(project=config.GCP_PROJECT_ID or None)
    return _client


def _apply_turn(txn, doc_ref, question: str, answer: str) -> None:
    """Append one turn to a session document, inside `txn`.

    Split out of add_to_history so the read-append-cap-write logic can be
    tested without standing up Firestore's transaction machinery. Called
    only from add_to_history's transactional wrapper.
    """
    snap = doc_ref.get(transaction=txn)
    turns = snap.to_dict().get("turns", []) if snap.exists else []
    turns.append({"question": question, "answer": answer})
    txn.set(doc_ref, {
        "turns": turns[-_MAX_TURNS:],
        # A computed future timestamp, NOT firestore.SERVER_TIMESTAMP --
        # that sentinel resolves to write-time, which would make every
        # session expire immediately under a TTL policy.
        "expires_at": datetime.now(UTC) + timedelta(hours=config.SESSION_TTL_HOURS),
    })


def add_to_history(session_id: str, question: str, answer: str) -> None:
    """Save a Q&A turn to the session's history."""
    if not session_id:
        return

    try:
        client = _get_client()
        if client is None:
            return

        # Imported here, not at module scope: google.cloud.firestore is an
        # optional dependency this module is designed to run without (see
        # _get_client's fail-open contract), so a top-level import would make
        # the whole module unimportable wherever Firestore is not installed.
        from google.cloud import firestore

        doc_ref = client.collection(config.FIRESTORE_COLLECTION).document(session_id)

        # Read-append-write inside a Firestore transaction. Done unguarded,
        # two turns arriving close together in the same session both read the
        # same list and the second set() silently discarded the first --
        # visible to the user as an answer vanishing from their history, and
        # as contextualize_question() rewriting a follow-up against a turn
        # that is no longer there. Firestore retries the callback on
        # contention, so the loser re-reads instead of overwriting.
        #
        # _apply_turn is module-level rather than a closure so it stays
        # directly testable. That matters more than it looks: this function
        # swallows every exception by design, so a transaction that silently
        # never commits would look identical to success from the outside --
        # exactly the failure this codebase has been bitten by before.
        txn = client.transaction()
        firestore.transactional(
            lambda t: _apply_turn(t, doc_ref, question, answer)
        )(txn)
    except Exception:
        logger.warning("Failed to persist conversation turn to Firestore.", exc_info=True)


def contextualize_question(session_id: str, question: str) -> str:
    """
    If there is chat history for the session, rewrite the question to be
    standalone. Otherwise, return the original question.
    """
    if not session_id:
        return question

    try:
        client = _get_client()
        if client is None:
            return question

        doc_ref = client.collection(config.FIRESTORE_COLLECTION).document(session_id)
        snap = doc_ref.get()
        history = snap.to_dict().get("turns", []) if snap.exists else []
    except Exception:
        logger.warning("Failed to read conversation history from Firestore.", exc_info=True)
        return question

    if not history:
        return question

    messages = [SystemMessage(content=_CONTEXTUALIZE_PROMPT)]

    for turn in history:
        messages.append(HumanMessage(content=turn["question"]))
        messages.append(AIMessage(content=turn["answer"]))

    messages.append(HumanMessage(content=question))

    # Labelled so contextualization shows up as its own line in the cost
    # breakdown -- it runs BEFORE the semantic cache is consulted, so it is
    # the one stage that still costs money on an otherwise-free cache hit.
    llm = get_llm(temperature=0.0, stage="contextualize")
    response = llm.invoke(messages)

    # The LLM returns the rewritten standalone question
    return response.content.strip()
