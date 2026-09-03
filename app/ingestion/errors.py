"""Turn an ingestion exception into something a visitor can act on.

Every failure below used to reach the browser as `str(exc)`. `/jobs/{id}`
returns the job record verbatim and `ui.html` renders `job.error` directly,
so a Vertex AI client error arrived at an anonymous visitor complete with the
request URL, the project id and the upload bucket's name -- for a file whose
only problem was being slightly too long. That is two failures at once: the
message says nothing the reader can do, and it describes infrastructure to
someone who should never have seen it.

So classification happens here, once, and the pair it returns is used
deliberately:

- `message` goes in the job record, and therefore to the browser. It names
  what the visitor can change and nothing else.
- The original exception stays in the logs, with the job id, which is where
  someone debugging this should be looking anyway.

Matching is on substrings of the provider's message rather than exception
types on purpose. Groq and Vertex raise entirely different classes for the
same condition -- the same reasoning `llm/circuit.py` gives for counting
consecutive failures instead of trying to tell an outage from a bad request.
A miss costs a generic message, never a wrong one.
"""
from dataclasses import dataclass

# Ordered: the first pattern that matches wins, so specific conditions must
# precede the broad ones. "quota" appears in some token-limit messages too,
# which is why the size patterns are checked first.
_RULES: list[tuple[str, tuple[str, ...], str]] = [
    (
        "document_too_large",
        ("input token count", "supports up to", "too many tokens",
         "exceeds the maximum", "request payload size", "is too long"),
        (
            "This document is too large to process in one piece. Try splitting it "
            "into smaller files, or uploading only the sections you want to ask "
            "about."
        ),
    ),
    (
        "session_quota_exceeded",
        ("per-visitor limit", "session chunk budget"),
        (
            "This document is too large for one visitor's share of the demo "
            "knowledge base. Try a shorter document, or upload only the "
            "section you want to ask about."
        ),
    ),
    (
        "provider_busy",
        ("429", "resource_exhausted", "rate limit", "quota exceeded",
         "too many requests"),
        (
            "The service is handling more requests than usual right now. Wait a "
            "moment and upload the file again."
        ),
    ),
    (
        "provider_unavailable",
        ("503", "500 internal", "unavailable", "deadline exceeded", "timed out",
         "timeout", "connection reset"),
        (
            "The document processing service is temporarily unavailable. Your file "
            "was not saved -- please try uploading it again shortly."
        ),
    ),
    (
        "database_unavailable",
        ("psycopg2", "operationalerror", "could not connect", "connection refused",
         "server closed the connection", "pool exhausted"),
        (
            "The knowledge base is temporarily unreachable, so this file could not "
            "be indexed. Please try again shortly."
        ),
    ),
    (
        "storage_unavailable",
        ("no such object", "not present on this instance", "bucket",
         "forbidden", "403"),
        (
            "The uploaded file could not be retrieved for processing. Please "
            "upload it again."
        ),
    ),
    (
        "unsupported_content",
        ("no supported files", "cannot open", "could not read file", "not a pdf",
         "eof marker", "startxref", "unicodedecodeerror", "codec can't decode",
         "malformed", "corrupt", "parse", "badzipfile", "no text"),
        (
            "This file could not be read. It may be corrupt, password-protected, "
            "or a scanned image with no selectable text."
        ),
    ),
]

_FALLBACK = (
    "internal",
    (
        "Something went wrong while processing this file. If it keeps happening, "
        "try a different file or a smaller one."
    ),
)


@dataclass(frozen=True)
class IngestError:
    """A classified failure. `code` is for logs and metrics and is stable
    across message rewording; `message` is what the visitor reads."""

    code: str
    message: str


def classify(exc: BaseException | str) -> IngestError:
    """Best-effort classification. Never raises -- a failure here would
    replace a real error with a different one, at the exact moment someone
    is trying to find out what went wrong."""
    try:
        text = str(exc).lower()
    except Exception:
        return IngestError(*_FALLBACK)

    for code, needles, message in _RULES:
        if any(n in text for n in needles):
            return IngestError(code, message)
    return IngestError(*_FALLBACK)
