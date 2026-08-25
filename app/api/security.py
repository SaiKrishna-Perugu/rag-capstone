"""
Input screening, output screening, and PII redaction for logs.

Three separate concerns live here, deliberately, because each has a
different failure posture:

* ``screen_question()`` -- **gates**. A question carrying a recognised
  prompt-injection payload is refused before it reaches retrieval, so a
  blocked request costs zero LLM calls. Regex only: no network, no tokens,
  no added latency worth measuring.
* ``screen_answer()`` -- **repairs**. If an answer contains a fingerprint of
  one of this app's own system prompts, the model was talked into echoing
  its instructions; the answer is replaced rather than returned.
* ``redact_log_fields()`` -- **fails closed**, unlike almost everything else
  in this codebase. See the note on that function.

On precision over recall for injection patterns: this deployment is a public
demo whose entire purpose is being usable by a stranger, so a false positive
(refusing a real question) is a worse outcome than a false negative. The
generation prompt is already strict context-only, which makes the model
fairly resistant on its own -- screening is defence in depth, not the only
line. Patterns below therefore match explicit instruction-override and
prompt-exfiltration phrasing, not merely suspicious-sounding words.

**Known residual risk, not addressed here: indirect injection via uploaded
documents.** ``/upload`` accepts files from anonymous visitors, those files
become retrieved context, and context is fed to the LLM -- so a payload can
arrive in a document rather than in a question. Screening uploads was
considered and rejected: a corpus about security would be full of sentences
that look exactly like the patterns below, and rejecting documents for
discussing prompt injection is its own failure. ``screen_answer()`` is the
mitigation that still applies in that case, because it checks the *effect*
(instructions leaking into output) rather than the input text.
"""
import asyncio
import logging
import re
from dataclasses import dataclass

from app import config

logger = logging.getLogger(__name__)

# (compiled pattern, short reason recorded in logs/metrics)
_INJECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"ignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above|earlier|preceding)\s+"
                r"(?:instruction|prompt|rule|direction)s?", re.IGNORECASE), "override-instructions"),
    (re.compile(r"disregard\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above|earlier|system)\s+"
                r"(?:instruction|prompt|rule|direction)s?", re.IGNORECASE), "override-instructions"),
    (re.compile(r"forget\s+(?:everything|all)\s+(?:you|above|before|previously)", re.IGNORECASE),
     "override-instructions"),
    (re.compile(r"(?:reveal|show|print|repeat|output|display|tell\s+me)\s+(?:me\s+)?"
                r"(?:your|the)\s+(?:system\s+|initial\s+|original\s+)*(?:prompt|instruction)s?", re.IGNORECASE),
     "prompt-exfiltration"),
    (re.compile(r"what\s+(?:is|are|was|were)\s+your\s+"
                r"(?:system\s+|initial\s+|original\s+)*(?:prompt|instruction)s?", re.IGNORECASE),
     "prompt-exfiltration"),
    # Chat-template control tokens have no business in a user question --
    # their only purpose in this position is to forge a role boundary.
    (re.compile(r"<\|\s*(?:im_start|im_end|system|user|assistant|endoftext)\s*\|>", re.IGNORECASE),
     "control-token"),
    (re.compile(r"\[/?INST\]|<<SYS>>", re.IGNORECASE), "control-token"),
    (re.compile(r"(?:^|\n)\s*#{2,}\s*(?:system|instruction)", re.IGNORECASE), "control-token"),
    (re.compile(r"\bjailbreak\b|developer\s+mode\s+enabled|\bDAN\s+mode\b", re.IGNORECASE), "jailbreak"),
    (re.compile(r"you\s+are\s+now\s+(?:a|an|no\s+longer|in\s+)", re.IGNORECASE), "role-override"),
    (re.compile(r"(?:new|updated|revised)\s+(?:system\s+)?instructions\s*:", re.IGNORECASE), "role-override"),
]

# Distinctive fragments of this app's own system prompts (app/retrieval/rag.py,
# app/retrieval/hybrid.py, app/api/streaming.py). If one of these comes back inside an
# answer, the instructions leaked -- no legitimate corpus answer contains
# them, because they describe the assistant rather than the documents.
_PROMPT_FINGERPRINTS = (
    "you are a precise assistant that answers questions using only the provided context",
    "do not fabricate sources, numbers, or details not present in the context",
    "base your answer only on the context below",
    "you are a strict fact-checker",
    "you are a search relevance judge",
)

REFUSAL_MESSAGE = (
    "That request looks like an attempt to change how this assistant works "
    "rather than a question about the documents, so it was not run. Ask "
    "about the contents of the knowledge base instead."
)

LEAKED_PROMPT_REPLACEMENT = (
    "I don't have enough information in the provided documents to answer that."
)


@dataclass(frozen=True)
class ScreenVerdict:
    flagged: bool
    reason: str | None = None


CLEAN = ScreenVerdict(flagged=False)


def screen_question(question: str) -> ScreenVerdict:
    """Check a user question for prompt-injection payloads.

    Runs before retrieval and before any LLM call, so a blocked request
    consumes no tokens. Disabled entirely by ENABLE_INJECTION_SCREENING=false.
    """
    if not config.ENABLE_INJECTION_SCREENING or not question:
        return CLEAN

    for pattern, reason in _INJECTION_PATTERNS:
        if pattern.search(question):
            return ScreenVerdict(flagged=True, reason=reason)
    return CLEAN


def screen_answer(answer: str) -> ScreenVerdict:
    """Check a generated answer for leaked system-prompt text.

    This is the output half of the check, and the part that still helps when
    the payload arrived through an ingested document rather than through the
    question -- it looks at what came out, not what went in.
    """
    if not config.ENABLE_INJECTION_SCREENING or not answer:
        return CLEAN

    lowered = answer.lower()
    for fingerprint in _PROMPT_FINGERPRINTS:
        if fingerprint in lowered:
            return ScreenVerdict(flagged=True, reason="system-prompt-leak")
    return CLEAN


_dlp_client = None


def _get_dlp_client():
    """Lazily construct the DLP client -- same pattern as memory.py/jobs.py.
    Not memoized on failure, so a transient construction error is retried
    rather than permanently disabling redaction."""
    global _dlp_client
    if _dlp_client is None:
        from google.cloud import dlp_v2
        _dlp_client = dlp_v2.DlpServiceClient()
    return _dlp_client


async def redact_log_fields(fields: dict[str, str]) -> dict[str, str]:
    """De-identify text headed for the request log, via Cloud DLP.

    ``app/main.py`` logs the verbatim question and answer, which on a public
    demo is content typed by anonymous strangers. Retention (14 days, set on
    the _Default bucket) limits how long that is kept; this limits what is
    written in the first place.

    **This fails CLOSED, which is the opposite of cache.py/memory.py, and the
    difference is deliberate.** Those degrade to "no cache"/"no history"
    because the request still works without them. Here, degrading to "log it
    raw" would write exactly the PII this function exists to remove -- an
    availability fallback that defeats the control. So when DLP is enabled
    but unreachable, the field is replaced with a marker and a warning is
    logged. Debuggability is what gets sacrificed during a DLP outage, not
    the user's data.

    Inert (returns input unchanged) when ENABLE_PII_REDACTION is false, which
    is the default -- local dev and tests need no GCP setup, and the logs
    stay readable.

    Detector caveat, found by running this against the real API rather than
    a mock: DLP deliberately ignores well-known placeholder values. The
    canonical fake SSN "123-45-6789" is NOT detected at any likelihood
    threshold, while ordinary SSNs ("456-78-9012") come back VERY_LIKELY.
    So a test fixture using that famous placeholder proves nothing -- it
    passes against a mock and fails against production.

    One API call per log write regardless of field count: the fields go over
    as a single-row DLP table rather than one request each.

    Async because the actual DLP call is synchronous network I/O
    (`google-cloud-dlp`'s client has no asyncio variant); it runs via
    `asyncio.to_thread` so it doesn't block the event loop under
    concurrency -- six call sites across main.py/streaming.py, one per
    logged request, all now `await` this.
    """
    if not config.ENABLE_PII_REDACTION or not fields:
        return fields
    return await asyncio.to_thread(_redact_log_fields_blocking, fields)


def _redact_log_fields_blocking(fields: dict[str, str]) -> dict[str, str]:
    keys = list(fields)
    try:
        from google.cloud import dlp_v2

        client = _get_dlp_client()
        table = dlp_v2.Table(
            headers=[dlp_v2.FieldId(name=k) for k in keys],
            rows=[dlp_v2.Table.Row(
                values=[dlp_v2.Value(string_value=fields[k] or "") for k in keys]
            )],
        )
        response = client.deidentify_content(request={
            "parent": f"projects/{config.GCP_PROJECT_ID}/locations/global",
            "inspect_config": {
                "info_types": [{"name": t} for t in config.DLP_INFO_TYPES],
                "min_likelihood": getattr(
                    dlp_v2.Likelihood, config.DLP_MIN_LIKELIHOOD, dlp_v2.Likelihood.POSSIBLE
                ),
            },
            "deidentify_config": {
                "info_type_transformations": {
                    "transformations": [{
                        # Substitutes the finding with its type, e.g.
                        # "[US_SOCIAL_SECURITY_NUMBER]" -- so a redacted log
                        # still says what kind of thing was removed.
                        "primitive_transformation": {"replace_with_info_type_config": {}}
                    }]
                }
            },
            "item": {"table": table},
        })
        values = response.item.table.rows[0].values
        return {k: values[i].string_value for i, k in enumerate(keys)}
    except Exception:
        logger.warning(
            "Cloud DLP redaction failed; omitting these fields from the log "
            "rather than writing them unredacted.",
            exc_info=True,
        )
        return {k: "[redaction unavailable]" for k in keys}
