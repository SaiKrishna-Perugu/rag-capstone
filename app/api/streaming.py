"""
Streaming generator for SSE (Server-Sent Events) response.
"""

import asyncio
import json
import logging
import time
import uuid

from app import metrics
from app.api import security
from app.llm import cost
from app.llm.providers import get_llm
from app.retrieval import cache, memory
from app.retrieval.rag import (
    _ANSWER_SYSTEM_PROMPT,
    _format_context,
    check_groundedness,
    retrieve,
)

logger = logging.getLogger("rag_service")

async def stream_answer(
    question: str,
    session_id: str | None = None,
    top_k: int | None = None,
    doc_session_id: str | None = None,
):
    """
    Async generator that yields SSE strings containing JSON payloads.
    It retrieves context, streams tokens as they are generated, and
    sends a final payload with groundedness and sources.
    """
    request_id = str(uuid.uuid4())
    metrics.record_request("ask-stream")
    # Started HERE, inside the generator, not in main.py's route handler.
    # cost.py accumulates in a ContextVar, and an async generator runs in
    # the context of whoever resumes it -- so the set must happen in the
    # same body that later does the retrieval, the astream() call, and the
    # log write, or those three would each see a different accumulator.
    cost.start_request()
    start = time.perf_counter()

    # Screened before the SSE stream opens, so a refused request is a clean
    # error payload rather than a half-streamed answer that stops mid-token.
    verdict = security.screen_question(question)
    if verdict.flagged:
        metrics.record_injection_blocked(verdict.reason)
        logger.warning(json.dumps({
            "request_id": request_id, "event": "injection_blocked",
            "endpoint": "ask-stream", "reason": verdict.reason,
        }))
        yield f"data: {json.dumps({'error': security.REFUSAL_MESSAGE, 'reason': verdict.reason, 'request_id': request_id})}\n\n"
        return

    try:
        # 1. Contextualize the question based on chat history
        contextualized_q = await asyncio.to_thread(
            memory.contextualize_question, session_id, question
        )
        
        # 2. Check semantic cache -- skipped for visitors with their own
        #    documents, so an upload is never shadowed by a globally cached
        #    answer (see main.py::_session_has_uploads).
        cached_hit = None
        if not await asyncio.to_thread(cache.session_has_uploads, doc_session_id):
            cached_hit = await asyncio.to_thread(cache.get_cached_answer, contextualized_q)
        if cached_hit:
            # Save to memory even if it was a cache hit
            if session_id:
                await asyncio.to_thread(memory.add_to_history, session_id, question, cached_hit["answer"])
                
            latency_ms = int((time.perf_counter() - start) * 1000)
            metrics.record_latency(latency_ms)
            
            logger.info(json.dumps({
                "request_id": request_id,
                "event": "ask-stream",
                "cache": "HIT",
                **security.redact_log_fields({
                    "question": question,
                    "answer": cached_hit["answer"],
                }),
                "similarity_score": cached_hit["similarity_score"],
                "latency_ms": latency_ms,
                # Not always zero -- contextualize_question() may have made
                # an LLM call before the cache was consulted.
                **cost.current().as_log_fields(),
            }))
            
            # Yield the entire cached answer as a single token for simplicity,
            # then yield the final payload.
            yield f"data: {json.dumps({'token': cached_hit['answer']})}\n\n"
            
            final_payload = {
                "type": "final",
                "question": question,
                "answer": cached_hit["answer"],
                "groundedness": cached_hit["groundedness"],
                "sources": [],
                "latency_ms": latency_ms,
                "cached": True
            }
            yield f"data: {json.dumps(final_payload)}\n\n"
            return
            
        # 3. Retrieve documents
        try:
            chunks = await asyncio.to_thread(
                retrieve, contextualized_q, top_k, doc_session_id
            )
        except Exception as exc:
            logger.error(
                json.dumps({"request_id": request_id, "event": "error", "endpoint": "ask-stream/retrieve", "error": str(exc)}),
                exc_info=True,
            )
            yield f"data: {json.dumps({'error': 'Retrieval failed', 'details': str(exc), 'request_id': request_id})}\n\n"
            return
            
        context = _format_context(chunks)
        
        # 4. Stream LLM Response
        llm = get_llm(temperature=0.0, stage="generate_stream")
        messages = [
            # Imported rather than duplicated. This was a hand-copied twin of
            # rag.py's prompt, so any rule added there -- like the new
            # context-boundary rule -- silently did not apply to /ask-stream.
            ("system", _ANSWER_SYSTEM_PROMPT),
            ("human", f"QUESTION:\n{contextualized_q}\n\nCONTEXT:\n{context}"),
        ]
        
        full_answer = []
        
        # Run the blocking stream in a thread so it doesn't block the async event loop.
        try:
            async for chunk in llm.astream(messages):
                if chunk.content:
                    full_answer.append(chunk.content)
                    yield f"data: {json.dumps({'token': chunk.content})}\n\n"
        except Exception as exc:
            logger.error(
                json.dumps({"request_id": request_id, "event": "error", "endpoint": "ask-stream/generate", "error": str(exc)}),
                exc_info=True,
            )
            yield f"data: {json.dumps({'error': 'Generation failed', 'details': str(exc), 'request_id': request_id})}\n\n"
            return
            
        final_answer = "".join(full_answer)

        # Output screening on a stream is necessarily after the fact: the
        # tokens have already reached the reader, so there is nothing to
        # suppress. What it can still do is refuse to CACHE a leaked answer
        # (below) and record it, so one successful injection does not get
        # replayed to every later visitor asking a similar question.
        leak = security.screen_answer(final_answer)
        if leak.flagged:
            metrics.record_prompt_leak()
            logger.warning(json.dumps({
                "request_id": request_id, "event": "prompt_leak_suppressed",
                "endpoint": "ask-stream", "reason": leak.reason,
            }))

        # 5. Check groundedness post-generation
        groundedness = await asyncio.to_thread(check_groundedness, final_answer, chunks)
        
        # 6. Build sources and cache
        sources = [
            {
                "source": c.metadata.get("source", "unknown"),
                "page": c.metadata.get("page"),
                "excerpt": c.page_content[:200],
            }
            for c in chunks
        ]
        
        # Two independent reasons to withhold from the shared cache: a leaked
        # system prompt, and an answer grounded in this visitor's private
        # upload (the cache is global and consulted before retrieval).
        used_private = any(c.metadata.get("_session_id") for c in chunks)
        if not leak.flagged and not used_private:
            await asyncio.to_thread(cache.set_cached_answer, contextualized_q, final_answer, groundedness)
        if session_id:
            await asyncio.to_thread(memory.add_to_history, session_id, question, final_answer)
            
        latency_ms = int((time.perf_counter() - start) * 1000)
        metrics.record_latency(latency_ms)
        metrics.record_groundedness(groundedness)
        if not chunks:
            metrics.record_empty_retrieval()
            
        logger.info(json.dumps({
            "request_id": request_id,
            "event": "ask-stream",
            **security.redact_log_fields({
                "question": question,
                "contextualized_query": contextualized_q,
                "answer": final_answer,
            }),
            "groundedness": groundedness,
            "num_sources": len(sources),
            "latency_ms": latency_ms,
            # astream() usage lands here via _CostTrackingLLM, which merges
            # streamed chunks so the final message carries the totals.
            **cost.current().as_log_fields(),
        }))
        
        final_payload = {
            "type": "final",
            "question": question,
            "answer": final_answer,
            "groundedness": groundedness,
            "sources": sources,
            "latency_ms": latency_ms,
            "cached": False,
            "request_id": request_id,
        }
        yield f"data: {json.dumps(final_payload)}\n\n"
        
    except Exception as exc:
        # Catch ANY exception (e.g., API Key invalid, network timeout) and securely 
        # log it, then yield a proper JSON SSE payload instead of silently 500-ing.
        metrics.record_error("ask-stream")
        logger.error(
            json.dumps({
                "request_id": request_id, 
                "event": "error", 
                "endpoint": "ask-stream", 
                "question": question, 
                "error": str(exc),
            }),
            exc_info=True,
        )
        yield f"data: {json.dumps({'error': 'Server Error', 'details': 'An unexpected error occurred. Check server logs.', 'request_id': request_id})}\n\n"
