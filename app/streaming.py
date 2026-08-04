"""
Streaming generator for SSE (Server-Sent Events) response.
"""

import asyncio
import json
import logging
import time
import uuid

from app import cache, memory, metrics
from app.providers import get_llm
from app.rag import _format_context, check_groundedness, retrieve

logger = logging.getLogger("rag_service")

async def stream_answer(question: str, session_id: str | None = None, top_k: int | None = None):
    """
    Async generator that yields SSE strings containing JSON payloads.
    It retrieves context, streams tokens as they are generated, and
    sends a final payload with groundedness and sources.
    """
    request_id = str(uuid.uuid4())
    metrics.record_request("ask-stream")
    start = time.perf_counter()
    
    try:
        # 1. Contextualize the question based on chat history
        contextualized_q = await asyncio.to_thread(
            memory.contextualize_question, session_id, question
        )
        
        # 2. Check semantic cache
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
                "question": question,
                "answer": cached_hit["answer"],
                "similarity_score": cached_hit["similarity_score"],
                "latency_ms": latency_ms,
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
            chunks = await asyncio.to_thread(retrieve, contextualized_q, top_k)
        except Exception as exc:
            yield f"data: {json.dumps({'error': 'Retrieval failed', 'details': str(exc)})}\n\n"
            return
            
        context = _format_context(chunks)
        
        # 4. Stream LLM Response
        llm = get_llm(temperature=0.0)
        messages = [
            ("system", "You are a precise assistant that answers questions using ONLY the provided context. Follow these rules strictly:\n1. Base your answer only on the context below. Do not use outside knowledge.\n2. If the context does not contain enough information to answer, say exactly: \"I don't have enough information in the provided documents to answer that.\"\n3. Keep answers concise and factual.\n4. Do not fabricate sources, numbers, or details not present in the context."),
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
            yield f"data: {json.dumps({'error': 'Generation failed', 'details': str(exc)})}\n\n"
            return
            
        final_answer = "".join(full_answer)
        
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
            "question": question,
            "contextualized_query": contextualized_q,
            "answer": final_answer,
            "groundedness": groundedness,
            "num_sources": len(sources),
            "latency_ms": latency_ms,
        }))
        
        final_payload = {
            "type": "final",
            "question": question,
            "answer": final_answer,
            "groundedness": groundedness,
            "sources": sources,
            "latency_ms": latency_ms,
            "cached": False
        }
        yield f"data: {json.dumps(final_payload)}\n\n"
        
    except Exception as exc:
        # Catch ANY exception (e.g., API Key invalid, network timeout) and securely 
        # log it, then yield a proper JSON SSE payload instead of silently 500-ing.
        metrics.record_error("ask-stream")
        logger.error(json.dumps({
            "request_id": request_id, 
            "event": "error", 
            "endpoint": "ask-stream", 
            "question": question, 
            "error": str(exc)
        }))
        yield f"data: {json.dumps({'error': 'Server Error', 'details': str(exc)})}\n\n"
