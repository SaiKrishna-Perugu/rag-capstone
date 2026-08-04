"""
Conversation Memory for RAG.

Handles tracking chat history by session_id and contextualizing follow-up 
questions (e.g. rewriting "How much does it cost?" to "How much does the 
Pro plan cost?" based on previous turns) before they hit the vector store.
"""

import threading
from collections import defaultdict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.providers import get_llm

# In-memory store: session_id -> list of (user_message, ai_message)
_histories: dict[str, list[tuple[str, str]]] = defaultdict(list)
_lock = threading.Lock()

_CONTEXTUALIZE_PROMPT = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

def add_to_history(session_id: str, question: str, answer: str) -> None:
    """Save a Q&A turn to the session's history."""
    if not session_id:
        return
        
    with _lock:
        _histories[session_id].append((question, answer))
        # Keep only the last 5 turns to prevent prompt bloat
        if len(_histories[session_id]) > 5:
            _histories[session_id] = _histories[session_id][-5:]

def contextualize_question(session_id: str, question: str) -> str:
    """
    If there is chat history for the session, rewrite the question to be 
    standalone. Otherwise, return the original question.
    """
    if not session_id:
        return question
        
    with _lock:
        history = _histories.get(session_id, [])
        
    if not history:
        return question

    messages = [SystemMessage(content=_CONTEXTUALIZE_PROMPT)]
    
    # Add history
    for q, a in history:
        messages.append(HumanMessage(content=q))
        messages.append(AIMessage(content=a))
        
    # Add the current question
    messages.append(HumanMessage(content=question))
    
    llm = get_llm(temperature=0.0)
    response = llm.invoke(messages)
    
    # The LLM returns the rewritten standalone question
    return response.content.strip()
