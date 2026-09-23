"""
Agentic RAG graph built with LangGraph.

Replaces the single-pass retrieve-then-generate flow in rag.py with a
self-correcting loop:

    retrieve -> grade relevance -> [sufficient?] -> generate -> check groundedness -> END
                                        |
                                        v (not sufficient, retries remain)
                                    rewrite query -> retrieve (loop)
                                        |
                                        v (retries exhausted)
                                    fallback message -> END

Why a graph instead of an if/else chain: LangGraph gives us explicit,
inspectable state at every node, which is exactly what the @traceable
decorators below expose to LangSmith -- each node shows up as a named,
timed step (with the actual LLM calls nested inside), so a full run's
retrieve/grade/rewrite/generate path is visually inspectable, not just
logged as one opaque call. It's also the right foundation to extend for
more complex multi-agent orchestration later, rather than a throwaway
implementation.
"""
import json
import logging
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from app import config
from app.llm import typesafe
from app.llm.providers import get_llm
from app.retrieval.rag import (
    _format_context,
    build_sources,
    check_groundedness,
    generate_answer,
    retrieve,
)
from app.tracing import traced

logger = logging.getLogger("rag_service")

MAX_RETRIES = 2  # total retries after the first attempt (3 attempts overall)

FALLBACK_MESSAGE = (
    "I don't have enough information in the provided documents to answer "
    "that, even after rewriting the query. You may want to rephrase the "
    "question or check whether the relevant document has been ingested."
)

_GRADE_SYSTEM_PROMPT = """You are grading whether retrieved CONTEXT is \
sufficient to answer a QUESTION. Be strict: partial or tangentially related \
context should be graded as NOT sufficient.

Respond with exactly one word: "SUFFICIENT" or "INSUFFICIENT"."""

# The same judgment asked of TypeSafe (config.TYPESAFE_GRADER). Its answer is
# a probability, so the strictness the prompt above sets by wording ("Be
# strict") becomes TYPESAFE_GRADER_THRESHOLD instead, tunable against data.
_SUFFICIENCY_INSTRUCTIONS = (
    "Do the passages in `passages` contain the information needed to answer "
    "`question` directly, without guessing or relying on outside knowledge?"
)
_SUFFICIENCY_CRITERIA = {
    "true": "At least one passage states what the answer needs; together "
            "they are enough to answer the whole question.",
    "false": "The passages are about a related topic but do not state the "
             "answer, or cover only part of what the question asks.",
}

_REWRITE_SYSTEM_PROMPT = """You rewrite search queries to improve retrieval \
from a vector database. The previous query did not retrieve sufficient \
context. Rewrite it to be more specific, use different phrasing/synonyms, \
or decompose it -- whatever is most likely to surface better matches. \
Return ONLY the rewritten query, nothing else."""


class AgentState(TypedDict):
    original_question: str
    current_query: str
    # Document-visibility scope, not the conversation-memory session.
    session_id: str | None
    chunks: list
    grade: str            # "SUFFICIENT" | "INSUFFICIENT" | ""
    # TypeSafe's probability that the context is sufficient; None when the
    # LLM grader decided (switch off, or TypeSafe unavailable). Kept in state
    # so the value shows up in the trace next to the grade it produced.
    sufficiency_p: float | None
    retry_count: int
    answer: str
    groundedness: str
    sources: list


def _get_grading_llm(stage: str):
    """`stage` labels these calls in the per-request cost breakdown. The
    agentic loop's whole cost story is how much the grade/rewrite retries
    add on top of a plain /ask, and an unlabelled client puts both into one
    generic "llm" bucket where that is exactly what you cannot see."""
    return get_llm(temperature=0.0, stage=stage)


# --- Graph nodes ---------------------------------------------------------

@traced("agent.retrieve", run_type="retriever")
def node_retrieve(state: AgentState) -> AgentState:
    chunks = retrieve(state["current_query"], session_id=state.get("session_id"))
    return {**state, "chunks": chunks}


@traced("agent.grade_relevance", run_type="chain")
def node_grade(state: AgentState) -> AgentState:
    if not state["chunks"]:
        return {**state, "grade": "INSUFFICIENT"}

    if config.TYPESAFE_GRADER:
        p = typesafe.noul(
            state={
                "question": state["original_question"],
                "passages": [
                    {"source": c.metadata.get("source", "unknown"), "text": c.page_content}
                    for c in state["chunks"]
                ],
            },
            instructions=_SUFFICIENCY_INSTRUCTIONS,
            criteria=_SUFFICIENCY_CRITERIA,
            stage="grade",
        )
        if p is not None:
            grade = "SUFFICIENT" if p >= config.TYPESAFE_GRADER_THRESHOLD else "INSUFFICIENT"
            # One line per grade, attempt included: the distribution of p on
            # real questions is what TYPESAFE_GRADER_THRESHOLD is tuned from,
            # and the request log only carries the final outcome.
            logger.info(json.dumps({
                "event": "agent_grade", "sufficiency_p": round(p, 4),
                "grade": grade, "attempt": state.get("retry_count", 0),
                "threshold": config.TYPESAFE_GRADER_THRESHOLD,
            }))
            return {**state, "grade": grade, "sufficiency_p": p}

    return {**state, "grade": _grade_with_llm(state), "sufficiency_p": None}


def _grade_with_llm(state: AgentState) -> str:
    """The original grader, and the fallback whenever TypeSafe is not used.

    Any reply other than exactly one of the two words -- a trailing period,
    a sentence -- counts as INSUFFICIENT and costs a retry.
    """
    context = _format_context(state["chunks"])
    llm = _get_grading_llm("grade")
    messages = [
        ("system", _GRADE_SYSTEM_PROMPT),
        ("human", f"QUESTION:\n{state['original_question']}\n\nCONTEXT:\n{context}"),
    ]
    verdict = llm.invoke(messages).content.strip().upper()
    return verdict if verdict in ("SUFFICIENT", "INSUFFICIENT") else "INSUFFICIENT"


@traced("agent.rewrite_query", run_type="chain")
def node_rewrite_query(state: AgentState) -> AgentState:
    llm = _get_grading_llm("rewrite")
    messages = [
        ("system", _REWRITE_SYSTEM_PROMPT),
        ("human", (f"ORIGINAL QUESTION:\n{state['original_question']}\n\n"
                   f"PREVIOUS QUERY:\n{state['current_query']}")),
    ]
    new_query = llm.invoke(messages).content.strip()
    return {
        **state,
        "current_query": new_query,
        "retry_count": state["retry_count"] + 1,
    }


@traced("agent.generate", run_type="chain")
def node_generate(state: AgentState) -> AgentState:
    answer = generate_answer(state["original_question"], state["chunks"])
    groundedness = check_groundedness(answer, state["chunks"])
    sources = build_sources(state["chunks"])
    return {**state, "answer": answer, "groundedness": groundedness, "sources": sources}


@traced("agent.fallback", run_type="chain")
def node_fallback(state: AgentState) -> AgentState:
    return {
        **state,
        "answer": FALLBACK_MESSAGE,
        "groundedness": "GROUNDED",  # no claims made, trivially true
        "sources": [],
    }


# --- Routing ---------------------------------------------------------------

def route_after_grade(state: AgentState) -> Literal["generate", "rewrite", "fallback"]:
    if state["grade"] == "SUFFICIENT":
        return "generate"
    if state["retry_count"] < MAX_RETRIES:
        return "rewrite"
    return "fallback"


# --- Graph assembly ---------------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", node_retrieve)
    graph.add_node("grade", node_grade)
    graph.add_node("rewrite", node_rewrite_query)
    graph.add_node("generate", node_generate)
    graph.add_node("fallback", node_fallback)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {"generate": "generate", "rewrite": "rewrite", "fallback": "fallback"},
    )
    graph.add_edge("rewrite", "retrieve")  # loop back
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)

    return graph.compile()


_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


@traced("agentic_rag.run", run_type="chain")
def run_agentic_rag(question: str, session_id: str | None = None) -> AgentState:
    """Entry point used by main.py and eval.py."""
    initial_state: AgentState = {
        "original_question": question,
        "current_query": question,
        "session_id": session_id,
        "chunks": [],
        "grade": "",
        "sufficiency_p": None,
        "retry_count": 0,
        "answer": "",
        "groundedness": "NOT_CHECKED",
        "sources": [],
    }
    graph = get_compiled_graph()
    return graph.invoke(initial_state)
