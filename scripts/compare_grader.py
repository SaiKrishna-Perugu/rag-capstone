"""
Compare the agentic grader's two implementations on the same retrieved context.

The LLM grader (agent._grade_with_llm) returns SUFFICIENT/INSUFFICIENT parsed
from free text; the TypeSafe grader returns a probability that
config.TYPESAFE_GRADER_THRESHOLD turns into the same verdict. This runs both
over questions the corpus answers and questions it does not, so the threshold
is chosen from data before TYPESAFE_GRADER is switched on anywhere.

    uv run python scripts/compare_grader.py            # 5 LLM runs per question
    uv run python scripts/compare_grader.py --runs 10

Needs what eval.py needs (a populated DATABASE_URL, live Vertex AI) plus
TYPESAFE_API_KEY. Makes real, paid calls: (runs + 1) per question.

What to read in the output: an answerable question graded INSUFFICIENT is a
false rejection -- it costs a rewrite and retry, or ends in the "not enough
information" fallback. An unanswerable question graded SUFFICIENT is worse:
it sends the generator to answer from context that does not contain the
answer. Pick the threshold with the fewest false rejections that still
grades every unanswerable question INSUFFICIENT.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Must be set before app.config is imported: the key is only fetched when a
# judgment is switched on.
os.environ.setdefault("TYPESAFE_GRADER", "true")

from app import config
from app.llm import typesafe
from app.retrieval import agent
from app.retrieval.rag import retrieve
from eval import EVAL_SET

# Topics checked against docs/ with grep: none of them appear anywhere in the
# curated corpus. Re-check if docs/ changes, or these labels become wrong.
UNANSWERABLE = [
    "What is the CEO's name?",
    "What was the company's revenue last year?",
    "How many employees does the company have?",
    "Is there a student discount?",
    "Do you ship to Canada?",
]

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _grade(question: str, runs: int) -> dict:
    chunks = retrieve(question)
    state = {"original_question": question, "chunks": chunks}
    llm = [agent._grade_with_llm(state) for _ in range(runs)] if chunks else []
    p = typesafe.noul(
        state={
            "question": question,
            "passages": [
                {"source": c.metadata.get("source", "unknown"), "text": c.page_content}
                for c in chunks
            ],
        },
        instructions=agent._SUFFICIENCY_INSTRUCTIONS,
        criteria=agent._SUFFICIENCY_CRITERIA,
        stage="grade",
    ) if chunks else 0.0
    return {
        "sources": sorted({c.metadata.get("source", "?").split("/")[-1].split("\\")[-1] for c in chunks}),
        "llm_sufficient": sum(v == "SUFFICIENT" for v in llm),
        "p": p,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5, help="LLM grader runs per question")
    args = parser.parse_args()

    if not config.TYPESAFE_API_KEY:
        print("TYPESAFE_API_KEY is not set; nothing to compare.")
        return 1

    cases = [(c["question"], True) for c in EVAL_SET
             if not c["expected_answer"].startswith("I don't have enough")]
    cases += [(q, False) for q in UNANSWERABLE]

    rows = []
    print(f"{'answerable':<11}{'LLM sufficient':<16}{'TypeSafe p':<12}question  [retrieved from]")
    for question, answerable in cases:
        r = _grade(question, args.runs)
        rows.append((answerable, r))
        p = "n/a" if r["p"] is None else f"{r['p']:.3f}"
        print(f"{'yes' if answerable else 'no':<11}{r['llm_sufficient']}/{args.runs:<14}{p:<12}"
              f"{question}  [{', '.join(r['sources'])}]")

    scored = [(a, r) for a, r in rows if r["p"] is not None]
    if len(scored) < len(rows):
        print(f"\n{len(rows) - len(scored)} TypeSafe call(s) fell back; check the key and logs.")

    llm_false_reject = sum(args.runs - r["llm_sufficient"] for a, r in rows if a)
    llm_false_accept = sum(r["llm_sufficient"] for a, r in rows if not a)
    n_ans = sum(a for a, _ in rows)
    print(f"\nLLM grader: {llm_false_reject}/{n_ans * args.runs} answerable runs rejected, "
          f"{llm_false_accept}/{(len(rows) - n_ans) * args.runs} unanswerable runs accepted")

    print(f"\n{'threshold':<11}{'answerable rejected':<22}unanswerable accepted")
    for t in THRESHOLDS:
        rej = sum(1 for a, r in scored if a and r["p"] < t)
        acc = sum(1 for a, r in scored if not a and r["p"] >= t)
        print(f"{t:<11}{rej}/{sum(a for a, _ in scored):<20}{acc}/{sum(not a for a, _ in scored)}")
    print(f"\nCurrent TYPESAFE_GRADER_THRESHOLD = {config.TYPESAFE_GRADER_THRESHOLD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
