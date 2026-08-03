"""
Evaluation harness for the RAG pipeline.

Runs a fixed set of test questions with known-good expected answers through
the RAG pipeline, then uses an LLM-as-judge to score each answer against the
expected answer (this is a standard, real technique -- not a workaround).

Run:
    python eval.py

Edit EVAL_SET below to match whatever documents you actually ingested --
the questions/expected answers here are placeholders and MUST be replaced
with ones grounded in your own docs/ files.
"""
import json
import sys
from pathlib import Path

from app import config
from app.providers import get_llm
from app.rag import answer_question

# ---------------------------------------------------------------------------
# Replace these with 8-10 real Q&A pairs based on the documents you put in
# docs/. Keep expected_answer short -- a sentence or a fact, not a paragraph.
# ---------------------------------------------------------------------------
EVAL_SET = [
    {
        "question": "How long is the refund window?",
        "expected_answer": "Customers may request a full refund within 30 days of purchase.",
    },
    {
        "question": "Are digital products refundable?",
        "expected_answer": "No, digital products are non-refundable once downloaded.",
    },
    {
        "question": "How long does standard shipping take?",
        "expected_answer": "Standard shipping takes 5-8 business days within the country of purchase.",
    },
    {
        "question": "What does the hardware warranty cover?",
        "expected_answer": "The 1-year limited warranty covers manufacturing defects but does not cover accidental damage, water damage, or unauthorized repairs.",
    },
    {
        "question": "How do I reset my password?",
        "expected_answer": "Go to Settings > Security > Reset Password. A reset link will be emailed to your registered address, valid for 30 minutes.",
    },
    {
        "question": "What payment methods are accepted?",
        "expected_answer": "Visa, Mastercard, American Express, and PayPal. Bank transfers are supported for enterprise accounts only.",
    },
    {
        "question": "Is there a mobile app?",
        "expected_answer": "Yes, available on iOS and Android.",
    },
    {
        "question": "When is the scheduled maintenance window?",
        "expected_answer": "The first Sunday of every month, from 2:00 AM to 4:00 AM UTC.",
    },
    {
        "question": "What is the price of the Wireless Mouse?",
        "expected_answer": "The Wireless Mouse costs $24.99.",
    },
    # Negative test case — question NOT covered by any document.
    # Tests that the system refuses to hallucinate instead of guessing.
    {
        "question": "What is the CEO's name?",
        "expected_answer": "I don't have enough information in the provided documents to answer that.",
    },
]

_JUDGE_SYSTEM_PROMPT = """You are grading a QA system. You will be given a \
QUESTION, the EXPECTED_ANSWER, and the ACTUAL_ANSWER produced by the system.

Judge whether ACTUAL_ANSWER is factually consistent with EXPECTED_ANSWER \
(minor wording differences are fine; the facts must match).

Respond with exactly one word: "PASS" or "FAIL"."""


def judge(question: str, expected: str, actual: str) -> str:
    llm = get_llm(temperature=0.0)
    messages = [
        ("system", _JUDGE_SYSTEM_PROMPT),
        ("human", f"QUESTION:\n{question}\n\nEXPECTED_ANSWER:\n{expected}\n\nACTUAL_ANSWER:\n{actual}"),
    ]
    verdict = llm.invoke(messages).content.strip().upper()
    return verdict if verdict in ("PASS", "FAIL") else "UNKNOWN"


def run_eval() -> dict:
    results = []
    for case in EVAL_SET:
        rag_result = answer_question(case["question"], check_hallucination=True)
        verdict = judge(case["question"], case["expected_answer"], rag_result.answer)

        results.append({
            "question": case["question"],
            "expected_answer": case["expected_answer"],
            "actual_answer": rag_result.answer,
            "groundedness": rag_result.groundedness,
            "correctness": verdict,
        })

    passed = sum(1 for r in results if r["correctness"] == "PASS")
    grounded = sum(1 for r in results if r["groundedness"] == "GROUNDED")
    total = len(results)

    summary = {
        "total": total,
        "correctness_pass_rate": round(passed / total, 2) if total else 0,
        "groundedness_rate": round(grounded / total, 2) if total else 0,
        "results": results,
    }
    return summary


if __name__ == "__main__":
    if any(case["question"].startswith("REPLACE ME") for case in EVAL_SET):
        print(
            "eval.py: EVAL_SET still contains placeholder questions.\n"
            "Edit EVAL_SET with real questions/answers based on your docs/ "
            "files before running.",
            file=sys.stderr,
        )
        sys.exit(1)

    summary = run_eval()

    print(f"\n=== Eval Summary ===")
    print(f"Correctness pass rate:  {summary['correctness_pass_rate'] * 100:.0f}% "
          f"({sum(1 for r in summary['results'] if r['correctness'] == 'PASS')}/{summary['total']})")
    print(f"Groundedness rate:      {summary['groundedness_rate'] * 100:.0f}%")
    print()

    for r in summary["results"]:
        status = "PASS" if r["correctness"] == "PASS" else "FAIL"
        print(f"[{status}] [{r['groundedness']}] {r['question']}")

    out_path = Path("eval_results.json")
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nFull results written to {out_path}")
