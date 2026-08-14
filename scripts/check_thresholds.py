"""
CI eval-gate: fails the build if any RAGAS metric from eval_ragas.py's
last run is below its floor.

Run:
    python scripts/check_thresholds.py ragas_results.json

Absolute floors (not a rolling-baseline/percentage-regression scheme) --
right-sized for an 8-question eval set: with n=8 and LLM-as-judge metrics
carrying real run-to-run variance, gating on "regressed vs. last run"
would flap on noise rather than catch real quality drops. Floors are set
a bit below a real local baseline run of eval_ragas.py, not guessed.

Real baseline run (Groq's free-tier daily quota was exhausted from other
testing, so this ran via Vertex AI for the LLM only -- FastEmbed
embeddings and the real Postgres corpus unchanged; see eval_ragas.py's
_RagasCompatibleEmbeddings for a real RAGAS+FastEmbed bug found and fixed
in the process). Against a clean, CI-representative database (only the
git-tracked docs/ files, matching what eval.yml's ephemeral Postgres
actually ingests): faithfulness 0.875, answer_relevancy 0.821,
llm_context_precision_without_reference 1.0, context_recall 1.0 -- 7 of 8
questions scored at or near perfect; the 8th is EVAL_SET's deliberate
negative test case ("What is the CEO's name?"), which correctly refuses
to answer and so correctly scores near-zero on faithfulness/relevancy by
design, pulling down the mean. Floors below sit meaningfully under that
baseline -- LLM-as-judge metrics carry real run-to-run variance, and
precision/recall hit a perfect 1.0 with zero headroom otherwise -- while
still being real bars a genuine regression (e.g. dropping the reranker)
would breach.
"""
import json
import math
import sys
from pathlib import Path

THRESHOLDS = {
    "faithfulness": 0.6,
    "answer_relevancy": 0.55,
    "llm_context_precision_without_reference": 0.7,
    "context_recall": 0.7,
}


def check(results_path: Path) -> list[str]:
    data = json.loads(results_path.read_text())
    mean_scores = data["mean_scores"]

    failures = []
    for metric, floor in THRESHOLDS.items():
        if metric not in mean_scores:
            failures.append(f"{metric}: missing from results (expected >= {floor})")
            continue
        score = mean_scores[metric]
        # A metric that's entirely NaN (every sample for it failed) must
        # not silently pass -- NaN comparisons are always False in Python,
        # so `score < floor` alone would never catch this.
        if math.isnan(score) or score < floor:
            failures.append(f"{metric}: {score} (floor: {floor})")
    return failures


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/check_thresholds.py <results.json>", file=sys.stderr)
        sys.exit(1)

    results_path = Path(sys.argv[1])
    if not results_path.exists():
        print(f"check_thresholds: {results_path} not found -- did eval_ragas.py run first?", file=sys.stderr)
        sys.exit(1)

    failures = check(results_path)

    if failures:
        print("Eval gate FAILED -- metrics below threshold:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)

    print("Eval gate passed -- all metrics at or above threshold.")
