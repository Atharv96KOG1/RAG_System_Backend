"""Manual retrieval quality check — not wired into CI, just a fast way to see whether a
retrieval change actually helped instead of guessing. Point it at a PDF and a small
question set with an expected page or keyword, and it reports hit-rate.

Usage:
    python scripts/eval_retrieval.py path/to/doc.pdf questions.json

questions.json shape:
    [
      {"question": "What is the termination notice period?", "expected_page": 12},
      {"question": "Who is the primary contact for support?", "expected_keyword": "support@"}
    ]

A question "hits" if the final answer cites the expected page (as "page N", the
citation format the prompt requires — see PROMPT_SYSTEM in rag_chain.py) or contains
the expected keyword (case-insensitive substring). This is an end-to-end check
(retrieval + LLM phrasing), which is a more robust dev-loop signal than trying to
introspect the LCEL chain's internal retriever step.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.pipeline import build_pipeline  # noqa: E402
from src.rag.rag_chain import answer_query  # noqa: E402


def _hits(answer, expected):
    if "expected_page" in expected:
        return f"page {expected['expected_page']}" in answer
    if "expected_keyword" in expected:
        return expected["expected_keyword"].lower() in answer.lower()
    raise ValueError(f"Question has no expected_page or expected_keyword: {expected}")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <pdf_path> <questions.json>")
        sys.exit(1)

    pdf_path, questions_path = sys.argv[1], sys.argv[2]
    questions = json.loads(Path(questions_path).read_text())

    print(f"Ingesting {pdf_path}...")
    result = build_pipeline(pdf_path)
    rag_chain, doc_metadata = result["rag_chain"], result["doc_metadata"]

    hits = 0
    for q in questions:
        answer = answer_query(q["question"], rag_chain, doc_metadata)
        hit = _hits(answer, q)
        hits += hit
        status = "HIT " if hit else "MISS"
        print(f"{status}  {q['question']}\n      -> {answer[:160].replace(chr(10), ' ')}")

    print(f"\n{hits}/{len(questions)} hit rate ({100 * hits / len(questions):.0f}%)")


if __name__ == "__main__":
    main()
