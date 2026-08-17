import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.config import settings
from src.core.errors import RagError
from src.rag.pipeline import build_pipeline
from src.rag.rag_chain import answer_query


def main():
    source_path = sys.argv[1] if len(sys.argv) > 1 else settings.source_path
    try:
        result = build_pipeline(source_path)
    except RagError as exc:
        print(f"Could not process '{source_path}': {exc}")
        sys.exit(1)

    rag_chain, doc_metadata = result["rag_chain"], result["doc_metadata"]
    print(f"Pipeline ready. Doc metadata: {doc_metadata}")

    while True:
        try:
            question = input("\nAsk a question (or 'exit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if question.lower() in {"exit", "quit"}:
            break
        print(answer_query(question, rag_chain, doc_metadata))


if __name__ == "__main__":
    main()
