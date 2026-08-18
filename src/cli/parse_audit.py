import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.config import cache_paths_for, settings
from src.core.errors import RagError
from src.rag.chunker import chunk_document
from src.rag.document_parser import document_metadata, parse_document


def audit_one(pdf_path):
    paths = cache_paths_for(pdf_path)
    start = time.monotonic()
    doc = parse_document(pdf_path, paths["document"])
    meta = document_metadata(doc)
    chunk_texts, _ = chunk_document(doc, paths["chunks"])
    elapsed = time.monotonic() - start
    return {
        "ok": True,
        "elapsed_s": round(elapsed, 1),
        "pages": meta["total_pages"],
        "tables": meta["total_tables"],
        "pictures": meta["total_pictures"],
        "chunks": len(chunk_texts),
    }


def main():
    pdfs = sorted(settings.data_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {settings.data_dir}")
        return

    results = []
    for pdf_path in pdfs:
        print(f"Parsing {pdf_path.name} ...", flush=True)
        try:
            result = audit_one(pdf_path)
            results.append((pdf_path.name, result))
            print(
                f"  OK  {result['elapsed_s']}s  pages={result['pages']} "
                f"tables={result['tables']} pictures={result['pictures']} chunks={result['chunks']}"
            )
        except RagError as exc:
            results.append((pdf_path.name, {"ok": False, "error": str(exc)}))
            print(f"  FAIL (known): {exc}")
        except Exception as exc:
            results.append((pdf_path.name, {"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
            print(f"  FAIL (unexpected): {type(exc).__name__}: {exc}")

    print("\n--- Summary ---")
    ok_count = sum(1 for _, r in results if r["ok"])
    print(f"{ok_count}/{len(results)} parsed cleanly")
    for name, r in results:
        status = "OK" if r["ok"] else f"FAIL - {r['error']}"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
