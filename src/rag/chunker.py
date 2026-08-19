import json

from docling_core.transforms.chunker import HybridChunker
from docling_core.types.doc import DocItemLabel

from src.core.config import settings
from src.core.errors import EmptyDocumentError
from src.rag.document_parser import picture_image_filename


def _concrete_items_by_ref(doc):
    by_ref = {}
    for t in doc.tables:
        by_ref[t.self_ref] = ("table", t)
    for p in doc.pictures:
        by_ref[p.self_ref] = ("picture", p)
    return by_ref


def _picture_text(item, doc):
    parts = []
    caption = item.caption_text(doc)
    if caption:
        parts.append(f"Caption: {caption}")
    for ann in item.annotations:
        text = getattr(ann, "text", None)
        if not text:
            continue
        text = text.split("<end_of_utterance")[0].strip()
        if text:
            label = "OCR text" if getattr(ann, "provenance", "") == "tesseract-ocr" else "Description"
            parts.append(f"{label}: {text}")
    return "\n".join(parts) if parts else None


def _item_page(item):
    return item.prov[0].page_no if item.prov else None


_MAX_TABLE_TEXT_BYTES = 60_000


def _split_oversized_table_markdown(markdown, tokenizer):
    max_tokens = tokenizer.max_tokens
    if len(markdown.encode("utf-8")) <= _MAX_TABLE_TEXT_BYTES and tokenizer.count_tokens(markdown) <= max_tokens:
        return [markdown]

    lines = markdown.splitlines()
    if len(lines) < 3:
        encoded = markdown.encode("utf-8")[: _MAX_TABLE_TEXT_BYTES - 20]
        return [encoded.decode("utf-8", errors="ignore") + "\n\n[...truncated: row exceeded storage limit]"]

    header, separator, rows = lines[0], lines[1], lines[2:]
    header_bytes = len(header.encode("utf-8")) + len(separator.encode("utf-8")) + 2
    header_tokens = tokenizer.count_tokens(f"{header}\n{separator}")

    pieces = []
    current_rows, current_bytes, current_tokens = [], header_bytes, header_tokens
    for row in rows:
        row_bytes = len(row.encode("utf-8")) + 1
        row_tokens = tokenizer.count_tokens(row)
        if current_rows and (
            current_bytes + row_bytes > _MAX_TABLE_TEXT_BYTES or current_tokens + row_tokens > max_tokens
        ):
            pieces.append("\n".join([header, separator, *current_rows]))
            current_rows, current_bytes, current_tokens = [], header_bytes, header_tokens
        if header_bytes + row_bytes > _MAX_TABLE_TEXT_BYTES or header_tokens + row_tokens > max_tokens:
            while row and tokenizer.count_tokens(row) + header_tokens > max_tokens:
                row = row[: max(1, len(row) // 2)]
            row = row.encode("utf-8")[: _MAX_TABLE_TEXT_BYTES - header_bytes - 20].decode("utf-8", errors="ignore")
            row += "…"
            row_bytes = len(row.encode("utf-8")) + 1
            row_tokens = tokenizer.count_tokens(row)
        current_rows.append(row)
        current_bytes += row_bytes
        current_tokens += row_tokens
    if current_rows:
        pieces.append("\n".join([header, separator, *current_rows]))

    return pieces or [markdown]


_EMPTY_BBOX = {"left": 0.0, "top": 0.0, "width": 0.0, "height": 0.0}


def _chunk_bbox_fraction(doc_items, doc, page):
    if page is None or page not in doc.pages:
        return _EMPTY_BBOX
    boxes = [item.prov[0].bbox for item in doc_items if item.prov and item.prov[0].page_no == page]
    if not boxes:
        return _EMPTY_BBOX

    page_size = doc.pages[page].size
    if page_size.width <= 0 or page_size.height <= 0:
        return _EMPTY_BBOX

    top_left = [b.to_top_left_origin(page_size.height) for b in boxes]
    left, top = min(b.l for b in top_left), min(b.t for b in top_left)
    right, bottom = max(b.r for b in top_left), max(b.b for b in top_left)
    return {
        "left": max(0.0, left / page_size.width),
        "top": max(0.0, top / page_size.height),
        "width": max(0.0, (right - left) / page_size.width),
        "height": max(0.0, (bottom - top) / page_size.height),
    }


def _overlap_tail(text, fraction=None):
    words = text.split()
    if not words:
        return ""
    n = max(1, round(len(words) * (fraction if fraction is not None else settings.text_overlap_fraction)))
    return " ".join(words[-n:])


def chunk_document(doc, cache_path):
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            return cached["texts"], cached["metas"]
        except (json.JSONDecodeError, KeyError):
            cache_path.unlink(missing_ok=True)

    chunker = HybridChunker(
        tokenizer=settings.embed_model_name, max_tokens=settings.chunk_max_tokens, merge_peers=False
    )
    raw_chunks = list(chunker.chunk(doc))
    concrete_by_ref = _concrete_items_by_ref(doc)

    chunk_metas = []
    bodies = []

    for c in raw_chunks:
        doc_items = c.meta.doc_items or []
        headings = " > ".join(c.meta.headings) if c.meta.headings else ""
        page = _item_page(doc_items[0]) if doc_items else None

        resolved = [concrete_by_ref.get(i.self_ref) for i in doc_items]
        table_items = [item for kind, item in filter(None, resolved) if kind == "table"]
        picture_items = [item for kind, item in filter(None, resolved) if kind == "picture"]

        if table_items:
            table_texts = [
                piece
                for t in table_items
                for piece in _split_oversized_table_markdown(t.export_to_markdown(doc), chunker.tokenizer)
            ]
            content_type = "table"
            image_filename = ""

            bbox = _chunk_bbox_fraction(doc_items, doc, page)
            for text in table_texts:
                bodies.append(text)
                chunk_metas.append(
                    {
                        "chunk_index": len(bodies) - 1,
                        "page": page,
                        "headings": headings,
                        "content_type": content_type,
                        "has_table": True,
                        "has_picture": False,
                        "image_path": image_filename,
                        "bbox_left": bbox["left"],
                        "bbox_top": bbox["top"],
                        "bbox_width": bbox["width"],
                        "bbox_height": bbox["height"],
                    }
                )
            continue
        elif picture_items:
            picture_texts = [_picture_text(p, doc) for p in picture_items]
            picture_texts = [t for t in picture_texts if t]
            if not picture_texts:
                continue
            text = "\n\n".join(picture_texts)
            content_type = "picture"

            image_filename = picture_image_filename(picture_items[0])
        else:
            is_list = any(getattr(item, "label", None) == DocItemLabel.LIST_ITEM for item in doc_items)
            text = chunker.contextualize(c)
            content_type = "list" if is_list else "text"
            image_filename = ""

        bbox = _chunk_bbox_fraction(doc_items, doc, page)

        bodies.append(text)
        chunk_metas.append(
            {
                "chunk_index": len(bodies) - 1,
                "page": page,
                "headings": headings,
                "content_type": content_type,
                "has_table": content_type == "table",
                "has_picture": content_type == "picture",
                "image_path": image_filename,
                "bbox_left": bbox["left"],
                "bbox_top": bbox["top"],
                "bbox_width": bbox["width"],
                "bbox_height": bbox["height"],
            }
        )

    for i in range(1, len(bodies)):
        if chunk_metas[i]["content_type"] == "text" and chunk_metas[i - 1]["content_type"] == "text":
            tail = _overlap_tail(bodies[i - 1])
            if tail:
                bodies[i] = f"{tail} {bodies[i]}"

    chunk_texts = [
        f"{meta['headings']}\n\n{body}" if meta["headings"] else body
        for meta, body in zip(chunk_metas, bodies, strict=True)
    ]

    if not chunk_texts:
        raise EmptyDocumentError(
            "No searchable content found in this document (blank pages, or images "
            "with no caption/description/OCR text)."
        )

    cache_path.write_text(json.dumps({"texts": chunk_texts, "metas": chunk_metas}))
    return chunk_texts, chunk_metas


def to_langchain_documents(chunk_texts, chunk_metas):
    from langchain_core.documents import Document

    return [Document(page_content=chunk_texts[i], metadata=chunk_metas[i]) for i in range(len(chunk_texts))]
