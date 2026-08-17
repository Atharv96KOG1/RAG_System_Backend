import json

from docling_core.transforms.chunker import HybridChunker
from docling_core.types.doc import DocItemLabel

from src.core.config import settings
from src.core.errors import EmptyDocumentError


def _concrete_items_by_ref(doc):
    """HybridChunker's chunk.meta.doc_items are generic DocItem stubs — isinstance
    against TableItem/PictureItem always fails. The concrete subclass instances
    (with export_to_markdown/get_image/caption_text) live in doc.tables/doc.pictures,
    addressable by self_ref (e.g. "#/tables/0")."""
    by_ref = {}
    for t in doc.tables:
        by_ref[t.self_ref] = ("table", t)
    for p in doc.pictures:
        by_ref[p.self_ref] = ("picture", p)
    return by_ref


def _picture_text(item, doc):
    """caption_text is the PDF's own figure caption (e.g. "Figure 3: ...").
    annotations carry the VLM's visual description plus our Tesseract OCR pass
    (document_parser._ocr_pictures) — both wanted, neither should be dropped."""
    parts = []
    caption = item.caption_text(doc)
    if caption:
        parts.append(f"Caption: {caption}")
    for ann in item.annotations:
        text = getattr(ann, "text", None)
        if not text:
            continue
        text = text.split("<end_of_utterance")[0].strip()  # SmolVLM leaks its stop token into the output
        if text:
            label = "OCR text" if getattr(ann, "provenance", "") == "tesseract-ocr" else "Description"
            parts.append(f"{label}: {text}")
    return "\n".join(parts) if parts else None


def _item_page(item):
    return item.prov[0].page_no if item.prov else None


def _overlap_tail(text, fraction=None):
    """Last ~15% (settings.text_overlap_fraction) of a text chunk's words, used to
    prepend trailing context onto the next chunk so information split across a chunk
    boundary isn't lost to either side."""
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
            # Truncated/corrupted cache file (e.g. process killed mid-write) —
            # reparse instead of failing forever on every future upload.
            cache_path.unlink(missing_ok=True)

    chunker = HybridChunker(
        tokenizer=settings.embed_model_name, max_tokens=settings.chunk_max_tokens, merge_peers=False
    )
    raw_chunks = list(chunker.chunk(doc))
    concrete_by_ref = _concrete_items_by_ref(doc)

    chunk_metas = []
    bodies = []  # raw serialized text, headings NOT yet prepended — overlap is computed
    # against this raw body so a chunk's overlap-tail never drags in the *previous*
    # chunk's heading line, only its actual content.
    for c in raw_chunks:
        doc_items = c.meta.doc_items or []
        headings = " > ".join(c.meta.headings) if c.meta.headings else ""
        page = _item_page(doc_items[0]) if doc_items else None

        # merge_peers=False keeps each chunk to a single doc_items group, so a chunk
        # is either a table, a picture, or prose — never a blend. Tables/pictures get
        # their own clean serialization instead of the chunker's generic flattened text,
        # which is what was garbling table cells and dropping picture text before.
        resolved = [concrete_by_ref.get(i.self_ref) for i in doc_items]
        table_items = [item for kind, item in filter(None, resolved) if kind == "table"]
        picture_items = [item for kind, item in filter(None, resolved) if kind == "picture"]

        if table_items:
            text = "\n\n".join(t.export_to_markdown(doc) for t in table_items)
            content_type = "table"
        elif picture_items:
            picture_texts = [_picture_text(p, doc) for p in picture_items]
            picture_texts = [t for t in picture_texts if t]
            if not picture_texts:
                continue  # picture with no caption, no VLM description, no OCR text — nothing to embed
            text = "\n\n".join(picture_texts)
            content_type = "picture"
        else:
            # DocItem stubs already carry a populated `label` field directly — no need
            # for the table/picture self_ref indirection above to detect list content.
            is_list = any(getattr(item, "label", None) == DocItemLabel.LIST_ITEM for item in doc_items)
            text = chunker.contextualize(c)
            content_type = "list" if is_list else "text"

        bodies.append(text)
        chunk_metas.append(
            {
                "chunk_index": len(bodies) - 1,  # position in this doc's own chunk list — graph_builder.py
                # namespaces its chunk nodes as f"chunk:{source_file}:{chunk_index}" using this same value,
                # so retrieval fusion (retriever.py) can map a retrieved Document back to its graph node.
                "page": page,
                "headings": headings,
                "content_type": content_type,
                "has_table": content_type == "table",
                "has_picture": content_type == "picture",
            }
        )

    # Sliding-window overlap: only between consecutive plain-text chunks. Never into/out
    # of table, picture, or list chunks — overlap would corrupt a table's cell grid or a
    # list's bullet structure, and a table/picture/list chunk's own content already
    # stands alone (it's a discrete unit, not prose that got arbitrarily cut).
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
        # A PDF that's all blank pages, or all pictures with no caption/description/
        # OCR text, parses fine but chunks to nothing — BM25Retriever.from_documents([])
        # and Milvus.from_documents([]) both fail on an empty list, so catch it here
        # with a clear message instead of a confusing crash two steps downstream.
        raise EmptyDocumentError(
            "No searchable content found in this document (blank pages, or images "
            "with no caption/description/OCR text)."
        )

    cache_path.write_text(json.dumps({"texts": chunk_texts, "metas": chunk_metas}))
    return chunk_texts, chunk_metas


def to_langchain_documents(chunk_texts, chunk_metas):
    from langchain_core.documents import Document

    return [Document(page_content=chunk_texts[i], metadata=chunk_metas[i]) for i in range(len(chunk_texts))]
