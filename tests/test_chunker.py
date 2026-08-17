import json

import pytest

from src.core.errors import EmptyDocumentError
from src.rag import chunker


class FakeAnnotation:
    def __init__(self, text, provenance=""):
        self.text = text
        self.provenance = provenance


class FakePicture:
    def __init__(self, caption, annotations):
        self._caption = caption
        self.annotations = annotations

    def caption_text(self, doc):
        return self._caption


def test_picture_text_combines_caption_description_and_ocr():
    annotations = [
        FakeAnnotation("A chart of quarterly revenue<end_of_utterance>", provenance="vlm"),
        FakeAnnotation("Q1 Q2 Q3 Q4", provenance="tesseract-ocr"),
    ]
    picture = FakePicture("Figure 1: Revenue", annotations)

    text = chunker._picture_text(picture, doc=None)

    assert "Caption: Figure 1: Revenue" in text
    assert "Description: A chart of quarterly revenue" in text
    assert "<end_of_utterance" not in text  # SmolVLM's stray stop token got stripped
    assert "OCR text: Q1 Q2 Q3 Q4" in text


def test_picture_text_returns_none_when_nothing_usable():
    picture = FakePicture(None, [])
    assert chunker._picture_text(picture, doc=None) is None


def test_picture_text_skips_empty_annotation_text():
    picture = FakePicture("Just a caption", [FakeAnnotation("")])
    text = chunker._picture_text(picture, doc=None)
    assert text == "Caption: Just a caption"


class _FakeMeta:
    def __init__(self, doc_items=None, headings=None):
        self.doc_items = doc_items or []
        self.headings = headings or []


class _FakeRawChunk:
    def __init__(self, meta):
        self.meta = meta


class _FakeHybridChunker:
    def __init__(self, **kwargs):
        pass

    def chunk(self, doc):
        return [_FakeRawChunk(_FakeMeta())]

    def contextualize(self, c):
        return "hello world"


class _FakeEmptyHybridChunker:
    def __init__(self, **kwargs):
        pass

    def chunk(self, doc):
        return []


class _FakeDoc:
    tables = []
    pictures = []


def test_chunk_document_raises_on_zero_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(chunker, "HybridChunker", _FakeEmptyHybridChunker)
    cache_path = tmp_path / "chunks.cache.json"

    with pytest.raises(EmptyDocumentError):
        chunker.chunk_document(_FakeDoc(), cache_path)

    assert not cache_path.exists()  # never cache an empty/failed result


def test_chunk_document_recovers_from_corrupted_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(chunker, "HybridChunker", _FakeHybridChunker)
    cache_path = tmp_path / "chunks.cache.json"
    cache_path.write_text("{not valid json at all")

    texts, metas = chunker.chunk_document(_FakeDoc(), cache_path)

    assert texts == ["hello world"]
    assert len(metas) == 1
    # cache file is valid JSON now, and matches what was returned
    reloaded = json.loads(cache_path.read_text())
    assert reloaded["texts"] == texts


def test_chunk_document_uses_valid_cache_without_rechunking(tmp_path, monkeypatch):
    cache_path = tmp_path / "chunks.cache.json"
    cache_path.write_text(json.dumps({"texts": ["cached chunk"], "metas": [{"page": 1}]}))

    def explode(**kwargs):
        raise AssertionError("should not rechunk when cache is valid")

    monkeypatch.setattr(chunker, "HybridChunker", explode)

    texts, metas = chunker.chunk_document(_FakeDoc(), cache_path)
    assert texts == ["cached chunk"]
    assert metas == [{"page": 1}]
