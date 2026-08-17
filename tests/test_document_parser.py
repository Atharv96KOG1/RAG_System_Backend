import pickle

import pytest

from src.core.errors import DocumentParseError
from src.rag import document_parser


class FakePage:
    pass


class FakeDoc:
    """Module-level so pickle can locate the class by import path."""

    def __init__(self, pages):
        self.pages = pages

    def export_to_dict(self):
        return {"tables": [{}, {}], "pictures": [{}], "texts": [{}, {}, {}, {}]}

    def iterate_items(self):
        return iter([])


class FakeConverter:
    def __init__(self, *args, **kwargs):
        pass

    def convert(self, path):
        raise RuntimeError("file has not been decrypted")


class FakeConverterSuccess:
    def __init__(self, *args, **kwargs):
        pass

    def convert(self, path):
        class Result:
            document = FakeDoc(pages=[FakePage()])

        return Result()


class FakeConverterEmpty:
    def __init__(self, *args, **kwargs):
        pass

    def convert(self, path):
        class Result:
            document = FakeDoc(pages=[])

        return Result()


def test_parse_document_uses_cache_without_reconverting(tmp_path, monkeypatch):
    doc = FakeDoc(pages=[FakePage()])
    cache_path = tmp_path / "document.cache.pkl"
    cache_path.write_bytes(pickle.dumps(doc))

    def explode(*a, **kw):
        raise AssertionError("should not reconvert when cache is valid")

    monkeypatch.setattr(document_parser, "DocumentConverter", explode)

    result = document_parser.parse_document(str(tmp_path / "whatever.pdf"), cache_path)
    assert isinstance(result, FakeDoc)
    assert len(result.pages) == 1


def test_parse_document_recovers_from_corrupted_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / "document.cache.pkl"
    cache_path.write_bytes(b"not a valid pickle stream")

    monkeypatch.setattr(document_parser, "DocumentConverter", FakeConverterSuccess)
    monkeypatch.setattr(document_parser, "_ocr_pictures", lambda doc: None)

    source = tmp_path / "file.pdf"
    source.write_bytes(b"%PDF-1.4 fake")

    result = document_parser.parse_document(str(source), cache_path)
    assert isinstance(result, FakeDoc)
    # cache got overwritten with a valid pickle this time
    assert isinstance(pickle.loads(cache_path.read_bytes()), FakeDoc)


def test_parse_document_wraps_encrypted_or_corrupt_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr(document_parser, "DocumentConverter", FakeConverter)

    source = tmp_path / "encrypted.pdf"
    source.write_bytes(b"%PDF-1.4 fake")
    cache_path = tmp_path / "document.cache.pkl"

    with pytest.raises(DocumentParseError):
        document_parser.parse_document(str(source), cache_path)


def test_parse_document_rejects_zero_page_document(tmp_path, monkeypatch):
    monkeypatch.setattr(document_parser, "DocumentConverter", FakeConverterEmpty)

    source = tmp_path / "blank.pdf"
    source.write_bytes(b"%PDF-1.4 fake")
    cache_path = tmp_path / "document.cache.pkl"

    with pytest.raises(DocumentParseError):
        document_parser.parse_document(str(source), cache_path)


def test_document_metadata_counts_from_export_dict():
    meta = document_parser.document_metadata(FakeDoc(pages=[FakePage(), FakePage(), FakePage()]))
    assert meta == {
        "total_pages": 3,
        "total_tables": 2,
        "total_pictures": 1,
        "total_text_blocks": 4,
    }


def test_ocr_pictures_missing_tesseract_binary_degrades_gracefully(monkeypatch):
    import pytesseract

    class FakePicture:
        annotations = []

        def get_image(self, doc):
            return object()

    class FakeDocWithPicture:
        def iterate_items(self):
            return iter([(FakePicture(), 0)])

    monkeypatch.setattr(document_parser, "PictureItem", FakePicture)

    def raise_not_found(image):
        raise pytesseract.TesseractNotFoundError()

    monkeypatch.setattr(pytesseract, "image_to_string", raise_not_found)

    # Should not raise — missing OCR binary degrades to caption/VLM-only.
    document_parser._ocr_pictures(FakeDocWithPicture())
