import pickle

import pytesseract
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem
from docling_core.types.doc.document import DescriptionAnnotation

from src.core.errors import DocumentParseError


def parse_document(source_path, cache_path):
    if cache_path.exists():
        try:
            doc = pickle.loads(cache_path.read_bytes())
            return doc
        except Exception:
            cache_path.unlink(missing_ok=True)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = TesseractCliOcrOptions(force_full_page_ocr=False)
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    pipeline_options.generate_picture_images = True  # keep raster crops so we can OCR/caption each picture directly
    pipeline_options.images_scale = 2.0  # higher-res crops = better OCR accuracy on small in-image text
    pipeline_options.do_picture_description = True  # local VLM (SmolVLM) captions non-text figures/diagrams

    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})

    try:
        result = converter.convert(str(source_path))
    except Exception as exc:
        raise DocumentParseError(
            f"Could not read '{source_path}'. It may be password-protected, corrupted, or not a valid PDF."
        ) from exc

    doc = result.document
    if not doc.pages:
        raise DocumentParseError(f"'{source_path}' has no pages — nothing to index.")

    _ocr_pictures(doc)
    cache_path.write_bytes(pickle.dumps(doc))
    return doc


def _ocr_pictures(doc):
    """Docling's page-level OCR skips text baked into embedded picture bitmaps
    (charts/screenshots) — those items only get a VLM caption, which loses any
    literal text (e.g. labels, numbers). Run Tesseract on each picture's own
    crop and attach the result as a DescriptionAnnotation so chunker.chunk_document
    (via caption_text/annotations) can surface the literal text, not just a caption."""
    for item, _level in doc.iterate_items():
        if not isinstance(item, PictureItem):
            continue
        try:
            image = item.get_image(doc)
        except Exception:
            continue
        if image is None:
            continue
        try:
            text = pytesseract.image_to_string(image).strip()
        except pytesseract.TesseractNotFoundError:
            return
        except Exception:
            continue
        if text:
            item.annotations.append(DescriptionAnnotation(text=text, provenance="tesseract-ocr"))


def document_metadata(doc):
    data = doc.export_to_dict()
    return {
        "total_pages": len(doc.pages),
        "total_tables": len(data.get("tables", [])),
        "total_pictures": len(data.get("pictures", [])),
        "total_text_blocks": len(data.get("texts", [])),
    }
