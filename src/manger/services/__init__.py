"""Services for the Manga Translation application."""

from manger.services.ocr import BaseOCRService, DummyOCRService
from manger.services.translator import BaseTranslator, DummyTranslator
from manger.services.renderer import Renderer
from manger.services.pdf import PDFService
from manger.services.archive import ArchiveService, images_to_cbz
from manger.services.epub import EPUBService, Chapter, text_to_epub

__all__ = [
    "BaseOCRService",
    "DummyOCRService",
    "BaseTranslator",
    "DummyTranslator",
    "Renderer",
    "PDFService",
    "ArchiveService",
    "images_to_cbz",
    "EPUBService",
    "Chapter",
    "text_to_epub",
]
