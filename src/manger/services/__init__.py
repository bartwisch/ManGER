"""Services for the Manga Translation application."""

from manger.services.ocr import BaseOCRService, DummyOCRService
from manger.services.translator import BaseTranslator, DummyTranslator
from manger.services.renderer import Renderer
from manger.services.pdf import PDFService

__all__ = [
    "BaseOCRService",
    "DummyOCRService",
    "BaseTranslator",
    "DummyTranslator",
    "Renderer",
    "PDFService",
]