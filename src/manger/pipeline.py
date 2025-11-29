"""Main Pipeline for manga translation.

Orchestrates the flow:
1. Load Image
2. OCR (detect text blocks)
3. Filter (confidence check)
4. Group (merge close blocks)
5. Translate (batch translation)
6. Render (create final image)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from PIL import Image

from manger.config import AppConfig, get_config
from manger.domain.models import MangaPage, ProcessingResult, TextBlock
from manger.services.ocr import BaseOCRService, create_ocr_service, OCRError
from manger.services.translator import (
    BaseTranslator,
    DummyTranslator,
    TranslationError,
    create_translator,
)
from manger.services.renderer import Renderer, RenderError

if TYPE_CHECKING:
    pass


class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass


class MangaPipeline:
    """Main pipeline for manga translation.
    
    Coordinates OCR, translation, and rendering services
    while handling errors gracefully.
    """
    
    def __init__(
        self,
        config: AppConfig | None = None,
        ocr_service: BaseOCRService | None = None,
        translator: BaseTranslator | None = None,
        renderer: Renderer | None = None,
    ):
        """Initialize the pipeline.
        
        Args:
            config: Application configuration
            ocr_service: OCR service (uses DummyOCRService if not provided)
            translator: Translation service (creates based on config if not provided)
            renderer: Renderer service (creates with config if not provided)
        """
        self.config = config or get_config()
        
        # Initialize services
        self.ocr = ocr_service or create_ocr_service(self.config.ocr)
        self.translator = translator or create_translator(self.config.translation)
        self.renderer = renderer or Renderer(self.config.render)
        
        # Ensure directories exist
        self.config.ensure_directories()
        
        logger.info("MangaPipeline initialized")
    
    def process_image(
        self,
        image_path: str | Path,
        output_path: str | Path | None = None,
    ) -> ProcessingResult:
        """Process a single manga image through the full pipeline.
        
        Args:
            image_path: Path to the input image
            output_path: Optional path for the output image
            
        Returns:
            ProcessingResult with success/failure status
        """
        image_path = Path(image_path)
        warnings = []
        
        logger.info(f"Processing image: {image_path}")
        
        # 1. Load image
        try:
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            logger.debug(f"Loaded image: {width}x{height}")
        except Exception as e:
            error = f"Failed to load image: {e}"
            logger.error(error)
            return ProcessingResult.fail(error)
        
        # 2. Create MangaPage
        page = MangaPage(
            page_number=0,
            image_path=str(image_path),
            resolution=(width, height),
        )
        
        # 3. Run OCR
        try:
            text_blocks = self.ocr.process_image(image)
            page.text_blocks = text_blocks
            page.is_processed = True
            logger.info(f"Detected {len(text_blocks)} text blocks")
        except OCRError as e:
            error = f"OCR failed: {e}"
            logger.error(error)
            page.error_message = error
            return ProcessingResult.fail(error)
        
        if not text_blocks:
            warnings.append("No text blocks detected")
            logger.warning("No text blocks detected")
            return ProcessingResult.ok(page, warnings)
        
        # 4. Group/merge nearby blocks (optional enhancement)
        text_blocks = self._group_blocks(text_blocks)
        page.text_blocks = text_blocks
        
        # 5. Translate
        try:
            texts = [block.original_text for block in text_blocks]
            translations = self.translator.translate_batch(texts)
            
            for block, translation in zip(text_blocks, translations):
                block.translated_text = translation
            
            page.is_translated = True
            logger.info(f"Translated {len(translations)} text blocks")
            
        except TranslationError as e:
            error = f"Translation failed: {e}"
            logger.error(error)
            warnings.append(error)
            # Continue with partial results
        
        # 6. Render
        if output_path or any(block.translated_text for block in text_blocks):
            try:
                if output_path is None:
                    output_path = self.config.output_dir / f"translated_{image_path.name}"
                
                self.renderer.render_page(page, output_path)
                logger.info(f"Rendered to: {output_path}")
                
            except RenderError as e:
                error = f"Rendering failed: {e}"
                logger.error(error)
                warnings.append(error)
        
        return ProcessingResult.ok(page, warnings)
    
    def _group_blocks(self, blocks: list[TextBlock]) -> list[TextBlock]:
        """Group nearby text blocks.
        
        This is a simple implementation that could be enhanced
        to merge blocks based on spatial proximity.
        
        Args:
            blocks: Input text blocks
            
        Returns:
            Grouped text blocks
        """
        # For now, just return the original blocks
        # A more sophisticated implementation would:
        # 1. Calculate distances between block centers
        # 2. Merge blocks within merge_distance_threshold
        # 3. Combine their text
        return blocks
    
    def process_batch(
        self,
        image_paths: list[str | Path],
        output_dir: str | Path | None = None,
    ) -> list[ProcessingResult]:
        """Process multiple manga images.
        
        Args:
            image_paths: List of input image paths
            output_dir: Directory for output images
            
        Returns:
            List of ProcessingResult objects
        """
        output_dir = Path(output_dir) if output_dir else self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        for i, image_path in enumerate(image_paths):
            image_path = Path(image_path)
            output_path = output_dir / f"translated_{image_path.name}"
            
            logger.info(f"Processing {i+1}/{len(image_paths)}: {image_path.name}")
            result = self.process_image(image_path, output_path)
            results.append(result)
            
            if not result.success:
                logger.warning(f"Failed to process {image_path.name}: {result.error}")
        
        # Summary
        successful = sum(1 for r in results if r.success)
        logger.info(f"Batch complete: {successful}/{len(results)} successful")
        
        return results
    
    def extract_text(
        self,
        image_path: str | Path,
    ) -> list[TextBlock]:
        """Extract text from an image without translation.
        
        Useful for debugging or text-only extraction.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            List of detected TextBlock objects
        """
        image_path = Path(image_path)
        
        try:
            image = Image.open(image_path).convert("RGB")
            return self.ocr.process_image(image)
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return []
    
    def translate_only(
        self,
        texts: list[str],
    ) -> list[str]:
        """Translate texts without OCR.
        
        Useful for re-translating or testing translation.
        
        Args:
            texts: List of texts to translate
            
        Returns:
            List of translations
        """
        return self.translator.translate_batch(texts)


def create_pipeline(
    config: AppConfig | None = None,
    use_dummy_ocr: bool = True,
) -> MangaPipeline:
    """Factory function to create a pipeline.
    
    Args:
        config: Application configuration
        use_dummy_ocr: Whether to use dummy OCR (for testing)
        
    Returns:
        Configured MangaPipeline instance
    """
    config = config or get_config()
    
    # Choose OCR service
    if use_dummy_ocr:
        ocr_service = DummyOCRService(config.ocr)
    else:
        from manger.services.ocr import MagiOCRService
        ocr_service = MagiOCRService(config.ocr)
    
    return MangaPipeline(
        config=config,
        ocr_service=ocr_service,
    )