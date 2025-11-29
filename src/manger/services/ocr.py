"""OCR Service for text detection and recognition.

The OCR service wraps different OCR backends (Magi, etc.) and ensures
all coordinates are normalized immediately upon detection.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import random

from loguru import logger
from PIL import Image
import numpy as np

from manger.domain.models import BoundingBox, TextBlock
from manger.config import OCRConfig, get_config

if TYPE_CHECKING:
    pass


class OCRError(Exception):
    """Base exception for OCR-related errors."""
    pass


class OCRModelLoadError(OCRError):
    """Raised when the OCR model fails to load."""
    pass


class OCRProcessingError(OCRError):
    """Raised when OCR processing fails."""
    pass


class BaseOCRService(ABC):
    """Abstract base class for OCR services.
    
    All implementations must normalize coordinates to 0-1 range.
    """
    
    def __init__(self, config: OCRConfig | None = None):
        """Initialize the OCR service.
        
        Args:
            config: OCR configuration (uses global config if not provided)
        """
        self.config = config or get_config().ocr
        self._model_loaded = False
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the OCR model into memory.
        
        Raises:
            OCRModelLoadError: If model loading fails
        """
        pass
    
    @abstractmethod
    def detect_text(self, image: Image.Image) -> list[TextBlock]:
        """Detect and recognize text in an image.
        
        Args:
            image: PIL Image to process
            
        Returns:
            List of TextBlock objects with normalized coordinates
            
        Raises:
            OCRProcessingError: If processing fails
        """
        pass
    
    def process_image(self, image: Image.Image) -> list[TextBlock]:
        """Process an image and return filtered text blocks.
        
        This method handles:
        - Loading the model if needed
        - Running detection
        - Filtering by confidence threshold
        - Filtering by minimum text length
        
        Args:
            image: PIL Image to process
            
        Returns:
            List of filtered TextBlock objects
        """
        if not self._model_loaded:
            self.load_model()
        
        logger.debug(f"Processing image of size {image.size}")
        
        try:
            blocks = self.detect_text(image)
        except Exception as e:
            raise OCRProcessingError(f"Failed to process image: {e}") from e
        
        # Apply filters
        filtered = [
            block for block in blocks
            if block.confidence >= self.config.confidence_threshold
            and len(block.original_text) >= self.config.min_text_length
        ]
        
        logger.info(
            f"Detected {len(blocks)} blocks, {len(filtered)} passed filters"
        )
        
        return filtered
    
    def _normalize_coordinates(
        self,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
        width: int,
        height: int,
    ) -> BoundingBox:
        """Convert pixel coordinates to normalized bounding box.
        
        Args:
            x_min, y_min, x_max, y_max: Pixel coordinates
            width, height: Image dimensions
            
        Returns:
            Normalized BoundingBox
        """
        return BoundingBox.from_pixels(x_min, y_min, x_max, y_max, width, height)


class DummyOCRService(BaseOCRService):
    """Dummy OCR service for testing without actual OCR model.
    
    Generates random text blocks for demonstration purposes.
    """
    
    SAMPLE_TEXTS = [
        "こんにちは",
        "ありがとう",
        "お願いします",
        "大丈夫",
        "すごい！",
        "何？",
        "分かった",
        "行こう！",
        "待って",
        "助けて！",
    ]
    
    def load_model(self) -> None:
        """Simulate model loading."""
        logger.info("Loading dummy OCR model...")
        self._model_loaded = True
        logger.info("Dummy OCR model loaded successfully")
    
    def detect_text(self, image: Image.Image) -> list[TextBlock]:
        """Generate random text blocks for testing.
        
        Creates 3-8 random text blocks distributed across the image.
        """
        width, height = image.size
        num_blocks = random.randint(3, 8)
        
        blocks = []
        for i in range(num_blocks):
            # Generate random box in different regions
            region_x = (i % 3) / 3
            region_y = (i // 3) / 3
            
            x_min = region_x + random.uniform(0.02, 0.08)
            y_min = region_y + random.uniform(0.02, 0.08)
            box_width = random.uniform(0.1, 0.25)
            box_height = random.uniform(0.05, 0.15)
            
            x_max = min(x_min + box_width, 0.98)
            y_max = min(y_min + box_height, 0.98)
            
            block = TextBlock(
                bbox=BoundingBox(
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                ),
                original_text=random.choice(self.SAMPLE_TEXTS),
                confidence=random.uniform(0.6, 0.99),
                is_vertical=random.random() > 0.5,
                speaker_id=random.randint(1, 3) if random.random() > 0.3 else None,
                raw_ocr_result={"source": "dummy", "index": i},
            )
            blocks.append(block)
        
        logger.debug(f"Generated {len(blocks)} dummy text blocks")
        return blocks


# Optional: Magi OCR implementation placeholder
class MagiOCRService(BaseOCRService):
    """OCR service using the Magi (Manga Whisperer) model.
    
    This is a placeholder for the actual Magi integration.
    Requires: pip install manger[magi]
    """
    
    def __init__(self, config: OCRConfig | None = None):
        super().__init__(config)
        self._model = None
    
    def load_model(self) -> None:
        """Load the Magi model."""
        try:
            # This would be the actual Magi model loading
            # from magi import MagiModel
            # self._model = MagiModel.load()
            logger.warning(
                "Magi model not available. Install with: pip install manger[magi]"
            )
            raise OCRModelLoadError(
                "Magi model not installed. Use DummyOCRService for testing."
            )
        except ImportError as e:
            raise OCRModelLoadError(f"Failed to import Magi: {e}") from e
    
    def detect_text(self, image: Image.Image) -> list[TextBlock]:
        """Detect text using Magi model."""
        if self._model is None:
            raise OCRProcessingError("Model not loaded")
        
        # Placeholder for actual Magi processing
        # results = self._model.detect(image)
        # return self._convert_magi_results(results, image.size)
        return []