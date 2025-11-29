"""OCR Service for text detection and recognition.

The OCR service wraps different OCR backends (Magi, etc.) and ensures
all coordinates are normalized immediately upon detection.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any
import random

from loguru import logger
from PIL import Image
import numpy as np

from manger.domain.models import BoundingBox, TextBlock
from manger.config import OCRConfig, get_config

if TYPE_CHECKING:
    pass

# Check for torch/transformers availability
try:
    import torch
    from transformers import AutoModel
    MAGI_AVAILABLE = True
except ImportError:
    MAGI_AVAILABLE = False
    torch = None
    AutoModel = None


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
    
    Uses Magiv2 from HuggingFace: ragavsachdeva/magiv2
    This model provides:
    - Text detection and OCR
    - Character detection
    - Panel detection
    - Text-to-character associations
    
    Requires: pip install manger[magi]
    """
    
    def __init__(self, config: OCRConfig | None = None):
        super().__init__(config)
        self._model = None
        self._device = None
    
    def load_model(self) -> None:
        """Load the Magi model from HuggingFace."""
        if not MAGI_AVAILABLE:
            raise OCRModelLoadError(
                "Magi dependencies not installed. Install with: pip install manger[magi]"
            )
        
        try:
            logger.info("Loading Magi model from HuggingFace...")
            
            # Determine device
            if torch.cuda.is_available():
                self._device = "cuda"
                logger.info("Using CUDA for Magi model")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = "mps"
                logger.info("Using MPS (Apple Silicon) for Magi model")
            else:
                self._device = "cpu"
                logger.warning("Using CPU for Magi model - this will be slow")
            
            # Load model
            self._model = AutoModel.from_pretrained(
                "ragavsachdeva/magiv2",
                trust_remote_code=True
            )
            self._model = self._model.to(self._device).eval()
            
            self._model_loaded = True
            logger.info("Magi model loaded successfully")
            
        except Exception as e:
            raise OCRModelLoadError(f"Failed to load Magi model: {e}") from e
    
    def _prepare_image(self, image: Image.Image) -> np.ndarray:
        """Convert PIL Image to format expected by Magi.
        
        Magi expects RGB numpy arrays.
        """
        # Ensure RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")
        return np.array(image)
    
    def detect_text(self, image: Image.Image) -> list[TextBlock]:
        """Detect text using Magi model.
        
        This performs full detection including:
        - Text bounding boxes
        - OCR on detected text regions
        - Character detection (for speaker association)
        """
        if self._model is None:
            raise OCRProcessingError("Model not loaded. Call load_model() first.")
        
        width, height = image.size
        np_image = self._prepare_image(image)
        
        logger.debug(f"Running Magi detection on image {width}x{height}")
        
        try:
            with torch.no_grad():
                # For single image processing, we use predict_detections_and_associations
                # which returns text boxes, character boxes, etc.
                results = self._model.predict_detections_and_associations([np_image])
                result = results[0]  # Get first (only) image result
                
                # Get text bounding boxes
                text_bboxes = result.get("texts", [])
                
                if len(text_bboxes) == 0:
                    logger.debug("No text detected by Magi")
                    return []
                
                # Run OCR on detected text regions
                ocr_results = self._model.predict_ocr(
                    [np_image], 
                    [text_bboxes]
                )
                ocr_texts = ocr_results[0] if ocr_results else []
                
                # Get character associations if available
                text_char_associations = result.get("text_character_associations", [])
                is_essential = result.get("is_essential_text", [True] * len(text_bboxes))
                
                # Build association map: text_idx -> character_idx
                speaker_map = {}
                for text_idx, char_idx in text_char_associations:
                    speaker_map[text_idx] = char_idx
                
        except Exception as e:
            raise OCRProcessingError(f"Magi detection failed: {e}") from e
        
        # Convert results to TextBlock objects
        blocks = []
        for i, bbox in enumerate(text_bboxes):
            # Magi returns bounding boxes - need to check format
            # Could be [x_min, y_min, x_max, y_max] or [x_center, y_center, width, height]
            # or could already be normalized
            logger.debug(f"Raw bbox {i}: {bbox}")
            
            # Check if coordinates are already normalized (0-1 range)
            if all(0 <= coord <= 1 for coord in bbox[:4]):
                # Already normalized
                x_min, y_min, x_max, y_max = bbox[:4]
                x_min_px = int(x_min * width)
                y_min_px = int(y_min * height)
                x_max_px = int(x_max * width)
                y_max_px = int(y_max * height)
            else:
                # Pixel coordinates
                x_min_px, y_min_px, x_max_px, y_max_px = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # Apply minimal padding - just enough to not cut off text
            # Keep it tight so polygon detection works better
            box_width = x_max_px - x_min_px
            box_height = y_max_px - y_min_px
            padding_left = int(box_width * 0.08)   # 8% left padding
            padding_right = int(box_width * 0.02)  # 2% right padding
            padding_y = int(box_height * 0.02)     # 2% vertical padding
            
            x_min_px = max(0, x_min_px - padding_left)
            y_min_px = max(0, y_min_px - padding_y)
            x_max_px = min(width, x_max_px + padding_right)
            y_max_px = min(height, y_max_px + padding_y)
            
            # Get OCR text (if available)
            text = ocr_texts[i] if i < len(ocr_texts) else ""
            
            # Get speaker ID if associated
            speaker_id = speaker_map.get(i)
            
            # Determine if text is vertical (heuristic based on aspect ratio)
            # Recalculate after padding
            box_width_final = x_max_px - x_min_px
            box_height_final = y_max_px - y_min_px
            is_vertical = box_height_final > box_width_final * 1.5 if box_width_final > 0 else False
            
            # Check if this is essential text (dialogue vs. SFX)
            essential = is_essential[i] if i < len(is_essential) else True
            
            block = TextBlock(
                bbox=self._normalize_coordinates(
                    x_min_px, y_min_px, x_max_px, y_max_px,
                    width, height
                ),
                original_text=text,
                confidence=0.9,  # Magi doesn't provide per-box confidence
                is_vertical=is_vertical,
                speaker_id=speaker_id,
                raw_ocr_result={
                    "source": "magi",
                    "index": i,
                    "is_essential": essential,
                    "bbox_pixels": [x_min_px, y_min_px, x_max_px, y_max_px],
                },
            )
            blocks.append(block)
        
        logger.debug(f"Magi detected {len(blocks)} text blocks")
        return blocks


def create_ocr_service(config: OCRConfig | None = None) -> BaseOCRService:
    """Factory function to create the appropriate OCR service.
    
    Attempts to use Magi if available, falls back to DummyOCRService.
    
    Args:
        config: OCR configuration
        
    Returns:
        Appropriate OCR service instance
    """
    if MAGI_AVAILABLE:
        logger.info("Magi OCR is available, using MagiOCRService")
        return MagiOCRService(config)
    else:
        logger.warning(
            "Magi OCR not available (missing torch/transformers). "
            "Install with: pip install manger[magi]. "
            "Using DummyOCRService for testing."
        )
        return DummyOCRService(config)