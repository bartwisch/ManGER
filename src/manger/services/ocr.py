"""OCR Service for text detection and recognition.

The OCR service wraps different OCR backends (Magi, etc.) and ensures
all coordinates are normalized immediately upon detection.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any
from collections import Counter
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

# Check for manga-ocr availability
try:
    from manga_ocr import MangaOcr
    MANGA_OCR_AVAILABLE = True
except ImportError:
    MANGA_OCR_AVAILABLE = False
    MangaOcr = None


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

    def _extract_text_color(
        self,
        image: Image.Image,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
    ) -> tuple[int, int, int] | None:
        """Determine text color based on background brightness.
        
        Returns white text for dark backgrounds, black text for light backgrounds.
        
        Args:
            image: Source image
            x_min, y_min, x_max, y_max: Pixel coordinates
            
        Returns:
            RGB tuple - either white (255,255,255) or black (0,0,0)
        """
        try:
            # Ensure valid crop coordinates
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(image.width, x_max)
            y_max = min(image.height, y_max)
            
            if x_max <= x_min or y_max <= y_min:
                return None
            
            # Crop the text region
            region = image.crop((x_min, y_min, x_max, y_max))
            
            # Convert to RGB if necessary
            if region.mode != "RGB":
                region = region.convert("RGB")
            
            # Get all pixels
            pixels = list(region.getdata())
            
            if not pixels:
                return None
            
            # Calculate average brightness
            total_brightness = 0
            for r, g, b in pixels:
                # Perceived brightness formula
                brightness = 0.299 * r + 0.587 * g + 0.114 * b
                total_brightness += brightness
            
            avg_brightness = total_brightness / len(pixels)
            
            # If background is dark (< 128), use white text
            # If background is light (>= 128), use black text
            if avg_brightness < 128:
                text_color = (255, 255, 255)  # White
            else:
                text_color = (0, 0, 0)  # Black
            
            logger.debug(f"Background brightness: {avg_brightness:.1f}, using text color: {text_color}")
            return text_color
            
        except Exception as e:
            logger.warning(f"Failed to determine text color: {e}")
            return None


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
    
    Can optionally use manga-ocr for better text recognition.
    
    Requires: pip install manger[magi]
    """
    
    def __init__(self, config: OCRConfig | None = None, use_manga_ocr: bool = True, model_version: str = "v1"):
        """Initialize Magi OCR service.
        
        Args:
            config: OCR configuration
            use_manga_ocr: If True, use manga-ocr for text recognition instead of Magi's OCR
            model_version: "v1" for ragavsachdeva/magi (better OCR), "v2" for ragavsachdeva/magiv2
        """
        super().__init__(config)
        self._model = None
        self._device = None
        self._use_manga_ocr = use_manga_ocr and MANGA_OCR_AVAILABLE
        self._manga_ocr = None
        self._model_version = model_version
    
    def load_model(self) -> None:
        """Load the Magi model from HuggingFace."""
        if not MAGI_AVAILABLE:
            raise OCRModelLoadError(
                "Magi dependencies not installed. Install with: pip install manger[magi]"
            )
        
        try:
            # Select model based on version
            model_name = "ragavsachdeva/magi" if self._model_version == "v1" else "ragavsachdeva/magiv2"
            logger.info(f"Loading Magi model ({model_name}) from HuggingFace...")
            
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
                model_name,
                trust_remote_code=True
            )
            
            # Try to move to device, fallback to CPU if CUDA fails
            try:
                self._model = self._model.to(self._device).eval()
                # Test CUDA with a dummy operation to catch kernel errors early
                if self._device == "cuda":
                    _ = torch.zeros(1).to(self._device)
            except RuntimeError as e:
                if "CUDA" in str(e) or "kernel" in str(e).lower():
                    logger.warning(f"CUDA failed ({e}), falling back to CPU")
                    self._device = "cpu"
                    self._model = self._model.to("cpu").eval()
                else:
                    raise
            
            self._model_loaded = True
            logger.info(f"Magi model ({self._model_version}) loaded successfully")
            
            # Load manga-ocr if enabled
            if self._use_manga_ocr:
                try:
                    logger.info("Loading manga-ocr for text recognition...")
                    self._manga_ocr = MangaOcr()
                    logger.info("manga-ocr loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load manga-ocr: {e}. Using Magi OCR instead.")
                    self._use_manga_ocr = False
                    self._manga_ocr = None
            
        except Exception as e:
            raise OCRModelLoadError(f"Failed to load Magi model: {e}") from e
    
    def _normalize_aspect_ratio(self, image: Image.Image) -> tuple[Image.Image, dict]:
        """Normalize image aspect ratio for better OCR on extreme formats.
        
        Very narrow or very wide images can cause issues with Magi's detection.
        This pads the image to a more reasonable aspect ratio (max 2:1 or 1:2).
        
        Args:
            image: Original PIL Image
            
        Returns:
            Tuple of (normalized image, transform info for coordinate adjustment)
        """
        width, height = image.size
        aspect = width / height
        
        # Only normalize if aspect ratio is extreme (< 0.5 or > 2.0)
        if 0.5 <= aspect <= 2.0:
            return image, {"normalized": False}
        
        import cv2
        import numpy as np
        
        img_array = np.array(image)
        
        if aspect < 0.5:
            # Very tall/narrow image - pad width to make it 1:2
            target_width = height // 2
            pad_total = target_width - width
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            
            # Pad with white (common manga background)
            padded = cv2.copyMakeBorder(
                img_array, 0, 0, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            
            logger.debug(f"Normalized narrow image: {width}x{height} -> {padded.shape[1]}x{padded.shape[0]} (padded {pad_left}+{pad_right}px width)")
            
            return Image.fromarray(padded), {
                "normalized": True,
                "original_size": (width, height),
                "pad_left": pad_left,
                "pad_right": pad_right,
                "pad_top": 0,
                "pad_bottom": 0,
            }
            
        else:
            # Very wide image - pad height to make it 2:1
            target_height = width // 2
            pad_total = target_height - height
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            
            padded = cv2.copyMakeBorder(
                img_array, pad_top, pad_bottom, 0, 0,
                cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            
            logger.debug(f"Normalized wide image: {width}x{height} -> {padded.shape[1]}x{padded.shape[0]} (padded {pad_top}+{pad_bottom}px height)")
            
            return Image.fromarray(padded), {
                "normalized": True,
                "original_size": (width, height),
                "pad_left": 0,
                "pad_right": 0,
                "pad_top": pad_top,
                "pad_bottom": pad_bottom,
            }
    
    def _adjust_bbox_for_normalization(self, bbox: list, transform: dict, normalized_size: tuple) -> list:
        """Adjust bounding box coordinates back to original image coordinates.
        
        Args:
            bbox: Bounding box [x_min, y_min, x_max, y_max] in normalized image coordinates
            transform: Transform info from _normalize_aspect_ratio
            normalized_size: Size of the normalized image (width, height)
            
        Returns:
            Adjusted bounding box in original image coordinates
        """
        if not transform.get("normalized", False):
            return bbox
        
        x_min, y_min, x_max, y_max = bbox[:4]
        pad_left = transform["pad_left"]
        pad_top = transform["pad_top"]
        orig_width, orig_height = transform["original_size"]
        
        # Subtract padding offset
        x_min = x_min - pad_left
        x_max = x_max - pad_left
        y_min = y_min - pad_top
        y_max = y_max - pad_top
        
        # Clamp to original image bounds
        x_min = max(0, min(x_min, orig_width))
        x_max = max(0, min(x_max, orig_width))
        y_min = max(0, min(y_min, orig_height))
        y_max = max(0, min(y_max, orig_height))
        
        return [x_min, y_min, x_max, y_max]
    
    def _prepare_image(self, image: Image.Image, scale_factor: int = 2) -> tuple[np.ndarray, int]:
        """Convert PIL Image to format expected by Magi with preprocessing.
        
        Applies scaling and contrast enhancement for better OCR results.
        
        Args:
            image: PIL Image
            scale_factor: How much to scale up the image (2 = double size)
            
        Returns:
            Tuple of (processed numpy array, scale_factor used)
        """
        # Ensure RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        img_array = np.array(image)
        
        # Import cv2 for preprocessing
        try:
            import cv2
            
            # Scale up for better recognition of small/thin characters
            if scale_factor > 1:
                img_array = cv2.resize(
                    img_array, None, 
                    fx=scale_factor, fy=scale_factor, 
                    interpolation=cv2.INTER_CUBIC
                )
            
            # Convert to grayscale for enhancement
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # This helps with varying lighting and improves character edges
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Light denoising to preserve details but reduce noise
            denoised = cv2.fastNlMeansDenoising(enhanced, h=5, templateWindowSize=7, searchWindowSize=21)
            
            # Convert back to RGB (Magi expects RGB)
            img_array = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
            
            logger.debug(f"Preprocessed image: scaled {scale_factor}x, applied CLAHE and denoising")
            
        except ImportError:
            logger.warning("OpenCV not available, skipping preprocessing")
            scale_factor = 1
        
        return img_array, scale_factor
    
    def _run_manga_ocr(
        self, 
        original_image: Image.Image, 
        text_bboxes: list, 
        scale_factor: int
    ) -> list[str]:
        """Run manga-ocr on detected text regions.
        
        Args:
            original_image: Original PIL Image (not scaled)
            text_bboxes: List of bounding boxes from Magi (in scaled coordinates)
            scale_factor: Scale factor used for Magi detection
            
        Returns:
            List of OCR texts for each bounding box
        """
        if self._manga_ocr is None:
            return [""] * len(text_bboxes)
        
        width, height = original_image.size
        ocr_texts = []
        
        for bbox in text_bboxes:
            try:
                # Check if coordinates are already normalized (0-1 range)
                if all(0 <= coord <= 1 for coord in bbox[:4]):
                    # Normalized - convert to pixels
                    x_min = int(bbox[0] * width)
                    y_min = int(bbox[1] * height)
                    x_max = int(bbox[2] * width)
                    y_max = int(bbox[3] * height)
                else:
                    # Pixel coordinates from scaled image - scale back
                    x_min = int(bbox[0] / scale_factor)
                    y_min = int(bbox[1] / scale_factor)
                    x_max = int(bbox[2] / scale_factor)
                    y_max = int(bbox[3] / scale_factor)
                
                # Add padding for better OCR
                box_width = x_max - x_min
                box_height = y_max - y_min
                padding_x = int(box_width * 0.1)
                padding_y = int(box_height * 0.1)
                
                x_min = max(0, x_min - padding_x)
                y_min = max(0, y_min - padding_y)
                x_max = min(width, x_max + padding_x)
                y_max = min(height, y_max + padding_y)
                
                # Crop the region
                crop = original_image.crop((x_min, y_min, x_max, y_max))
                
                # Run manga-ocr on the crop
                text = self._manga_ocr(crop)
                ocr_texts.append(text)
                logger.debug(f"manga-ocr result: '{text}'")
                
            except Exception as e:
                logger.warning(f"manga-ocr failed for region: {e}")
                ocr_texts.append("")
        
        return ocr_texts
    
    def detect_text(self, image: Image.Image) -> list[TextBlock]:
        """Detect text using Magi model.
        
        This performs full detection including:
        - Text bounding boxes
        - OCR on detected text regions (using manga-ocr if available)
        - Character detection (for speaker association)
        """
        if self._model is None:
            raise OCRProcessingError("Model not loaded. Call load_model() first.")
        
        # Store original dimensions
        orig_width, orig_height = image.size
        
        # Normalize aspect ratio for extreme formats (very narrow/wide images)
        normalized_image, aspect_transform = self._normalize_aspect_ratio(image)
        norm_width, norm_height = normalized_image.size
        
        # Prepare for Magi
        np_image, scale_factor = self._prepare_image(normalized_image, scale_factor=2)
        
        if aspect_transform.get("normalized"):
            logger.debug(f"Running Magi detection on normalized image {norm_width}x{norm_height} (original: {orig_width}x{orig_height}, scaled {scale_factor}x)")
        else:
            logger.debug(f"Running Magi detection on image {orig_width}x{orig_height} (scaled {scale_factor}x)")
        
        try:
            with torch.no_grad():
                # For single image processing, we use predict_detections_and_associations
                # which returns text boxes, character boxes, etc.
                try:
                    results = self._model.predict_detections_and_associations([np_image])
                except RuntimeError as cuda_err:
                    if "CUDA" in str(cuda_err) or "kernel" in str(cuda_err).lower():
                        logger.warning(f"CUDA inference failed ({cuda_err}), retrying on CPU")
                        self._device = "cpu"
                        self._model = self._model.to("cpu")
                        results = self._model.predict_detections_and_associations([np_image])
                    else:
                        raise
                result = results[0]  # Get first (only) image result
                
                # Get text bounding boxes
                text_bboxes = result.get("texts", [])
                
                if len(text_bboxes) == 0:
                    logger.debug("No text detected by Magi")
                    return []
                
                # Run OCR on detected text regions
                if self._use_manga_ocr and self._manga_ocr is not None:
                    # Use manga-ocr for text recognition
                    logger.debug("Using manga-ocr for text recognition")
                    ocr_texts = self._run_manga_ocr(image, text_bboxes, scale_factor)
                else:
                    # Use Magi's built-in OCR
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
                # Already normalized - convert to pixel coords in normalized image
                x_min, y_min, x_max, y_max = bbox[:4]
                x_min_px = int(x_min * norm_width)
                y_min_px = int(y_min * norm_height)
                x_max_px = int(x_max * norm_width)
                y_max_px = int(y_max * norm_height)
            else:
                # Pixel coordinates - need to scale back from Magi's scaled size
                x_min_px = int(bbox[0] / scale_factor)
                y_min_px = int(bbox[1] / scale_factor)
                x_max_px = int(bbox[2] / scale_factor)
                y_max_px = int(bbox[3] / scale_factor)
            
            # Adjust coordinates back to original image if aspect ratio was normalized
            if aspect_transform.get("normalized"):
                adjusted = self._adjust_bbox_for_normalization(
                    [x_min_px, y_min_px, x_max_px, y_max_px],
                    aspect_transform,
                    (norm_width, norm_height)
                )
                x_min_px, y_min_px, x_max_px, y_max_px = [int(c) for c in adjusted]
                logger.debug(f"Adjusted bbox {i} for aspect normalization: [{x_min_px}, {y_min_px}, {x_max_px}, {y_max_px}]")
            
            # Apply minimal padding - just enough to not cut off text
            # Keep it tight so polygon detection works better
            box_width = x_max_px - x_min_px
            box_height = y_max_px - y_min_px
            padding_left = int(box_width * 0.08)   # 8% left padding
            padding_right = int(box_width * 0.02)  # 2% right padding
            padding_y = int(box_height * 0.02)     # 2% vertical padding
            
            x_min_px = max(0, x_min_px - padding_left)
            y_min_px = max(0, y_min_px - padding_y)
            x_max_px = min(orig_width, x_max_px + padding_right)
            y_max_px = min(orig_height, y_max_px + padding_y)
            
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
            
            # Extract text color from original image
            text_color = self._extract_text_color(
                image, x_min_px, y_min_px, x_max_px, y_max_px
            )
            
            block = TextBlock(
                bbox=self._normalize_coordinates(
                    x_min_px, y_min_px, x_max_px, y_max_px,
                    orig_width, orig_height
                ),
                original_text=text,
                confidence=0.9,  # Magi doesn't provide per-box confidence
                is_vertical=is_vertical,
                speaker_id=speaker_id,
                text_color=text_color,
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