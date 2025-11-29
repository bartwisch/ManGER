"""Domain models for the Manga Translation application.

All coordinate systems use normalized values (0.0-1.0) for resolution independence.
"""

from __future__ import annotations

import uuid
from typing import Tuple

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Normalized bounding box with coordinates in range 0.0-1.0.
    
    This ensures resolution independence - coordinates are only converted
    to absolute pixels at the final rendering stage.
    """
    
    x_min: float = Field(..., ge=0.0, le=1.0, description="Left edge (normalized)")
    y_min: float = Field(..., ge=0.0, le=1.0, description="Top edge (normalized)")
    x_max: float = Field(..., ge=0.0, le=1.0, description="Right edge (normalized)")
    y_max: float = Field(..., ge=0.0, le=1.0, description="Bottom edge (normalized)")

    def to_pixels(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Convert normalized coordinates to absolute pixel values.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Tuple of (x_min, y_min, x_max, y_max) in absolute pixels
        """
        return (
            int(self.x_min * width),
            int(self.y_min * height),
            int(self.x_max * width),
            int(self.y_max * height),
        )

    @classmethod
    def from_pixels(
        cls, x_min: int, y_min: int, x_max: int, y_max: int, width: int, height: int
    ) -> BoundingBox:
        """Create a BoundingBox from absolute pixel coordinates.
        
        Args:
            x_min: Left edge in pixels
            y_min: Top edge in pixels
            x_max: Right edge in pixels
            y_max: Bottom edge in pixels
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            BoundingBox with normalized coordinates
        """
        return cls(
            x_min=x_min / width,
            y_min=y_min / height,
            x_max=x_max / width,
            y_max=y_max / height,
        )

    @property
    def width_normalized(self) -> float:
        """Get normalized width of the bounding box."""
        return self.x_max - self.x_min

    @property
    def height_normalized(self) -> float:
        """Get normalized height of the bounding box."""
        return self.y_max - self.y_min

    @property
    def center(self) -> Tuple[float, float]:
        """Get the center point of the bounding box (normalized)."""
        return (
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2,
        )

    @property
    def area(self) -> float:
        """Get the normalized area of the bounding box."""
        return self.width_normalized * self.height_normalized


class TextBlock(BaseModel):
    """Represents a detected text region with OCR results and translation.
    
    Tracks the full lifecycle of text: detection -> OCR -> translation.
    """
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    bbox: BoundingBox
    original_text: str = Field(default="", description="OCR-extracted text")
    translated_text: str | None = Field(default=None, description="Translated text")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="OCR confidence score"
    )
    speaker_id: int | None = Field(
        default=None, description="Speaker grouping from Magi"
    )
    is_vertical: bool = Field(
        default=False, description="Whether text is vertical (common in manga)"
    )
    
    # Traceability fields
    raw_ocr_result: dict | None = Field(
        default=None, description="Original OCR output for debugging"
    )

    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "id": "abc123",
                "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
                "original_text": "こんにちは",
                "translated_text": "Hello",
                "confidence": 0.95,
                "speaker_id": 1,
                "is_vertical": True,
            }
        }


class MangaPage(BaseModel):
    """Represents a single manga page with all detected text blocks.
    
    Acts as the main data container for the translation pipeline.
    """
    
    page_number: int = Field(..., ge=0, description="Page number (0-indexed)")
    image_path: str = Field(..., description="Path to the source image")
    resolution: Tuple[int, int] = Field(
        ..., description="Image resolution as (width, height)"
    )
    text_blocks: list[TextBlock] = Field(
        default_factory=list, description="Detected text blocks"
    )
    
    # Processing state
    is_processed: bool = Field(default=False, description="Whether OCR has been run")
    is_translated: bool = Field(default=False, description="Whether translation is complete")
    error_message: str | None = Field(
        default=None, description="Error message if processing failed"
    )

    @property
    def width(self) -> int:
        """Get image width in pixels."""
        return self.resolution[0]

    @property
    def height(self) -> int:
        """Get image height in pixels."""
        return self.resolution[1]

    def get_text_blocks_by_confidence(self, min_confidence: float = 0.5) -> list[TextBlock]:
        """Filter text blocks by minimum confidence threshold.
        
        Args:
            min_confidence: Minimum confidence score (0.0-1.0)
            
        Returns:
            List of text blocks meeting the confidence threshold
        """
        return [tb for tb in self.text_blocks if tb.confidence >= min_confidence]

    def get_untranslated_blocks(self) -> list[TextBlock]:
        """Get all text blocks that haven't been translated yet."""
        return [tb for tb in self.text_blocks if tb.translated_text is None]


class ProcessingResult(BaseModel):
    """Result container for pipeline operations.
    
    Provides a consistent way to handle success/failure states.
    """
    
    success: bool
    page: MangaPage | None = None
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    
    @classmethod
    def ok(cls, page: MangaPage, warnings: list[str] | None = None) -> ProcessingResult:
        """Create a successful result."""
        return cls(success=True, page=page, warnings=warnings or [])
    
    @classmethod
    def fail(cls, error: str) -> ProcessingResult:
        """Create a failed result."""
        return cls(success=False, error=error)