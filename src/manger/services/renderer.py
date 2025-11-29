"""Renderer Service for image processing and typesetting.

Handles:
- In-painting (removing original text)
- Typesetting (drawing translated text)
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from manger.config import RenderConfig, get_config
from manger.domain.models import BoundingBox, MangaPage, TextBlock

if TYPE_CHECKING:
    pass


class RenderError(Exception):
    """Base exception for rendering errors."""
    pass


class Renderer:
    """Renderer for manga page processing.
    
    Handles in-painting (text removal) and typesetting (text rendering).
    """
    
    def __init__(self, config: RenderConfig | None = None):
        """Initialize the renderer.
        
        Args:
            config: Render configuration
        """
        self.config = config or get_config().render
        self._font_cache: dict[int, ImageFont.FreeTypeFont] = {}
    
    def _get_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """Get a font of the specified size.
        
        Uses caching to avoid repeated font loading.
        
        Args:
            size: Font size in pixels
            
        Returns:
            PIL Font object
        """
        if size in self._font_cache:
            return self._font_cache[size]
        
        try:
            if self.config.font_path:
                font = ImageFont.truetype(self.config.font_path, size)
            else:
                # Try to load a system font
                try:
                    font = ImageFont.truetype("arial.ttf", size)
                except OSError:
                    try:
                        # macOS
                        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
                    except OSError:
                        try:
                            # Linux
                            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
                        except OSError:
                            # Fallback to default
                            logger.warning(f"No system font found, using default")
                            font = ImageFont.load_default()
            
            self._font_cache[size] = font
            return font
            
        except Exception as e:
            logger.warning(f"Failed to load font: {e}, using default")
            return ImageFont.load_default()
    
    def inpaint(self, image: Image.Image, blocks: list[TextBlock]) -> Image.Image:
        """Remove original text from the image.
        
        Currently uses simple color fill. Can be extended to use
        OpenCV inpainting for better results.
        
        Args:
            image: Source image
            blocks: Text blocks to remove
            
        Returns:
            Inpainted image
        """
        result = image.copy()
        draw = ImageDraw.Draw(result)
        width, height = image.size
        
        for block in blocks:
            bbox = block.bbox
            
            # Get pixel coordinates
            x1, y1, x2, y2 = bbox.to_pixels(width, height)
            
            # Add padding
            padding_x = int((x2 - x1) * self.config.padding_ratio)
            padding_y = int((y2 - y1) * self.config.padding_ratio)
            
            x1 = max(0, x1 - padding_x)
            y1 = max(0, y1 - padding_y)
            x2 = min(width, x2 + padding_x)
            y2 = min(height, y2 + padding_y)
            
            # Simple inpainting: fill with background color
            if self.config.inpaint_method == "simple":
                # Sample the background color from the edges of the box
                bg_color = self._sample_background_color(image, x1, y1, x2, y2)
                draw.rectangle([x1, y1, x2, y2], fill=bg_color)
            else:
                # OpenCV inpainting would go here
                draw.rectangle(
                    [x1, y1, x2, y2],
                    fill=self.config.background_color
                )
        
        logger.debug(f"Inpainted {len(blocks)} text regions")
        return result
    
    def _sample_background_color(
        self, image: Image.Image, x1: int, y1: int, x2: int, y2: int
    ) -> tuple[int, int, int]:
        """Sample the background color from the edges of a region.
        
        Args:
            image: Source image
            x1, y1, x2, y2: Region bounds
            
        Returns:
            RGB color tuple
        """
        try:
            # Sample pixels from the border
            samples = []
            
            # Top edge
            for x in range(x1, min(x2, x1 + 10)):
                if 0 <= y1 < image.height and 0 <= x < image.width:
                    samples.append(image.getpixel((x, y1)))
            
            # Left edge
            for y in range(y1, min(y2, y1 + 10)):
                if 0 <= y < image.height and 0 <= x1 < image.width:
                    samples.append(image.getpixel((x1, y)))
            
            if samples:
                # Average the samples
                r = sum(s[0] if isinstance(s, tuple) else s for s in samples) // len(samples)
                g = sum(s[1] if isinstance(s, tuple) and len(s) > 1 else r for s in samples) // len(samples)
                b = sum(s[2] if isinstance(s, tuple) and len(s) > 2 else r for s in samples) // len(samples)
                return (r, g, b)
        except Exception as e:
            logger.debug(f"Failed to sample background: {e}")
        
        return self.config.background_color
    
    def typeset(
        self,
        image: Image.Image,
        blocks: list[TextBlock],
        use_translated: bool = True,
    ) -> Image.Image:
        """Render text onto the image.
        
        Args:
            image: Source image (should be inpainted first)
            blocks: Text blocks with translations
            use_translated: Whether to use translated_text (True) or original_text
            
        Returns:
            Image with typeset text
        """
        result = image.copy()
        draw = ImageDraw.Draw(result)
        width, height = image.size
        
        for block in blocks:
            text = block.translated_text if use_translated else block.original_text
            if not text:
                continue
            
            bbox = block.bbox
            x1, y1, x2, y2 = bbox.to_pixels(width, height)
            
            # Calculate box dimensions with padding
            box_width = x2 - x1
            box_height = y2 - y1
            padding = int(min(box_width, box_height) * self.config.padding_ratio)
            
            inner_width = box_width - (2 * padding)
            inner_height = box_height - (2 * padding)
            
            if inner_width <= 0 or inner_height <= 0:
                continue
            
            # Find optimal font size
            font, wrapped_text = self._fit_text(
                text, inner_width, inner_height, block.is_vertical
            )
            
            # Calculate text position (centered)
            text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            text_x = x1 + padding + (inner_width - text_width) // 2
            text_y = y1 + padding + (inner_height - text_height) // 2
            
            # Draw text
            draw.text(
                (text_x, text_y),
                wrapped_text,
                fill=self.config.text_color,
                font=font,
            )
        
        logger.debug(f"Typeset {len(blocks)} text blocks")
        return result
    
    def _fit_text(
        self,
        text: str,
        max_width: int,
        max_height: int,
        is_vertical: bool = False,
    ) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, str]:
        """Find the optimal font size and text wrapping.
        
        Args:
            text: Text to fit
            max_width: Maximum width in pixels
            max_height: Maximum height in pixels
            is_vertical: Whether to use vertical layout
            
        Returns:
            Tuple of (font, wrapped_text)
        """
        # For vertical text, we would need special handling
        # For now, we just use horizontal text
        
        # Binary search for optimal font size
        min_size = self.config.min_font_size
        max_size = self.config.max_font_size
        
        best_font = None
        best_text = text
        
        while min_size <= max_size:
            mid_size = (min_size + max_size) // 2
            font = self._get_font(mid_size)
            
            # Try different wrap widths
            wrapped = self._wrap_text(text, font, max_width)
            
            # Measure the wrapped text
            dummy_img = Image.new("RGB", (1, 1))
            dummy_draw = ImageDraw.Draw(dummy_img)
            
            try:
                text_bbox = dummy_draw.textbbox((0, 0), wrapped, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except Exception:
                # Fallback for older Pillow versions
                text_width, text_height = dummy_draw.textsize(wrapped, font=font)
            
            if text_width <= max_width and text_height <= max_height:
                best_font = font
                best_text = wrapped
                min_size = mid_size + 1
            else:
                max_size = mid_size - 1
        
        if best_font is None:
            best_font = self._get_font(self.config.min_font_size)
            best_text = self._wrap_text(text, best_font, max_width)
        
        return best_font, best_text
    
    def _wrap_text(
        self, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, max_width: int
    ) -> str:
        """Wrap text to fit within a maximum width.
        
        Args:
            text: Text to wrap
            font: Font to use for measuring
            max_width: Maximum width in pixels
            
        Returns:
            Wrapped text with newlines
        """
        # Start with character-based wrapping estimation
        avg_char_width = max_width // max(1, len(text)) if text else 10
        
        # Try to measure actual character widths
        dummy_img = Image.new("RGB", (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        
        try:
            # Measure a sample character
            sample_bbox = dummy_draw.textbbox((0, 0), "W", font=font)
            avg_char_width = sample_bbox[2] - sample_bbox[0]
        except Exception:
            pass
        
        # Estimate characters per line
        chars_per_line = max(1, max_width // max(1, avg_char_width))
        
        # Use textwrap for word-based wrapping
        wrapper = textwrap.TextWrapper(
            width=chars_per_line,
            break_long_words=True,
            break_on_hyphens=True,
        )
        
        wrapped_lines = wrapper.wrap(text)
        return "\n".join(wrapped_lines)
    
    def render_page(
        self,
        page: MangaPage,
        output_path: Path | str | None = None,
    ) -> Image.Image:
        """Render a complete manga page with translations.
        
        Args:
            page: MangaPage with translated text blocks
            output_path: Optional path to save the result
            
        Returns:
            Rendered image
        """
        # Load the source image
        source_image = Image.open(page.image_path).convert("RGB")
        
        # Get blocks with translations
        translated_blocks = [
            block for block in page.text_blocks
            if block.translated_text is not None
        ]
        
        if not translated_blocks:
            logger.warning("No translated blocks to render")
            return source_image
        
        # Inpaint
        inpainted = self.inpaint(source_image, translated_blocks)
        
        # Typeset
        result = self.typeset(inpainted, translated_blocks)
        
        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.save(output_path, quality=95)
            logger.info(f"Saved rendered page to {output_path}")
        
        return result