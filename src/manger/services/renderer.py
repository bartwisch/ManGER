"""Renderer Service for image processing and typesetting.

Handles:
- In-painting (removing original text)
- Typesetting (drawing translated text)
- Text mask generation for precise text coverage
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

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
    
    def extract_text_polygon(
        self,
        image: Image.Image,
        bbox: BoundingBox,
        threshold: int = 200,
        min_area: int = 50,
    ) -> list[tuple[int, int]]:
        """Extract a polygon that covers the speech bubble containing the text.
        
        Uses edge detection to find the speech bubble boundary, then falls back
        to text-based detection if no bubble is found.
        
        Args:
            image: Source image
            bbox: Bounding box containing text
            threshold: Grayscale threshold for text detection (higher = more sensitive)
            min_area: Minimum area for valid contours
            
        Returns:
            List of (x, y) polygon points in absolute pixels
        """
        width, height = image.size
        x1, y1, x2, y2 = bbox.to_pixels(width, height)
        
        # Expand the search area to find the full speech bubble
        box_w = x2 - x1
        box_h = y2 - y1
        expand_x = int(box_w * 0.3)
        expand_y = int(box_h * 0.3)
        
        search_x1 = max(0, x1 - expand_x)
        search_y1 = max(0, y1 - expand_y)
        search_x2 = min(width, x2 + expand_x)
        search_y2 = min(height, y2 + expand_y)
        
        # Try to detect speech bubble first
        bubble_polygon = self._detect_speech_bubble(
            image, search_x1, search_y1, search_x2, search_y2,
            x1, y1, x2, y2  # Original text bbox for validation
        )
        
        if bubble_polygon and len(bubble_polygon) >= 3:
            return bubble_polygon
        
        # Fallback: detect text region
        return self._detect_text_region(image, x1, y1, x2, y2, threshold)
    
    def _detect_speech_bubble(
        self,
        image: Image.Image,
        search_x1: int, search_y1: int,
        search_x2: int, search_y2: int,
        text_x1: int, text_y1: int,
        text_x2: int, text_y2: int,
    ) -> list[tuple[int, int]] | None:
        """Detect the speech bubble boundary around text.
        
        Looks for a closed contour (dark line on light background) that
        encloses the text bounding box.
        
        Returns:
            Polygon points or None if no bubble found
        """
        # Crop the search region
        region = image.crop((search_x1, search_y1, search_x2, search_y2))
        region_gray = region.convert("L")
        region_np = np.array(region_gray)
        
        h, w = region_np.shape
        
        # Detect edges - speech bubbles typically have dark outlines
        # Use Sobel-like edge detection
        edges = self._detect_edges(region_np)
        
        # Find the enclosing contour
        # Start from the center (where text is) and expand outward
        text_center_x = (text_x1 + text_x2) // 2 - search_x1
        text_center_y = (text_y1 + text_y2) // 2 - search_y1
        
        # Ensure center is within bounds
        text_center_x = max(0, min(w - 1, text_center_x))
        text_center_y = max(0, min(h - 1, text_center_y))
        
        # Use flood fill from center to find the bubble interior
        bubble_mask = self._flood_fill_bubble(region_np, edges, text_center_x, text_center_y)
        
        if bubble_mask is None:
            return None
        
        # Extract boundary points from the mask
        boundary_points = self._extract_boundary(bubble_mask)
        
        if len(boundary_points) < 10:
            return None
        
        # Simplify the polygon to reduce points
        simplified = self._simplify_polygon(boundary_points, tolerance=3.0)
        
        # Convert to absolute coordinates
        polygon = [(int(p[0] + search_x1), int(p[1] + search_y1)) for p in simplified]
        
        return polygon
    
    def _detect_edges(self, img: np.ndarray) -> np.ndarray:
        """Detect edges using a simple gradient-based method.
        
        Returns:
            Binary edge image
        """
        h, w = img.shape
        edges = np.zeros_like(img)
        
        # Compute gradients
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                # Sobel-like gradient
                gx = (int(img[y, x + 1]) - int(img[y, x - 1]))
                gy = (int(img[y + 1, x]) - int(img[y - 1, x]))
                gradient = abs(gx) + abs(gy)
                edges[y, x] = min(255, gradient)
        
        # Threshold to get binary edges
        edge_threshold = 30
        return (edges > edge_threshold).astype(np.uint8) * 255
    
    def _flood_fill_bubble(
        self,
        img: np.ndarray,
        edges: np.ndarray,
        start_x: int,
        start_y: int,
    ) -> np.ndarray | None:
        """Flood fill from center to find bubble interior.
        
        Stops at edges (bubble boundary) or dark pixels.
        
        Returns:
            Binary mask of bubble interior, or None if fill escapes bounds
        """
        h, w = img.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Check if starting point is valid (should be light/white background)
        if img[start_y, start_x] < 200:
            # Starting point is too dark, not inside a typical speech bubble
            return None
        
        visited = np.zeros((h, w), dtype=bool)
        stack = [(start_x, start_y)]
        
        filled_count = 0
        max_fill = h * w * 0.8  # Don't fill more than 80% of search area
        
        while stack:
            x, y = stack.pop()
            
            if x < 0 or x >= w or y < 0 or y >= h:
                # Hit the edge of search region - bubble might extend beyond
                continue
            
            if visited[y, x]:
                continue
            
            visited[y, x] = True
            
            # Stop at edges (bubble boundary)
            if edges[y, x] > 0:
                continue
            
            # Stop at very dark pixels (likely panel border or outside bubble)
            if img[y, x] < 100:
                continue
            
            mask[y, x] = 255
            filled_count += 1
            
            if filled_count > max_fill:
                # Filled too much - probably escaped the bubble
                return None
            
            # Add neighbors
            stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])
        
        # Check if we filled a reasonable area
        if filled_count < 100:
            return None
        
        return mask
    
    def _extract_boundary(self, mask: np.ndarray) -> list[tuple[int, int]]:
        """Extract boundary points from a binary mask.
        
        Returns:
            List of (x, y) boundary points
        """
        h, w = mask.shape
        boundary = []
        
        for y in range(h):
            for x in range(w):
                if mask[y, x] > 0:
                    # Check if this is a boundary pixel
                    is_boundary = False
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if ny < 0 or ny >= h or nx < 0 or nx >= w:
                            is_boundary = True
                            break
                        if mask[ny, nx] == 0:
                            is_boundary = True
                            break
                    
                    if is_boundary:
                        boundary.append((x, y))
        
        return boundary
    
    def _simplify_polygon(
        self,
        points: list[tuple[int, int]],
        tolerance: float = 2.0
    ) -> list[tuple[int, int]]:
        """Simplify a polygon using Douglas-Peucker algorithm.
        
        Args:
            points: List of polygon points
            tolerance: Distance tolerance for simplification
            
        Returns:
            Simplified polygon
        """
        if len(points) < 3:
            return points
        
        # Sort points by angle from centroid to create ordered polygon
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        
        import math
        sorted_points = sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
        
        # Apply Douglas-Peucker simplification
        return self._douglas_peucker(sorted_points, tolerance)
    
    def _douglas_peucker(
        self,
        points: list[tuple[int, int]],
        tolerance: float
    ) -> list[tuple[int, int]]:
        """Douglas-Peucker line simplification algorithm."""
        if len(points) <= 2:
            return points
        
        # Find the point with maximum distance from the line between first and last
        first = points[0]
        last = points[-1]
        
        max_dist = 0
        max_idx = 0
        
        for i in range(1, len(points) - 1):
            dist = self._point_line_distance(points[i], first, last)
            if dist > max_dist:
                max_dist = dist
                max_idx = i
        
        # If max distance is greater than tolerance, recursively simplify
        if max_dist > tolerance:
            left = self._douglas_peucker(points[:max_idx + 1], tolerance)
            right = self._douglas_peucker(points[max_idx:], tolerance)
            return left[:-1] + right
        else:
            return [first, last]
    
    def _point_line_distance(
        self,
        point: tuple[int, int],
        line_start: tuple[int, int],
        line_end: tuple[int, int]
    ) -> float:
        """Calculate perpendicular distance from point to line."""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5
        
        t = max(0, min(1, ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)))
        
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        
        return ((x0 - proj_x) ** 2 + (y0 - proj_y) ** 2) ** 0.5
    
    def _detect_text_region(
        self,
        image: Image.Image,
        x1: int, y1: int, x2: int, y2: int,
        threshold: int = 200
    ) -> list[tuple[int, int]]:
        """Fallback: detect text region using thresholding.
        
        Returns:
            Convex hull polygon around text pixels
        """
        # Add small padding
        width, height = image.size
        pad = 3
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(width, x2 + pad)
        y2 = min(height, y2 + pad)
        
        region = image.crop((x1, y1, x2, y2))
        region_gray = region.convert("L")
        region_np = np.array(region_gray)
        
        _, binary = self._threshold_image(region_np, threshold)
        contours = self._find_contours(binary)
        
        if not contours:
            return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        
        all_points = []
        for contour in contours:
            if len(contour) >= 3:
                all_points.extend(contour)
        
        if len(all_points) < 3:
            return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        
        hull = self._convex_hull(all_points)
        polygon = [(int(p[0] + x1), int(p[1] + y1)) for p in hull]
        
        return polygon
    
    def _threshold_image(
        self, img_array: np.ndarray, threshold: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply binary threshold to detect text pixels.
        
        Args:
            img_array: Grayscale image as numpy array
            threshold: Threshold value
            
        Returns:
            Tuple of (original, binary mask)
        """
        # Detect if background is light or dark
        mean_val = np.mean(img_array)
        
        if mean_val > 127:
            # Light background, dark text
            binary = (img_array < threshold).astype(np.uint8) * 255
        else:
            # Dark background, light text
            binary = (img_array > (255 - threshold)).astype(np.uint8) * 255
        
        return img_array, binary
    
    def _find_contours(self, binary: np.ndarray) -> list[list[tuple[int, int]]]:
        """Find contours in a binary image using a simple algorithm.
        
        Args:
            binary: Binary image (0 or 255)
            
        Returns:
            List of contours, each contour is a list of (x, y) points
        """
        # Simple contour tracing using edge detection
        h, w = binary.shape
        visited = np.zeros_like(binary, dtype=bool)
        contours = []
        
        # Find connected components using flood fill approach
        def flood_fill(start_y, start_x):
            points = []
            stack = [(start_y, start_x)]
            
            while stack:
                y, x = stack.pop()
                if y < 0 or y >= h or x < 0 or x >= w:
                    continue
                if visited[y, x] or binary[y, x] == 0:
                    continue
                    
                visited[y, x] = True
                points.append((x, y))
                
                # Check neighbors
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if not visited[ny, nx] and binary[ny, nx] > 0:
                            stack.append((ny, nx))
            
            return points
        
        # Scan for contour starting points
        for y in range(h):
            for x in range(w):
                if binary[y, x] > 0 and not visited[y, x]:
                    points = flood_fill(y, x)
                    if len(points) >= 10:  # Minimum points for valid contour
                        contours.append(points)
        
        return contours
    
    def _convex_hull(self, points: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Compute the convex hull of a set of points using Graham scan.
        
        Args:
            points: List of (x, y) points
            
        Returns:
            Convex hull as list of (x, y) points
        """
        if len(points) < 3:
            return points
        
        # Find the bottom-most point (or left-most in case of tie)
        def bottom_left(points):
            return min(points, key=lambda p: (p[1], p[0]))
        
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        def dist_sq(a, b):
            return (a[0] - b[0])**2 + (a[1] - b[1])**2
        
        start = bottom_left(points)
        
        # Sort points by polar angle with respect to start
        import math
        def angle_key(p):
            if p == start:
                return (-math.inf, 0)
            angle = math.atan2(p[1] - start[1], p[0] - start[0])
            return (angle, dist_sq(start, p))
        
        sorted_points = sorted(points, key=angle_key)
        
        # Remove duplicates
        unique_points = []
        for p in sorted_points:
            if not unique_points or p != unique_points[-1]:
                unique_points.append(p)
        
        if len(unique_points) < 3:
            return unique_points
        
        # Build hull
        hull = []
        for p in unique_points:
            while len(hull) >= 2 and cross(hull[-2], hull[-1], p) <= 0:
                hull.pop()
            hull.append(p)
        
        return hull
    
    def create_text_mask(
        self,
        image: Image.Image,
        blocks: list[TextBlock],
        expand_pixels: int = 2,
    ) -> tuple[Image.Image, list[list[tuple[int, int]]]]:
        """Create a mask covering all text regions with precise polygons.
        
        Args:
            image: Source image
            blocks: Text blocks to mask
            expand_pixels: Pixels to expand the mask by
            
        Returns:
            Tuple of (mask image, list of polygons)
        """
        width, height = image.size
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        polygons = []
        for block in blocks:
            polygon = self.extract_text_polygon(image, block.bbox)
            polygons.append(polygon)
            
            if len(polygon) >= 3:
                draw.polygon(polygon, fill=255)
        
        # Expand the mask slightly
        if expand_pixels > 0:
            mask = mask.filter(ImageFilter.MaxFilter(expand_pixels * 2 + 1))
        
        return mask, polygons
    
    def inpaint_with_mask(
        self,
        image: Image.Image,
        blocks: list[TextBlock],
        use_polygons: bool = True,
    ) -> Image.Image:
        """Remove original text using precise polygon masks.
        
        Args:
            image: Source image
            blocks: Text blocks to remove
            use_polygons: Whether to use polygon masks (True) or rectangles (False)
            
        Returns:
            Inpainted image
        """
        result = image.copy()
        draw = ImageDraw.Draw(result)
        width, height = image.size
        
        for block in blocks:
            if use_polygons:
                polygon = self.extract_text_polygon(image, block.bbox)
                if len(polygon) >= 3:
                    # Sample background color from outside the polygon
                    x1 = min(p[0] for p in polygon)
                    y1 = min(p[1] for p in polygon)
                    x2 = max(p[0] for p in polygon)
                    y2 = max(p[1] for p in polygon)
                    bg_color = self._sample_background_color(image, x1, y1, x2, y2)
                    draw.polygon(polygon, fill=bg_color)
            else:
                bbox = block.bbox
                x1, y1, x2, y2 = bbox.to_pixels(width, height)
                bg_color = self._sample_background_color(image, x1, y1, x2, y2)
                draw.rectangle([x1, y1, x2, y2], fill=bg_color)
        
        logger.debug(f"Inpainted {len(blocks)} text regions with {'polygons' if use_polygons else 'rectangles'}")
        return result
        
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