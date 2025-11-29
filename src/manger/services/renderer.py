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
    
    def inpaint(
        self, 
        image: Image.Image, 
        blocks: list[TextBlock],
        use_polygons: bool = True,
    ) -> Image.Image:
        """Remove original text from the image using precise polygons.
        
        Uses the text polygon (speech bubble shape) for precise inpainting,
        falling back to rectangle if polygon detection fails.
        
        Args:
            image: Source image
            blocks: Text blocks to remove
            use_polygons: If True, use precise polygons for inpainting
            
        Returns:
            Inpainted image
        """
        result = image.copy()
        draw = ImageDraw.Draw(result)
        width, height = image.size
        
        for block in blocks:
            bbox = block.bbox
            x1, y1, x2, y2 = bbox.to_pixels(width, height)
            
            # Try to get polygon for precise inpainting
            polygon = None
            if use_polygons:
                try:
                    polygon = self.extract_text_polygon(image, bbox)
                    if polygon and len(polygon) >= 3:
                        logger.debug(f"Using polygon with {len(polygon)} points for inpainting")
                except Exception as e:
                    logger.debug(f"Polygon extraction failed: {e}")
                    polygon = None
            
            # Sample background color
            bg_color = self._sample_background_color(image, x1, y1, x2, y2)
            
            if polygon and len(polygon) >= 3:
                # Use polygon for precise inpainting
                draw.polygon(polygon, fill=bg_color)
            else:
                # Fallback to rectangle
                padding_x = int((x2 - x1) * self.config.padding_ratio)
                padding_y = int((y2 - y1) * self.config.padding_ratio)
                
                x1 = max(0, x1 - padding_x)
                y1 = max(0, y1 - padding_y)
                x2 = min(width, x2 + padding_x)
                y2 = min(height, y2 + padding_y)
                
                draw.rectangle([x1, y1, x2, y2], fill=bg_color)
        
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
        
        # Erode the mask to shrink it inward (away from the bubble outline)
        # This ensures we don't overwrite the speech bubble border
        bubble_mask = self._erode_mask(bubble_mask, iterations=12)
        
        if bubble_mask is None or np.sum(bubble_mask) < 100:
            return None
        
        # Extract boundary points from the mask
        boundary_points = self._extract_boundary(bubble_mask)
        
        if len(boundary_points) < 10:
            return None
        
        # Simplify the polygon to reduce points
        simplified = self._simplify_polygon(boundary_points, tolerance=3.0)
        
        # Smooth the polygon aggressively to round off sharp corners
        # More iterations = rounder corners
        smoothed = self._smooth_polygon(simplified, iterations=10)
        
        # Shrink polygon inward to create buffer from bubble outline
        # Use percentage-based shrinking for better scaling across different bubble sizes
        shrunk = self._shrink_polygon_percent(smoothed, percent=0.15)
        
        # Convert to absolute coordinates
        polygon = [(int(p[0] + search_x1), int(p[1] + search_y1)) for p in shrunk]
        
        return polygon
    
    def _smooth_polygon(
        self,
        polygon: list[tuple[float, float]],
        iterations: int = 2
    ) -> list[tuple[float, float]]:
        """Smooth a polygon by averaging neighboring vertices.
        
        This rounds off sharp corners that might poke outside the bubble.
        
        Args:
            polygon: List of (x, y) points
            iterations: Number of smoothing passes
            
        Returns:
            Smoothed polygon
        """
        if len(polygon) < 5:
            return polygon
        
        result = list(polygon)
        
        for _ in range(iterations):
            new_result = []
            n = len(result)
            for i in range(n):
                # Get surrounding points for stronger smoothing
                prev2_p = result[(i - 2) % n]
                prev_p = result[(i - 1) % n]
                curr_p = result[i]
                next_p = result[(i + 1) % n]
                next2_p = result[(i + 2) % n]
                
                # Always use strong 5-point smoothing to round ALL corners equally
                # This ensures consistent rounding on all sides
                new_x = 0.1 * prev2_p[0] + 0.2 * prev_p[0] + 0.4 * curr_p[0] + 0.2 * next_p[0] + 0.1 * next2_p[0]
                new_y = 0.1 * prev2_p[1] + 0.2 * prev_p[1] + 0.4 * curr_p[1] + 0.2 * next_p[1] + 0.1 * next2_p[1]
                
                new_result.append((new_x, new_y))
            
            result = new_result
        
        return result
    
    def _shrink_polygon(
        self, 
        polygon: list[tuple[int, int]], 
        pixels: int = 5
    ) -> list[tuple[int, int]]:
        """Shrink a polygon inward by moving each vertex toward the centroid.
        
        Args:
            polygon: List of (x, y) points
            pixels: How many pixels to shrink inward
            
        Returns:
            Shrunk polygon
        """
        if len(polygon) < 3:
            return polygon
        
        # Calculate centroid
        cx = sum(p[0] for p in polygon) / len(polygon)
        cy = sum(p[1] for p in polygon) / len(polygon)
        
        shrunk = []
        for px, py in polygon:
            # Vector from point to centroid
            dx = cx - px
            dy = cy - py
            
            # Distance to centroid
            dist = (dx * dx + dy * dy) ** 0.5
            
            if dist > 0:
                # Move point toward centroid by 'pixels' amount
                factor = pixels / dist
                new_x = px + dx * factor
                new_y = py + dy * factor
                shrunk.append((new_x, new_y))
            else:
                shrunk.append((px, py))
        
        return shrunk
    
    def _shrink_polygon_percent(
        self, 
        polygon: list[tuple[float, float]], 
        percent: float = 0.1,
        asymmetric: bool = True
    ) -> list[tuple[float, float]]:
        """Shrink a polygon inward by a percentage of the distance to centroid.
        
        This scales better across different bubble sizes.
        
        Args:
            polygon: List of (x, y) points
            percent: Percentage to shrink (0.1 = 10% toward center)
            asymmetric: If True, shrink less on the left side (for text that starts at left)
            
        Returns:
            Shrunk polygon
        """
        if len(polygon) < 3:
            return polygon
        
        # Calculate centroid
        cx = sum(p[0] for p in polygon) / len(polygon)
        cy = sum(p[1] for p in polygon) / len(polygon)
        
        # Find left and right bounds
        min_x = min(p[0] for p in polygon)
        max_x = max(p[0] for p in polygon)
        width = max_x - min_x
        
        shrunk = []
        for px, py in polygon:
            # Calculate shrink factor
            if asymmetric and width > 0:
                # Less shrink on the left (where text starts), more on the right
                # Left 30% of polygon: shrink only 30% of the percent
                # Right 70%: normal shrink
                relative_x = (px - min_x) / width  # 0 = left, 1 = right
                if relative_x < 0.3:
                    local_percent = percent * 0.3  # Much less shrink on left
                else:
                    local_percent = percent
            else:
                local_percent = percent
            
            # Move point toward centroid by percentage
            new_x = px + (cx - px) * local_percent
            new_y = py + (cy - py) * local_percent
            shrunk.append((new_x, new_y))
        
        return shrunk
    
    def _erode_mask(self, mask: np.ndarray, iterations: int = 3) -> np.ndarray:
        """Erode a binary mask to shrink it inward.
        
        This creates a buffer zone between the mask and the speech bubble outline.
        
        Args:
            mask: Binary mask
            iterations: Number of erosion iterations (more = smaller mask)
            
        Returns:
            Eroded mask
        """
        h, w = mask.shape
        result = mask.copy()
        
        for _ in range(iterations):
            new_result = np.zeros_like(result)
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    # Only keep pixel if all 4-neighbors are also set
                    if result[y, x] > 0:
                        if (result[y-1, x] > 0 and result[y+1, x] > 0 and
                            result[y, x-1] > 0 and result[y, x+1] > 0):
                            new_result[y, x] = 255
            result = new_result
        
        return result
    
    def _detect_edges(self, img: np.ndarray) -> np.ndarray:
        """Detect edges using a simple gradient-based method.
        
        Returns:
            Binary edge image
        """
        h, w = img.shape
        edges = np.zeros_like(img)
        
        # Compute gradients with a slightly larger kernel for better edge detection
        for y in range(2, h - 2):
            for x in range(2, w - 2):
                # Sobel-like gradient with 3x3 neighborhood
                gx = (int(img[y-1, x+1]) + 2*int(img[y, x+1]) + int(img[y+1, x+1]) -
                      int(img[y-1, x-1]) - 2*int(img[y, x-1]) - int(img[y+1, x-1]))
                gy = (int(img[y+1, x-1]) + 2*int(img[y+1, x]) + int(img[y+1, x+1]) -
                      int(img[y-1, x-1]) - 2*int(img[y-1, x]) - int(img[y-1, x+1]))
                gradient = (abs(gx) + abs(gy)) // 4
                edges[y, x] = min(255, gradient)
        
        # Use a higher threshold for stronger edges (speech bubble outlines)
        edge_threshold = 50
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
        if img[start_y, start_x] < 180:
            # Starting point is too dark, not inside a typical speech bubble
            return None
        
        visited = np.zeros((h, w), dtype=bool)
        stack = [(start_x, start_y)]
        
        filled_count = 0
        # Be more conservative - don't fill more than 60% of search area
        max_fill = h * w * 0.6
        
        # Track if we hit the boundary of search region
        hit_boundary = 0
        max_boundary_hits = (h + w) * 0.3  # Allow some boundary touches but not too many
        
        while stack:
            x, y = stack.pop()
            
            if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
                # Hit the edge of search region
                hit_boundary += 1
                if hit_boundary > max_boundary_hits:
                    # Too many boundary hits - we're probably escaping the bubble
                    return None
                continue
            
            if visited[y, x]:
                continue
            
            visited[y, x] = True
            
            # Stop at edges (bubble boundary) - strong edges
            if edges[y, x] > 0:
                continue
            
            # Stop at dark pixels (likely text or outside bubble)
            # But be more lenient for slightly colored backgrounds
            if img[y, x] < 150:
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
    
    def _order_boundary_points(
        self,
        points: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Order boundary points to form a proper closed contour.
        
        Uses nearest-neighbor approach to trace the boundary.
        """
        if len(points) < 3:
            return points
        
        # Start from the topmost-leftmost point
        points = list(points)
        ordered = [min(points, key=lambda p: (p[1], p[0]))]
        points.remove(ordered[0])
        
        while points:
            current = ordered[-1]
            # Find nearest unvisited point
            nearest = min(points, key=lambda p: (p[0]-current[0])**2 + (p[1]-current[1])**2)
            ordered.append(nearest)
            points.remove(nearest)
        
        return ordered
    
    def _simplify_polygon(
        self,
        points: list[tuple[int, int]],
        tolerance: float = 2.0
    ) -> list[tuple[int, int]]:
        """Simplify a polygon using convex hull for cleaner shape.
        
        Args:
            points: List of polygon points
            tolerance: Distance tolerance for simplification
            
        Returns:
            Simplified polygon
        """
        if len(points) < 3:
            return points
        
        # Use convex hull for a clean, simple polygon
        hull = self._convex_hull(points)
        
        if len(hull) < 3:
            # Fallback: sort by angle
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)
            import math
            return sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
        
        return hull
    
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
        use_polygons: bool = True,
    ) -> Image.Image:
        """Render text onto the image within the detected text polygon.
        
        Args:
            image: Source image (should be inpainted first)
            blocks: Text blocks with translations
            use_translated: Whether to use translated_text (True) or original_text
            use_polygons: If True, fit text within polygon bounds
            
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
            
            # Try to get polygon for text fitting
            polygon = None
            if use_polygons:
                try:
                    polygon = self.extract_text_polygon(image, bbox)
                except Exception:
                    polygon = None
            
            # Calculate the text area from polygon or bbox
            if polygon and len(polygon) >= 3:
                # Get bounding box of polygon
                poly_xs = [p[0] for p in polygon]
                poly_ys = [p[1] for p in polygon]
                px1, py1 = min(poly_xs), min(poly_ys)
                px2, py2 = max(poly_xs), max(poly_ys)
                
                # Use polygon bounds with some inner padding
                box_width = px2 - px1
                box_height = py2 - py1
                padding = int(min(box_width, box_height) * 0.1)
                
                inner_x1 = px1 + padding
                inner_y1 = py1 + padding
                inner_width = box_width - (2 * padding)
                inner_height = box_height - (2 * padding)
            else:
                # Use bounding box
                box_width = x2 - x1
                box_height = y2 - y1
                padding = int(min(box_width, box_height) * self.config.padding_ratio)
                
                inner_x1 = x1 + padding
                inner_y1 = y1 + padding
                inner_width = box_width - (2 * padding)
                inner_height = box_height - (2 * padding)
            
            if inner_width <= 0 or inner_height <= 0:
                continue
            
            # Find optimal font size
            font, wrapped_text = self._fit_text(
                text, inner_width, inner_height, block.is_vertical
            )
            
            # Calculate text position (centered in the text area)
            text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            text_x = inner_x1 + (inner_width - text_width) // 2
            text_y = inner_y1 + (inner_height - text_height) // 2
            
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