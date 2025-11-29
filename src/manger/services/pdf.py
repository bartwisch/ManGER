"""PDF Service for handling PDF manga files.

Provides functionality to:
- Load PDF files
- Extract page count and metadata
- Convert PDF pages to images
- Create PDF from images
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO
import tempfile
import io

from loguru import logger
from PIL import Image
import fitz

if TYPE_CHECKING:
    pass


class PDFError(Exception):
    """Base exception for PDF-related errors."""
    pass


class PDFLoadError(PDFError):
    """Raised when PDF loading fails."""
    pass


class PDFPageError(PDFError):
    """Raised when page extraction fails."""
    pass


class PDFService:
    """Service for handling PDF manga files.
    
    Uses PyMuPDF (fitz) for PDF processing.
    """
    
    def __init__(self, dpi: int = 200):
        """Initialize the PDF service.
        
        Args:
            dpi: Resolution for rendering pages (default: 200)
        """
        self.dpi = dpi
        self._doc = None
        self._file_path: str | None = None
    
    def load(self, file_path: str | Path | BinaryIO) -> int:
        """Load a PDF file.
        
        Args:
            file_path: Path to PDF file or file-like object
            
        Returns:
            Number of pages in the PDF
            
        Raises:
            PDFLoadError: If loading fails
        """
        try:
            import fitz
        except ImportError:
            raise PDFLoadError(
                "PyMuPDF not installed. Install with: pip install pymupdf"
            )
        
        try:
            if hasattr(file_path, 'read'):
                # It's a file-like object
                data = file_path.read()
                self._doc = fitz.open(stream=data, filetype="pdf")
                self._file_path = "<uploaded>"
            else:
                self._doc = fitz.open(str(file_path))
                self._file_path = str(file_path)
            
            page_count = len(self._doc)
            logger.info(f"Loaded PDF with {page_count} pages: {self._file_path}")
            return page_count
            
        except Exception as e:
            raise PDFLoadError(f"Failed to load PDF: {e}") from e
    
    def load_from_bytes(self, data: bytes) -> int:
        """Load a PDF from bytes.
        
        Args:
            data: PDF file content as bytes
            
        Returns:
            Number of pages in the PDF
            
        Raises:
            PDFLoadError: If loading fails
        """
        try:
            import fitz
        except ImportError:
            raise PDFLoadError(
                "PyMuPDF not installed. Install with: pip install pymupdf"
            )
        
        try:
            self._doc = fitz.open(stream=data, filetype="pdf")
            self._file_path = "<bytes>"
            
            page_count = len(self._doc)
            logger.info(f"Loaded PDF from bytes with {page_count} pages")
            return page_count
            
        except Exception as e:
            raise PDFLoadError(f"Failed to load PDF from bytes: {e}") from e
    
    @property
    def page_count(self) -> int:
        """Get the number of pages in the loaded PDF."""
        if self._doc is None:
            return 0
        return len(self._doc)
    
    @property
    def is_loaded(self) -> bool:
        """Check if a PDF is currently loaded."""
        return self._doc is not None
    
    def get_page_info(self, page_number: int) -> dict:
        """Get information about a specific page.
        
        Args:
            page_number: Page number (0-indexed)
            
        Returns:
            Dictionary with page width, height, and rotation
        """
        if self._doc is None:
            raise PDFError("No PDF loaded")
        
        if page_number < 0 or page_number >= len(self._doc):
            raise PDFPageError(f"Page {page_number} out of range")
        
        page = self._doc[page_number]
        rect = page.rect
        
        return {
            "page_number": page_number,
            "width": rect.width,
            "height": rect.height,
            "rotation": page.rotation,
        }
    
    def get_page_image(self, page_number: int, dpi: int | None = None) -> Image.Image:
        """Extract a page as a PIL Image.
        
        Args:
            page_number: Page number (0-indexed)
            dpi: Optional DPI override (uses instance default if not specified)
            
        Returns:
            PIL Image of the page
            
        Raises:
            PDFPageError: If extraction fails
        """
        if self._doc is None:
            raise PDFError("No PDF loaded")
        
        if page_number < 0 or page_number >= len(self._doc):
            raise PDFPageError(f"Page {page_number} out of range (0-{len(self._doc)-1})")
        
        dpi = dpi or self.dpi
        
        try:
            page = self._doc[page_number]
            
            # Calculate zoom factor for desired DPI
            # Default PDF resolution is 72 DPI
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            logger.debug(
                f"Extracted page {page_number}: {img.size[0]}x{img.size[1]} at {dpi} DPI"
            )
            return img
            
        except Exception as e:
            raise PDFPageError(f"Failed to extract page {page_number}: {e}") from e
    
    def get_pages_images(
        self,
        page_numbers: list[int] | None = None,
        dpi: int | None = None,
    ) -> list[tuple[int, Image.Image]]:
        """Extract multiple pages as PIL Images.
        
        Args:
            page_numbers: List of page numbers (0-indexed). If None, extracts all pages.
            dpi: Optional DPI override
            
        Returns:
            List of (page_number, Image) tuples
        """
        if self._doc is None:
            raise PDFError("No PDF loaded")
        
        if page_numbers is None:
            page_numbers = list(range(len(self._doc)))
        
        results = []
        for page_num in page_numbers:
            try:
                img = self.get_page_image(page_num, dpi)
                results.append((page_num, img))
            except PDFPageError as e:
                logger.warning(f"Skipping page {page_num}: {e}")
        
        return results
    
    def get_thumbnail(
        self,
        page_number: int,
        max_size: tuple[int, int] = (200, 200),
    ) -> Image.Image:
        """Get a thumbnail of a page.
        
        Args:
            page_number: Page number (0-indexed)
            max_size: Maximum thumbnail size (width, height)
            
        Returns:
            Thumbnail PIL Image
        """
        # Use low DPI for thumbnails
        img = self.get_page_image(page_number, dpi=72)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        return img
    
    def get_all_thumbnails(
        self,
        max_size: tuple[int, int] = (150, 150),
    ) -> list[tuple[int, Image.Image]]:
        """Get thumbnails for all pages.
        
        Args:
            max_size: Maximum thumbnail size
            
        Returns:
            List of (page_number, thumbnail) tuples
        """
        if self._doc is None:
            return []
        
        thumbnails = []
        for i in range(len(self._doc)):
            try:
                thumb = self.get_thumbnail(i, max_size)
                thumbnails.append((i, thumb))
            except PDFPageError as e:
                logger.warning(f"Failed to create thumbnail for page {i}: {e}")
        
        return thumbnails
    
    def close(self) -> None:
        """Close the current PDF document."""
        if self._doc is not None:
            self._doc.close()
            self._doc = None
            self._file_path = None
            logger.debug("PDF document closed")
    
    def create_pdf_from_images(
        self,
        images: list[Image.Image],
        output_path: str | None = None,
    ) -> bytes:
        """Create a PDF from a list of PIL Images.
        
        Args:
            images: List of PIL Images to combine into PDF
            output_path: Optional path to save the PDF file
            
        Returns:
            PDF file content as bytes
        """
        if not images:
            raise PDFError("No images provided")
        
        try:
            # Create a new PDF
            doc = fitz.open()
            
            for img in images:
                # Convert PIL Image to bytes
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="PNG")
                img_bytes.seek(0)
                
                # Create a new page with the image dimensions
                # PyMuPDF uses points (72 per inch), so we convert pixels
                width_pts = img.width * 72 / 150  # Assume 150 DPI for sizing
                height_pts = img.height * 72 / 150
                
                page = doc.new_page(width=width_pts, height=height_pts)
                
                # Insert the image
                rect = fitz.Rect(0, 0, width_pts, height_pts)
                page.insert_image(rect, stream=img_bytes.getvalue())
            
            # Get PDF as bytes
            pdf_bytes = doc.tobytes()
            
            # Optionally save to file
            if output_path:
                with open(output_path, "wb") as f:
                    f.write(pdf_bytes)
                logger.info(f"Saved PDF to {output_path}")
            
            doc.close()
            logger.info(f"Created PDF with {len(images)} pages")
            return pdf_bytes
            
        except Exception as e:
            raise PDFError(f"Failed to create PDF: {e}") from e

    def __enter__(self) -> PDFService:
        """Context manager entry."""
        return self
    
    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.close()
    
    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()