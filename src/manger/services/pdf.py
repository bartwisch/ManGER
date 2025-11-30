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

from manger.config import PDFConfig, get_config

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
    
    def __init__(self, dpi: int = 200, config: PDFConfig | None = None):
        """Initialize the PDF service.
        
        Args:
            dpi: Resolution for rendering pages (default: 200)
            config: PDF configuration for output settings
        """
        self.dpi = dpi
        self.config = config or get_config().pdf
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
        quality: int | None = None,
        max_dimension: int | None = None,
    ) -> bytes:
        """Create a PDF from a list of PIL Images.
        
        Optimized for small file sizes while maintaining readability.
        Uses config values if parameters not specified.
        
        Args:
            images: List of PIL Images to combine into PDF
            output_path: Optional path to save the PDF file
            quality: JPEG quality (1-100), uses config if not specified
            max_dimension: Maximum width/height in pixels, uses config if not specified
            
        Returns:
            PDF file content as bytes
        """
        if not images:
            raise PDFError("No images provided")
        
        # Use config values if not explicitly provided
        quality = quality if quality is not None else self.config.jpeg_quality
        max_dimension = max_dimension if max_dimension is not None else self.config.max_dimension
        output_dpi = self.config.dpi
        
        try:
            # Create a new PDF
            doc = fitz.open()
            
            for img in images:
                # Aggressively resize to reduce file size
                # Manga is typically readable at 1400px max dimension
                if max_dimension and (img.width > max_dimension or img.height > max_dimension):
                    ratio = min(max_dimension / img.width, max_dimension / img.height)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    logger.debug(f"Resized image to {new_size[0]}x{new_size[1]}")
                
                # Convert to RGB if necessary (JPEG doesn't support RGBA)
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Create white background for transparent images
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Convert PIL Image to JPEG bytes with aggressive compression
                img_bytes = io.BytesIO()
                img.save(
                    img_bytes, 
                    format="JPEG", 
                    quality=quality, 
                    optimize=True,
                    progressive=True,  # Progressive JPEG for better compression
                    subsampling=2,  # 4:2:0 chroma subsampling for smaller size
                )
                img_bytes.seek(0)
                
                # Create a new page with the image dimensions
                # PyMuPDF uses points (72 per inch), so we convert pixels
                width_pts = img.width * 72 / output_dpi
                height_pts = img.height * 72 / output_dpi
                
                page = doc.new_page(width=width_pts, height=height_pts)
                
                # Insert the image
                rect = fitz.Rect(0, 0, width_pts, height_pts)
                page.insert_image(rect, stream=img_bytes.getvalue())
            
            # Get PDF as bytes with maximum compression
            # garbage=4: remove unused objects, compact xref, merge duplicates, clean pages
            # deflate=True: compress streams
            # clean=True: clean and sanitize content streams
            pdf_bytes = doc.tobytes(
                deflate=True, 
                garbage=4,
                clean=True,
            )
            
            # Optionally save to file
            if output_path:
                with open(output_path, "wb") as f:
                    f.write(pdf_bytes)
                logger.info(f"Saved PDF to {output_path}")
            
            doc.close()
            logger.info(f"Created PDF with {len(images)} pages ({len(pdf_bytes) / 1024 / 1024:.1f} MB)")
            return pdf_bytes
            
        except Exception as e:
            raise PDFError(f"Failed to create PDF: {e}") from e

    def merge_pdfs(self, pdf_files: list[bytes]) -> bytes:
        """Merge multiple PDF files into one.
        
        Args:
            pdf_files: List of PDF file contents as bytes
            
        Returns:
            Merged PDF content as bytes
        """
        if not pdf_files:
            raise PDFError("No PDF files provided")
            
        try:
            # Create a new empty PDF
            merged_doc = fitz.open()
            
            for pdf_bytes in pdf_files:
                # Open each PDF from bytes
                with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                    merged_doc.insert_pdf(doc)
            
            # Return merged PDF as bytes
            return merged_doc.tobytes()
            
        except Exception as e:
            raise PDFError(f"Failed to merge PDFs: {e}") from e
        finally:
            if 'merged_doc' in locals():
                merged_doc.close()

    def compress_pdf(self, pdf_bytes: bytes) -> bytes:
        """Compress a PDF file (structure only, not images).
        
        Args:
            pdf_bytes: PDF file content as bytes
            
        Returns:
            Compressed PDF content as bytes
        """
        if not pdf_bytes:
            raise PDFError("No PDF content provided")
            
        try:
            # Open PDF from bytes
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                # Save with compression options
                # garbage=4: remove unused objects, compact xref, merge duplicate objects
                # deflate=True: compress streams
                return doc.tobytes(garbage=4, deflate=True)
                
        except Exception as e:
            raise PDFError(f"Failed to compress PDF: {e}") from e

    def shrink_pdf(
        self,
        pdf_bytes: bytes,
        quality: int | None = None,
        max_dimension: int | None = None,
        output_path: str | None = None,
    ) -> bytes:
        """Shrink a PDF by re-compressing all images.
        
        This is useful for reducing the size of existing large PDFs.
        Each page is extracted as an image and re-compressed with the specified settings.
        
        Args:
            pdf_bytes: PDF file content as bytes
            quality: JPEG quality (1-100), uses config if not specified
            max_dimension: Maximum image dimension in pixels, uses config if not specified
            output_path: Optional path to save the shrunk PDF
            
        Returns:
            Shrunk PDF content as bytes
        """
        if not pdf_bytes:
            raise PDFError("No PDF content provided")
        
        # Use config values if not explicitly provided
        quality = quality if quality is not None else self.config.jpeg_quality
        max_dimension = max_dimension if max_dimension is not None else self.config.max_dimension
        output_dpi = self.config.dpi
        
        original_size = len(pdf_bytes) / 1024 / 1024  # MB
        logger.info(f"Shrinking PDF: {original_size:.1f} MB, quality={quality}, max_dim={max_dimension}")
        
        try:
            # Open the source PDF
            src_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            # Extract all pages as images and re-compress
            images = []
            for page_num in range(len(src_doc)):
                page = src_doc[page_num]
                
                # Render page to image at reasonable DPI
                # Use 150 DPI for extraction (good balance of quality/size)
                zoom = 150 / 72.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
                
                logger.debug(f"Extracted page {page_num + 1}/{len(src_doc)}")
            
            src_doc.close()
            
            # Create new PDF with compressed images
            result = self.create_pdf_from_images(
                images,
                output_path=output_path,
                quality=quality,
                max_dimension=max_dimension,
            )
            
            new_size = len(result) / 1024 / 1024  # MB
            reduction = (1 - new_size / original_size) * 100
            logger.info(f"PDF shrunk: {original_size:.1f} MB → {new_size:.1f} MB ({reduction:.0f}% reduction)")
            
            return result
            
        except Exception as e:
            raise PDFError(f"Failed to shrink PDF: {e}") from e

    def __enter__(self) -> PDFService:
        """Context manager entry."""
        return self
    
    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.close()
    
    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()