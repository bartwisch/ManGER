"""Comic Book Archive Service for handling CBZ/CBR files.

Provides functionality to:
- Create CBZ (ZIP-based) comic archives
- Create CBR (RAR-based) comic archives (requires rarfile)
- Extract images from CBZ/CBR files

CBZ (Comic Book Zip) and CBR (Comic Book Rar) are standard formats for
digital comic books and manga. They are simply renamed archive files
containing image files (JPG, PNG, WebP) for each page.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Literal

from loguru import logger
from PIL import Image


class ArchiveError(Exception):
    """Base exception for archive-related errors."""

    pass


class ArchiveService:
    """Service for creating and handling comic book archives (CBZ/CBR).

    CBZ is preferred as it uses ZIP compression which is universally supported.
    CBR uses RAR which requires additional software to create.
    """

    def __init__(
        self,
        image_format: Literal["JPEG", "WEBP", "PNG"] = "JPEG",
        quality: int = 85,
    ):
        """Initialize the archive service.

        Args:
            image_format: Format for images in the archive (JPEG, WEBP, PNG)
            quality: Quality for lossy formats (1-100, default 85)
        """
        self.image_format = image_format
        self.quality = quality

    def create_cbz(
        self,
        images: list[Image.Image],
        output_path: str | None = None,
        image_format: Literal["JPEG", "WEBP", "PNG"] | None = None,
        quality: int | None = None,
        filename_prefix: str = "page",
    ) -> bytes:
        """Create a CBZ archive from a list of PIL Images.

        CBZ is a ZIP file containing images, typically named sequentially.
        Most comic readers expect files to be sorted alphabetically.

        Args:
            images: List of PIL Images to include in the archive
            output_path: Optional path to save the CBZ file
            image_format: Override default image format
            quality: Override default quality
            filename_prefix: Prefix for page filenames (default: "page")

        Returns:
            CBZ file content as bytes
        """
        if not images:
            raise ArchiveError("No images provided")

        fmt = image_format or self.image_format
        qual = quality if quality is not None else self.quality

        # Determine file extension
        ext_map = {"JPEG": "jpg", "WEBP": "webp", "PNG": "png"}
        ext = ext_map.get(fmt, "jpg")

        try:
            # Create ZIP in memory
            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_STORED) as zf:
                # ZIP_STORED = no compression (images are already compressed)
                # This is faster and doesn't bloat the file

                for idx, img in enumerate(images):
                    # Generate filename with zero-padded index for correct sorting
                    # e.g., page_001.jpg, page_002.jpg, ...
                    page_num = str(idx + 1).zfill(3)
                    filename = f"{filename_prefix}_{page_num}.{ext}"

                    # Convert image to bytes
                    img_bytes = self._image_to_bytes(img, fmt, qual)

                    # Add to archive
                    zf.writestr(filename, img_bytes)
                    logger.debug(f"Added {filename} ({len(img_bytes) / 1024:.1f} KB)")

            # Get the complete ZIP data
            cbz_bytes = zip_buffer.getvalue()

            # Optionally save to file
            if output_path:
                with open(output_path, "wb") as f:
                    f.write(cbz_bytes)
                logger.info(f"Saved CBZ to {output_path}")

            logger.info(
                f"Created CBZ with {len(images)} pages ({len(cbz_bytes) / 1024 / 1024:.2f} MB)"
            )
            return cbz_bytes

        except Exception as e:
            raise ArchiveError(f"Failed to create CBZ: {e}") from e

    def create_cbr(
        self,
        images: list[Image.Image],
        output_path: str | None = None,
        image_format: Literal["JPEG", "WEBP", "PNG"] | None = None,
        quality: int | None = None,
        filename_prefix: str = "page",
    ) -> bytes:
        """Create a CBR archive from a list of PIL Images.

        CBR is a RAR file containing images. Note that creating RAR files
        requires the 'rarfile' package and the 'rar' command-line tool.

        For compatibility, CBZ (ZIP-based) is generally preferred.
        This method falls back to CBZ if RAR creation is not available.

        Args:
            images: List of PIL Images to include in the archive
            output_path: Optional path to save the CBR file
            image_format: Override default image format
            quality: Override default quality
            filename_prefix: Prefix for page filenames

        Returns:
            CBR file content as bytes
        """
        # RAR creation requires proprietary tools, so we'll use ZIP internally
        # but with .cbr extension for compatibility
        logger.warning(
            "CBR creation uses ZIP internally (RAR requires proprietary tools). "
            "The file will work with most readers but is technically a ZIP file."
        )

        return self.create_cbz(
            images=images,
            output_path=output_path,
            image_format=image_format,
            quality=quality,
            filename_prefix=filename_prefix,
        )

    def extract_cbz(self, cbz_data: bytes | str | Path) -> list[Image.Image]:
        """Extract images from a CBZ archive.

        Args:
            cbz_data: CBZ file as bytes, path string, or Path object

        Returns:
            List of PIL Images in page order
        """
        try:
            if isinstance(cbz_data, (str, Path)):
                with open(cbz_data, "rb") as f:
                    cbz_data = f.read()

            images = []
            zip_buffer = io.BytesIO(cbz_data)

            with zipfile.ZipFile(zip_buffer, "r") as zf:
                # Get list of files, sorted for correct page order
                filenames = sorted(
                    [
                        name
                        for name in zf.namelist()
                        if self._is_image_file(name) and not name.startswith("__MACOSX")
                    ]
                )

                for filename in filenames:
                    with zf.open(filename) as img_file:
                        img = Image.open(io.BytesIO(img_file.read()))
                        img.load()  # Ensure image is fully loaded
                        images.append(img)
                        logger.debug(f"Extracted {filename}")

            logger.info(f"Extracted {len(images)} images from CBZ")
            return images

        except Exception as e:
            raise ArchiveError(f"Failed to extract CBZ: {e}") from e

    def extract_cbr(self, cbr_data: bytes | str | Path) -> list[Image.Image]:
        """Extract images from a CBR archive.

        Args:
            cbr_data: CBR file as bytes, path string, or Path object

        Returns:
            List of PIL Images in page order
        """
        # Try to extract as RAR first, fall back to ZIP
        try:
            import rarfile

            if isinstance(cbr_data, (str, Path)):
                rf = rarfile.RarFile(str(cbr_data))
            else:
                # rarfile doesn't support BytesIO directly, need temp file
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".cbr", delete=False) as tmp:
                    tmp.write(cbr_data)
                    tmp_path = tmp.name
                rf = rarfile.RarFile(tmp_path)

            images = []
            filenames = sorted([name for name in rf.namelist() if self._is_image_file(name)])

            for filename in filenames:
                with rf.open(filename) as img_file:
                    img = Image.open(io.BytesIO(img_file.read()))
                    img.load()
                    images.append(img)

            rf.close()
            logger.info(f"Extracted {len(images)} images from CBR (RAR)")
            return images

        except ImportError:
            logger.warning("rarfile not installed, trying ZIP extraction")
            return self.extract_cbz(cbr_data)
        except Exception:
            # If RAR extraction fails, try ZIP (our CBR files are ZIP internally)
            logger.warning("RAR extraction failed, trying ZIP extraction")
            return self.extract_cbz(cbr_data)

    def _image_to_bytes(
        self,
        img: Image.Image,
        fmt: str,
        quality: int,
    ) -> bytes:
        """Convert a PIL Image to bytes in the specified format.

        Args:
            img: PIL Image to convert
            fmt: Image format (JPEG, WEBP, PNG)
            quality: Quality for lossy formats

        Returns:
            Image data as bytes
        """
        # Convert to RGB if necessary (JPEG doesn't support RGBA)
        if fmt == "JPEG" and img.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            if img.mode in ("RGBA", "LA"):
                background.paste(img, mask=img.split()[-1])
            img = background
        elif fmt == "JPEG" and img.mode != "RGB":
            img = img.convert("RGB")

        img_buffer = io.BytesIO()

        if fmt == "JPEG":
            img.save(
                img_buffer,
                format="JPEG",
                quality=quality,
                optimize=True,
                progressive=True,
            )
        elif fmt == "WEBP":
            img.save(
                img_buffer,
                format="WEBP",
                quality=quality,
                method=4,  # Compression method (0-6, higher = slower but smaller)
            )
        elif fmt == "PNG":
            img.save(
                img_buffer,
                format="PNG",
                optimize=True,
            )
        else:
            raise ArchiveError(f"Unsupported image format: {fmt}")

        return img_buffer.getvalue()

    def _is_image_file(self, filename: str) -> bool:
        """Check if a filename is a supported image format."""
        lower = filename.lower()
        return any(
            lower.endswith(ext)
            for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"]
        )


def images_to_cbz(
    images: list[Image.Image],
    image_format: Literal["JPEG", "WEBP", "PNG"] = "JPEG",
    quality: int = 85,
) -> bytes:
    """Convenience function to create a CBZ from images.

    Args:
        images: List of PIL Images
        image_format: Image format to use
        quality: Quality for lossy formats

    Returns:
        CBZ file as bytes
    """
    service = ArchiveService(image_format=image_format, quality=quality)
    return service.create_cbz(images)
