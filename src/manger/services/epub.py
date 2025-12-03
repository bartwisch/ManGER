"""EPUB Service for creating light novel ebooks.

EPUB (Electronic Publication) is the standard format for ebooks.
It's essentially a ZIP file containing:
- XHTML content files (chapters)
- CSS stylesheets
- Images
- Metadata (OPF package file)
- Navigation (NCX/NAV files)

This service creates EPUB 3.0 compatible files.
"""

from __future__ import annotations

import io
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree as ET

from loguru import logger
from PIL import Image


class EPUBError(Exception):
    """Base exception for EPUB-related errors."""

    pass


class Chapter:
    """Represents a chapter in the EPUB."""

    def __init__(
        self,
        title: str,
        content: str,
        chapter_id: str | None = None,
    ):
        """Initialize a chapter.

        Args:
            title: Chapter title
            content: Chapter content (plain text or HTML)
            chapter_id: Optional unique identifier
        """
        self.title = title
        self.content = content
        self.chapter_id = chapter_id or f"chapter_{uuid.uuid4().hex[:8]}"


class EPUBService:
    """Service for creating EPUB ebooks.

    Creates EPUB 3.0 compatible files suitable for light novels
    and other text-based content.
    """

    # EPUB namespace constants
    CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"
    OPF_NS = "http://www.idpf.org/2007/opf"
    DC_NS = "http://purl.org/dc/elements/1.1/"
    XHTML_NS = "http://www.w3.org/1999/xhtml"
    EPUB_NS = "http://www.idpf.org/2007/ops"

    def __init__(self):
        """Initialize the EPUB service."""
        pass

    def create_epub(
        self,
        title: str,
        author: str,
        chapters: list[Chapter],
        cover_image: Image.Image | None = None,
        language: str = "en",
        publisher: str = "",
        description: str = "",
        output_path: str | None = None,
    ) -> bytes:
        """Create an EPUB file from chapters.

        Args:
            title: Book title
            author: Author name
            chapters: List of Chapter objects
            cover_image: Optional cover image (PIL Image)
            language: Language code (e.g., "en", "de", "ja")
            publisher: Publisher name
            description: Book description
            output_path: Optional path to save the EPUB file

        Returns:
            EPUB file content as bytes
        """
        if not chapters:
            raise EPUBError("No chapters provided")

        book_id = f"urn:uuid:{uuid.uuid4()}"

        try:
            epub_buffer = io.BytesIO()

            with zipfile.ZipFile(epub_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                # 1. Add mimetype (must be first and uncompressed)
                zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)

                # 2. Add container.xml
                zf.writestr("META-INF/container.xml", self._create_container_xml())

                # 3. Add cover image if provided
                cover_filename = None
                if cover_image:
                    cover_filename = "OEBPS/images/cover.jpg"
                    cover_bytes = self._image_to_bytes(cover_image)
                    zf.writestr(cover_filename, cover_bytes)

                # 4. Add stylesheet
                zf.writestr("OEBPS/styles/style.css", self._create_stylesheet())

                # 5. Add chapter XHTML files
                chapter_files = []
                for idx, chapter in enumerate(chapters):
                    filename = f"OEBPS/text/chapter_{idx + 1:03d}.xhtml"
                    xhtml_content = self._create_chapter_xhtml(chapter)
                    zf.writestr(filename, xhtml_content)
                    chapter_files.append((filename, chapter))

                # 6. Add cover page if we have a cover image
                if cover_image:
                    cover_xhtml = self._create_cover_xhtml()
                    zf.writestr("OEBPS/text/cover.xhtml", cover_xhtml)

                # 7. Add title page
                title_xhtml = self._create_title_page_xhtml(title, author)
                zf.writestr("OEBPS/text/title.xhtml", title_xhtml)

                # 8. Add navigation document (EPUB 3)
                nav_xhtml = self._create_nav_xhtml(chapters, has_cover=cover_image is not None)
                zf.writestr("OEBPS/text/nav.xhtml", nav_xhtml)

                # 9. Add OPF package document
                opf_content = self._create_opf(
                    title=title,
                    author=author,
                    book_id=book_id,
                    language=language,
                    publisher=publisher,
                    description=description,
                    chapters=chapters,
                    has_cover=cover_image is not None,
                )
                zf.writestr("OEBPS/content.opf", opf_content)

            epub_bytes = epub_buffer.getvalue()

            if output_path:
                with open(output_path, "wb") as f:
                    f.write(epub_bytes)
                logger.info(f"Saved EPUB to {output_path}")

            logger.info(
                f"Created EPUB '{title}' with {len(chapters)} chapters ({len(epub_bytes) / 1024:.1f} KB)"
            )
            return epub_bytes

        except Exception as e:
            raise EPUBError(f"Failed to create EPUB: {e}") from e

    def create_epub_from_text(
        self,
        title: str,
        author: str,
        text: str,
        cover_image: Image.Image | None = None,
        language: str = "en",
        chapter_separator: str = "\n\n---\n\n",
        output_path: str | None = None,
    ) -> bytes:
        """Create an EPUB from a single text string, splitting into chapters.

        Args:
            title: Book title
            author: Author name
            text: Full text content
            cover_image: Optional cover image
            language: Language code
            chapter_separator: String used to split text into chapters
            output_path: Optional path to save the EPUB

        Returns:
            EPUB file content as bytes
        """
        # Split text into chapters
        parts = text.split(chapter_separator)
        chapters = []

        for idx, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue

            # Try to extract chapter title from first line
            lines = part.split("\n", 1)
            if len(lines) > 1 and len(lines[0]) < 100:
                chapter_title = lines[0].strip()
                chapter_content = lines[1].strip()
            else:
                chapter_title = f"Chapter {idx + 1}"
                chapter_content = part

            chapters.append(Chapter(title=chapter_title, content=chapter_content))

        if not chapters:
            # No separators found, treat entire text as one chapter
            chapters = [Chapter(title="Chapter 1", content=text)]

        return self.create_epub(
            title=title,
            author=author,
            chapters=chapters,
            cover_image=cover_image,
            language=language,
            output_path=output_path,
        )

    def _create_container_xml(self) -> str:
        """Create the META-INF/container.xml file."""
        return """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""

    def _create_opf(
        self,
        title: str,
        author: str,
        book_id: str,
        language: str,
        publisher: str,
        description: str,
        chapters: list[Chapter],
        has_cover: bool,
    ) -> str:
        """Create the OPF package document."""
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # Build manifest items
        manifest_items = [
            '    <item id="nav" href="text/nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
            '    <item id="style" href="styles/style.css" media-type="text/css"/>',
            '    <item id="title" href="text/title.xhtml" media-type="application/xhtml+xml"/>',
        ]

        if has_cover:
            manifest_items.append(
                '    <item id="cover-image" href="images/cover.jpg" media-type="image/jpeg" properties="cover-image"/>'
            )
            manifest_items.append(
                '    <item id="cover" href="text/cover.xhtml" media-type="application/xhtml+xml"/>'
            )

        for idx, chapter in enumerate(chapters):
            manifest_items.append(
                f'    <item id="chapter{idx + 1}" href="text/chapter_{idx + 1:03d}.xhtml" media-type="application/xhtml+xml"/>'
            )

        # Build spine items
        spine_items = []
        if has_cover:
            spine_items.append('    <itemref idref="cover"/>')
        spine_items.append('    <itemref idref="title"/>')
        for idx in range(len(chapters)):
            spine_items.append(f'    <itemref idref="chapter{idx + 1}"/>')

        opf = f"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="BookId">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="BookId">{self._escape_xml(book_id)}</dc:identifier>
    <dc:title>{self._escape_xml(title)}</dc:title>
    <dc:creator>{self._escape_xml(author)}</dc:creator>
    <dc:language>{self._escape_xml(language)}</dc:language>
    <dc:publisher>{self._escape_xml(publisher)}</dc:publisher>
    <dc:description>{self._escape_xml(description)}</dc:description>
    <meta property="dcterms:modified">{now}</meta>
  </metadata>
  <manifest>
{chr(10).join(manifest_items)}
  </manifest>
  <spine>
{chr(10).join(spine_items)}
  </spine>
</package>"""

        return opf

    def _create_nav_xhtml(self, chapters: list[Chapter], has_cover: bool) -> str:
        """Create the navigation document (EPUB 3 nav.xhtml)."""
        nav_items = []
        if has_cover:
            nav_items.append('        <li><a href="cover.xhtml">Cover</a></li>')
        nav_items.append('        <li><a href="title.xhtml">Title Page</a></li>')

        for idx, chapter in enumerate(chapters):
            nav_items.append(
                f'        <li><a href="chapter_{idx + 1:03d}.xhtml">{self._escape_xml(chapter.title)}</a></li>'
            )

        return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Table of Contents</title>
  <link rel="stylesheet" type="text/css" href="../styles/style.css"/>
</head>
<body>
  <nav epub:type="toc" id="toc">
    <h1>Table of Contents</h1>
    <ol>
{chr(10).join(nav_items)}
    </ol>
  </nav>
</body>
</html>"""

    def _create_chapter_xhtml(self, chapter: Chapter) -> str:
        """Create an XHTML file for a chapter."""
        # Convert plain text to HTML paragraphs
        content = chapter.content

        # Check if content is already HTML
        if not content.strip().startswith("<"):
            # Convert plain text to paragraphs
            paragraphs = content.split("\n\n")
            html_paragraphs = []
            for p in paragraphs:
                p = p.strip()
                if p:
                    # Escape HTML entities and wrap in <p>
                    p = self._escape_xml(p)
                    # Convert single newlines to <br/>
                    p = p.replace("\n", "<br/>\n")
                    html_paragraphs.append(f"  <p>{p}</p>")
            content = "\n".join(html_paragraphs)

        return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>{self._escape_xml(chapter.title)}</title>
  <link rel="stylesheet" type="text/css" href="../styles/style.css"/>
</head>
<body>
  <h1>{self._escape_xml(chapter.title)}</h1>
{content}
</body>
</html>"""

    def _create_cover_xhtml(self) -> str:
        """Create the cover page XHTML."""
        return """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Cover</title>
  <link rel="stylesheet" type="text/css" href="../styles/style.css"/>
</head>
<body class="cover">
  <img src="../images/cover.jpg" alt="Cover"/>
</body>
</html>"""

    def _create_title_page_xhtml(self, title: str, author: str) -> str:
        """Create the title page XHTML."""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>{self._escape_xml(title)}</title>
  <link rel="stylesheet" type="text/css" href="../styles/style.css"/>
</head>
<body class="title-page">
  <div class="title-content">
    <h1 class="book-title">{self._escape_xml(title)}</h1>
    <p class="book-author">by {self._escape_xml(author)}</p>
  </div>
</body>
</html>"""

    def _create_stylesheet(self) -> str:
        """Create the CSS stylesheet."""
        return """/* EPUB Stylesheet */

body {
  font-family: Georgia, "Times New Roman", serif;
  line-height: 1.6;
  margin: 1em;
  text-align: justify;
}

h1 {
  font-size: 1.5em;
  margin-bottom: 1em;
  text-align: center;
}

p {
  margin: 0.5em 0;
  text-indent: 1.5em;
}

p:first-of-type {
  text-indent: 0;
}

/* Cover page */
body.cover {
  margin: 0;
  padding: 0;
  text-align: center;
}

body.cover img {
  max-width: 100%;
  max-height: 100vh;
  object-fit: contain;
}

/* Title page */
body.title-page {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  text-align: center;
}

.title-content {
  padding: 2em;
}

.book-title {
  font-size: 2em;
  margin-bottom: 0.5em;
}

.book-author {
  font-size: 1.2em;
  font-style: italic;
  text-indent: 0;
}

/* Navigation */
nav ol {
  list-style-type: none;
  padding-left: 0;
}

nav li {
  margin: 0.5em 0;
}

nav a {
  color: #333;
  text-decoration: none;
}

nav a:hover {
  text-decoration: underline;
}
"""

    def _image_to_bytes(self, img: Image.Image, quality: int = 85) -> bytes:
        """Convert a PIL Image to JPEG bytes."""
        # Convert to RGB if necessary
        if img.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            if img.mode in ("RGBA", "LA"):
                background.paste(img, mask=img.split()[-1])
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        return buffer.getvalue()

    def _escape_xml(self, text: str) -> str:
        """Escape special XML characters."""
        if not text:
            return ""
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace('"', "&quot;")
        text = text.replace("'", "&apos;")
        return text


def text_to_epub(
    title: str,
    author: str,
    text: str,
    language: str = "en",
) -> bytes:
    """Convenience function to create an EPUB from text.

    Args:
        title: Book title
        author: Author name
        text: Full text content
        language: Language code

    Returns:
        EPUB file as bytes
    """
    service = EPUBService()
    return service.create_epub_from_text(
        title=title,
        author=author,
        text=text,
        language=language,
    )
