import streamlit as st
import sys
import re
import io
import time
from pathlib import Path

from PIL import Image

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from manger.services.pdf import PDFService, PDFError
from manger.services.archive import ArchiveService
from manger.services.epub import EPUBService, Chapter

st.set_page_config(
    page_title="Tools - ManGER",
    page_icon="🛠️",
    layout="wide",
)

st.title("🛠️ Tools")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "📄 PDF Combiner",
        "📉 PDF Shrinker",
        "🌐 Web Manga to PDF",
        "📚 CBZ Creator",
        "📖 EPUB Creator",
        "🔄 Format Converter",
        "👁️ CBZ Reader",
    ]
)

with tab1:
    st.header("PDF Combiner")
    st.markdown(
        "Upload multiple PDF files to combine them into a single document. **Drag to reorder!**"
    )

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Select multiple files to merge.",
        key="pdf_combiner_uploader",
    )

    if uploaded_files:
        # Initialize file order in session state
        if "pdf_file_order" not in st.session_state:
            st.session_state.pdf_file_order = list(range(len(uploaded_files)))

        # Reset order if number of files changed
        if len(st.session_state.pdf_file_order) != len(uploaded_files):
            st.session_state.pdf_file_order = list(range(len(uploaded_files)))

        st.subheader(f"📋 Arrange Order ({len(uploaded_files)} files)")
        st.caption("Use the buttons to move files up or down in the merge order.")

        # Store file data to preserve across reorders
        if "pdf_file_data" not in st.session_state or len(
            st.session_state.get("pdf_file_data", [])
        ) != len(uploaded_files):
            st.session_state.pdf_file_data = []
            for f in uploaded_files:
                f.seek(0)
                st.session_state.pdf_file_data.append(
                    {
                        "name": f.name,
                        "data": f.read(),
                        "size": f.size,
                    }
                )

        # Display files in current order with move buttons
        order = st.session_state.pdf_file_order

        for display_idx, file_idx in enumerate(order):
            file_info = st.session_state.pdf_file_data[file_idx]
            size_mb = file_info["size"] / (1024 * 1024)

            col1, col2, col3, col4 = st.columns([0.5, 4, 1, 1])

            with col1:
                st.write(f"**{display_idx + 1}.**")

            with col2:
                st.write(f"📄 {file_info['name']} ({size_mb:.1f} MB)")

            with col3:
                if display_idx > 0:
                    if st.button("⬆️", key=f"up_{file_idx}", help="Move up"):
                        # Swap with previous
                        order[display_idx], order[display_idx - 1] = (
                            order[display_idx - 1],
                            order[display_idx],
                        )
                        st.rerun()
                else:
                    st.write("")  # Empty placeholder

            with col4:
                if display_idx < len(order) - 1:
                    if st.button("⬇️", key=f"down_{file_idx}", help="Move down"):
                        # Swap with next
                        order[display_idx], order[display_idx + 1] = (
                            order[display_idx + 1],
                            order[display_idx],
                        )
                        st.rerun()
                else:
                    st.write("")  # Empty placeholder

        st.divider()

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔄 Reset Order", use_container_width=True):
                st.session_state.pdf_file_order = list(range(len(uploaded_files)))
                st.rerun()

        with col2:
            if st.button("📑 Merge PDFs", type="primary", use_container_width=True):
                with st.spinner("Merging PDFs..."):
                    try:
                        # Get files in the specified order
                        ordered_data = [st.session_state.pdf_file_data[i]["data"] for i in order]

                        # Merge
                        service = PDFService()
                        merged_pdf = service.merge_pdfs(ordered_data)

                        st.success("✅ PDFs merged successfully!")

                        # Calculate merged size
                        merged_size_mb = len(merged_pdf) / (1024 * 1024)
                        st.info(f"📊 Merged PDF size: {merged_size_mb:.1f} MB")

                        # Default filename based on first file in order
                        first_name = st.session_state.pdf_file_data[order[0]]["name"].rsplit(
                            ".", 1
                        )[0]
                        default_name = f"{first_name}_merged.pdf"

                        st.download_button(
                            label="📥 Download Merged PDF",
                            data=merged_pdf,
                            file_name=default_name,
                            mime="application/pdf",
                            type="primary",
                            use_container_width=True,
                        )

                    except Exception as e:
                        st.error(f"Failed to merge PDFs: {e}")

with tab2:
    st.header("PDF Shrinker")
    st.markdown("""
    Reduce the file size of your PDF documents by re-compressing all images.
    
    **Perfect for:** Large scanned documents, manga PDFs, or any image-heavy PDFs.
    """)

    uploaded_file = st.file_uploader(
        "Choose a PDF file to shrink", type="pdf", help="Upload a PDF file to reduce its size"
    )

    if uploaded_file:
        # Show original file info
        original_bytes = uploaded_file.read()
        original_size = len(original_bytes)
        original_size_mb = original_size / (1024 * 1024)

        st.info(f"📄 Original file: **{uploaded_file.name}** ({original_size_mb:.2f} MB)")

        # Compression settings
        st.subheader("⚙️ Compression Settings")

        col1, col2 = st.columns(2)
        with col1:
            quality = st.slider(
                "JPEG Quality",
                min_value=30,
                max_value=95,
                value=75,
                step=5,
                help="Lower = smaller file, but less quality. 60-75 is usually good for manga.",
            )
        with col2:
            max_dimension = st.slider(
                "Max Image Size (pixels)",
                min_value=800,
                max_value=2400,
                value=1400,
                step=100,
                help="Maximum width/height of images. 1200-1400 is usually readable.",
            )

        # Estimate file size
        estimated_reduction = min(90, max(30, (95 - quality) + (2400 - max_dimension) / 20))
        estimated_size = original_size_mb * (1 - estimated_reduction / 100)
        st.caption(f"📊 Estimated output: ~{estimated_size:.1f} MB (this is a rough estimate)")

        if st.button("🔧 Shrink PDF", type="primary", use_container_width=True):
            with st.spinner("Shrinking PDF... This may take a while for large files."):
                try:
                    # Shrink with image re-compression
                    service = PDFService()
                    shrunk_pdf = service.shrink_pdf(
                        original_bytes,
                        quality=quality,
                        max_dimension=max_dimension,
                    )

                    # Calculate stats
                    shrunk_size = len(shrunk_pdf)
                    shrunk_size_mb = shrunk_size / (1024 * 1024)
                    reduction = ((original_size - shrunk_size) / original_size) * 100

                    st.success("✅ PDF shrunk successfully!")

                    # Show stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Size", f"{original_size_mb:.2f} MB")
                    with col2:
                        st.metric("New Size", f"{shrunk_size_mb:.2f} MB")
                    with col3:
                        st.metric(
                            "Reduction",
                            f"{reduction:.1f}%",
                            delta=f"-{original_size_mb - shrunk_size_mb:.1f} MB",
                        )

                    # Default filename
                    base_name = uploaded_file.name.rsplit(".", 1)[0]
                    default_name = f"{base_name}_shrunk.pdf"

                    st.download_button(
                        label="📥 Download Shrunk PDF",
                        data=shrunk_pdf,
                        file_name=default_name,
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True,
                    )

                except Exception as e:
                    st.error(f"Failed to shrink PDF: {e}")
                    st.exception(e)

with tab3:
    st.header("🌐 Web Manga to PDF")
    st.markdown("""
    Download manga chapters from websites and convert them to PDF.
    
    **✨ Uses a real browser** to load JavaScript-rendered pages and print to PDF.
    """)

    def download_page_as_pdf(url: str, progress_callback=None, method: str = "extract") -> bytes:
        """Load a webpage and convert it to PDF.
        
        Args:
            url: URL to download
            progress_callback: Optional callback(progress, text)
            method: "extract" (download images) or "print" (browser print to PDF)
        """
        from playwright.sync_api import sync_playwright
        from loguru import logger
        import requests

        logger.info(f"Starting download for: {url} (method={method})")

        if progress_callback:
            progress_callback(0.05, "Starting browser...")

        with sync_playwright() as p:
            # Launch headless browser
            logger.debug("Launching Chromium...")
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1200, "height": 800},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            )
            page = context.new_page()

            if progress_callback:
                progress_callback(0.1, "Loading page...")

            # Navigate to URL
            logger.debug(f"Navigating to {url}...")
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
            except Exception as e:
                logger.warning(f"Initial load timeout, continuing anyway: {e}")

            # Wait for page to settle
            logger.debug("Waiting for page to settle...")
            time.sleep(3)

            if progress_callback:
                progress_callback(0.2, "Scrolling to load all images...")

            # Scroll to bottom to trigger lazy loading
            logger.debug("Scrolling page...")
            page.evaluate("""
                async () => {
                    await new Promise((resolve) => {
                        let totalHeight = 0;
                        const distance = 500;
                        const timer = setInterval(() => {
                            window.scrollBy(0, distance);
                            totalHeight += distance;
                            if (totalHeight >= document.body.scrollHeight) {
                                clearInterval(timer);
                                resolve();
                            }
                        }, 100);
                        setTimeout(() => { clearInterval(timer); resolve(); }, 15000);
                    });
                }
            """)

            if progress_callback:
                progress_callback(0.5, "Waiting for images to load...")

            # Wait for images to load after scrolling
            time.sleep(3)

            if method == "print":
                # Original "Print to PDF" method
                # Scroll back to top
                page.evaluate("window.scrollTo(0, 0)")
                time.sleep(0.5)

                if progress_callback:
                    progress_callback(0.6, "Hiding non-essential elements...")

                logger.debug("Hiding non-content elements...")
                page.evaluate("""
                    () => {
                        const selectorsToHide = [
                            'header', 'footer', 'nav', '.navbar', '.nav',
                            '.header', '.footer', '.sidebar', '.ad', '.ads',
                            '.advertisement', '[class*="banner"]', '[class*="popup"]',
                            '[class*="modal"]', '[class*="cookie"]', '[class*="consent"]',
                            '.social-share', '.comments', '#comments', '.related',
                            '[class*="recommend"]', '.navigation', '.breadcrumb'
                        ];
                        selectorsToHide.forEach(selector => {
                            document.querySelectorAll(selector).forEach(el => {
                                el.style.display = 'none';
                            });
                        });
                        document.querySelectorAll('*').forEach(el => {
                            const style = window.getComputedStyle(el);
                            if (style.position === 'fixed' || style.position === 'sticky') {
                                el.style.position = 'relative';
                            }
                        });
                    }
                """)

                if progress_callback:
                    progress_callback(0.8, "Generating PDF...")

                logger.debug("Generating PDF...")
                pdf_bytes = page.pdf(
                    format="A4",
                    print_background=True,
                    margin={"top": "10mm", "bottom": "10mm", "left": "5mm", "right": "5mm"},
                )
                
            else:
                # "Extract Images" method
                if progress_callback:
                    progress_callback(0.6, "Finding images...")
                
                # Extract image URLs
                # Heuristic: Filter for images that are likely manga pages (large enough)
                image_urls = page.evaluate("""
                    () => {
                        const images = Array.from(document.querySelectorAll('img'));
                        return images
                            .filter(img => {
                                // Filter out small icons/ads based on natural size or displayed size
                                const width = img.naturalWidth || img.width;
                                const height = img.naturalHeight || img.height;
                                return width > 200 && height > 300;
                            })
                            .map(img => img.src)
                            .filter(src => src && !src.startsWith('data:')); // Ignore base64 for now if complex
                    }
                """)
                
                # Deduplicate
                image_urls = list(dict.fromkeys(image_urls))
                logger.info(f"Found {len(image_urls)} potential manga images")
                
                if not image_urls:
                    raise Exception("No suitable images found on the page.")
                
                if progress_callback:
                    progress_callback(0.7, f"Downloading {len(image_urls)} images...")
                
                downloaded_images = []
                for i, img_url in enumerate(image_urls):
                    try:
                        # Download image
                        response = requests.get(img_url, timeout=10)
                        if response.status_code == 200:
                            img = Image.open(io.BytesIO(response.content))
                            img.load() # Verify it's a valid image
                            downloaded_images.append(img)
                        
                        if progress_callback:
                            # Update progress within the 0.7-0.9 range
                            p = 0.7 + (0.2 * (i + 1) / len(image_urls))
                            progress_callback(p, f"Downloading image {i+1}/{len(image_urls)}...")
                            
                    except Exception as e:
                        logger.warning(f"Failed to download image {img_url}: {e}")
                
                if not downloaded_images:
                    raise Exception("Failed to download any images.")
                
                if progress_callback:
                    progress_callback(0.9, "Creating PDF from images...")
                
                # Create PDF using PDFService
                # We use a temporary service instance here
                service = PDFService()
                pdf_bytes = service.create_pdf_from_images(downloaded_images)

            logger.info(f"PDF generated: {len(pdf_bytes) / 1024 / 1024:.1f} MB")
            browser.close()

            if progress_callback:
                progress_callback(1.0, "Done!")

            return pdf_bytes

    def extract_chapter_info(url: str) -> tuple[str, str]:
        """Try to extract manga title and chapter number from URL."""
        patterns = [
            r"/title/([^/]+)/chapter/([^/]+)",
            r"/manga/([^/]+)/chapter[/-]?(\d+)",
            r"/read/([^/]+)/chapter[/-]?(\d+)",
            r"/([^/]+)/chapter[/-]?(\d+)",
            r"/chapter[/-]?(\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    title = groups[0].replace("-", " ").replace("_", " ").title()
                    chapter = groups[1]
                    return title, chapter
                elif len(groups) == 1:
                    return "Manga", groups[0]

        return "Manga", "Chapter"

    # Mode selection
    mode = st.radio(
        "Mode",
        options=["Single Chapter", "Series"],
        horizontal=True,
        help="Download a single chapter or a whole series/range of chapters.",
    )

    if mode == "Single Chapter":
        # UI
        url = st.text_input(
            "🔗 Manga Chapter URL",
            placeholder="https://example.com/manga/title/chapter/1",
            help="Paste the URL of a manga chapter page",
        )

        # Settings
        st.subheader("⚙️ Settings")
        col1, col2 = st.columns(2)
        with col1:
            method_display = st.radio(
                "Download Method",
                options=["Extract Images", "Print Page"],
                help="Extract Images: Finds and downloads images (Best for most sites).\nPrint Page: Prints the whole page as PDF (Good for simple sites).",
                key="method_single"
            )
            method = "extract" if method_display == "Extract Images" else "print"
            
        with col2:
            quality = st.slider(
                "JPEG Quality",
                min_value=40,
                max_value=90,
                value=70,
                step=5,
                help="Lower = smaller file. 60-70 is usually good for manga.",
                key="webmanga_quality_single",
            )
            
        max_dimension = st.slider(
            "Max Image Size (px)",
            min_value=800,
            max_value=2000,
            value=1400,
            step=100,
            help="Maximum width/height. 1200-1400 is readable.",
            key="webmanga_maxdim_single",
        )

        # Always show filename input and button
        if url:
            title, chapter = extract_chapter_info(url)
            default_filename = f"{title} - Chapter {chapter}.pdf"
        else:
            default_filename = "manga_chapter.pdf"

        filename = st.text_input("📁 Output filename", value=default_filename)
        if not filename.endswith(".pdf"):
            filename += ".pdf"

        # Button always visible, disabled if no URL
        if st.button(
            "📥 Download & Create PDF", type="primary", use_container_width=True, disabled=not url
        ):
            progress_bar = st.progress(0, text="Starting...")
            status_text = st.empty()  # For detailed status updates

            def update_progress(progress, text):
                progress_bar.progress(min(progress, 0.99), text=text)
                status_text.info(f"🔄 {text}")

            try:
                # Step 1: Download page as PDF
                raw_pdf_bytes = download_page_as_pdf(url, update_progress, method=method)
                raw_size_mb = len(raw_pdf_bytes) / (1024 * 1024)

                # Step 2: Compress the PDF using our shrinker
                # If we extracted images, they are already compressed by create_pdf_from_images if we used that service method?
                # Actually create_pdf_from_images in PDFService takes PIL images and makes a PDF.
                # It does compress them.
                # But here we are calling shrink_pdf again on the result?
                # If method is 'extract', raw_pdf_bytes is already a clean PDF.
                # shrink_pdf might be redundant or even degrade quality if we re-compress.
                # However, for consistency and ensuring max_dimension/quality, we can leave it, 
                # or skip it if method is extract and we trust create_pdf_from_images (which we should configure).
                # Wait, create_pdf_from_images in PDFService uses config values if not passed.
                # In download_page_as_pdf, I called service.create_pdf_from_images(downloaded_images).
                # It uses default config.
                # So re-shrinking here allows the user to apply the specific slider settings from THIS page.
                # So I will keep the shrink step for now, although it's double compression.
                # Ideally I should pass quality/max_dim to download_page_as_pdf -> create_pdf_from_images.
                # But to minimize changes, I'll leave the shrink step.
                
                update_progress(0.85, f"Optimizing PDF ({raw_size_mb:.1f} MB)...")

                service = PDFService()
                pdf_bytes = service.shrink_pdf(
                    raw_pdf_bytes,
                    quality=quality,
                    max_dimension=max_dimension,
                )

                # Clear progress displays
                progress_bar.empty()
                status_text.empty()

                pdf_size_mb = len(pdf_bytes) / (1024 * 1024)
                reduction = (1 - pdf_size_mb / raw_size_mb) * 100 if raw_size_mb > 0 else 0

                st.success(
                    f"✅ PDF created: {pdf_size_mb:.1f} MB (optimized from {raw_size_mb:.1f} MB)"
                )

                st.download_button(
                    label="📥 Download PDF",
                    data=pdf_bytes,
                    file_name=filename,
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True,
                )

            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Error: {e}")
                st.exception(e)

    else:  # Series Mode
        st.info(
            "ℹ️ **Series Mode:** Enter a URL pattern with `{chapter}` placeholder to download a range of chapters."
        )

        url_pattern = st.text_input(
            "🔗 URL Pattern",
            placeholder="https://w16.sololevelinganime.com/manga/solo-leveling-chapter-{chapter}/",
            help="Use {chapter} where the chapter number should go.",
        )

        col1, col2 = st.columns(2)
        with col1:
            start_chapter = st.number_input("Start Chapter", min_value=1, value=1, step=1)
        with col2:
            end_chapter = st.number_input("End Chapter", min_value=1, value=10, step=1)

        series_title = st.text_input("Series Title", placeholder="Solo Leveling")

        # URL Preview
        if url_pattern and "{chapter}" in url_pattern:
            total_chapters_preview = end_chapter - start_chapter + 1
            if total_chapters_preview > 0:
                with st.expander(f"👁️ Preview {total_chapters_preview} URLs"):
                    preview_urls = []
                    # Show first 5 and last 5 if too many
                    if total_chapters_preview > 10:
                        for i in range(start_chapter, start_chapter + 5):
                            preview_urls.append(url_pattern.replace("{chapter}", str(i)))
                        preview_urls.append("...")
                        for i in range(end_chapter - 4, end_chapter + 1):
                            preview_urls.append(url_pattern.replace("{chapter}", str(i)))
                    else:
                        for i in range(start_chapter, end_chapter + 1):
                            preview_urls.append(url_pattern.replace("{chapter}", str(i)))
                    
                    for url in preview_urls:
                        st.code(url, language="text")
            else:
                st.warning("End chapter must be greater than or equal to start chapter.")
        elif url_pattern:
            st.warning("⚠️ URL pattern must contain `{chapter}` placeholder.")

        # Settings
        st.subheader("⚙️ Settings")
        col1, col2 = st.columns(2)
        with col1:
            method_display = st.radio(
                "Download Method",
                options=["Extract Images", "Print Page"],
                help="Extract Images: Finds and downloads images (Best for most sites).\nPrint Page: Prints the whole page as PDF (Good for simple sites).",
                key="method_series"
            )
            method = "extract" if method_display == "Extract Images" else "print"

        with col2:
            quality = st.slider(
                "JPEG Quality",
                min_value=40,
                max_value=90,
                value=70,
                step=5,
                help="Lower = smaller file. 60-70 is usually good for manga.",
                key="webmanga_quality_series",
            )
            
        max_dimension = st.slider(
            "Max Image Size (px)",
            min_value=800,
            max_value=2000,
            value=1400,
            step=100,
            help="Maximum width/height. 1200-1400 is readable.",
            key="webmanga_maxdim_series",
        )

        default_filename = f"{series_title} - Chapters {start_chapter}-{end_chapter}.pdf" if series_title else "series.pdf"
        filename = st.text_input("📁 Output filename", value=default_filename, key="series_filename")
        if not filename.endswith(".pdf"):
            filename += ".pdf"

        if st.button(
            "📥 Download Series & Create PDF",
            type="primary",
            use_container_width=True,
            disabled=not url_pattern or not series_title,
        ):
            progress_bar = st.progress(0, text="Starting...")
            status_text = st.empty()
            
            # Container for chapter status
            chapter_status = st.container()

            def update_progress(progress, text):
                # Scale progress to current chapter's slot
                # This is tricky with multiple chapters, so we'll just show the text
                status_text.info(f"🔄 {text}")

            try:
                chapters_pdf_bytes = []
                total_chapters = end_chapter - start_chapter + 1
                
                service = PDFService()

                for i, chapter_num in enumerate(range(start_chapter, end_chapter + 1)):
                    current_url = url_pattern.replace("{chapter}", str(chapter_num))
                    
                    # Update overall progress
                    overall_progress = i / total_chapters
                    progress_bar.progress(overall_progress, text=f"Processing Chapter {chapter_num} ({i+1}/{total_chapters})")
                    
                    with chapter_status:
                        st.write(f"Downloading Chapter {chapter_num}: {current_url}")

                    # Download chapter
                    # We pass a dummy callback or None because we handle progress differently here
                    # Or we could create a wrapper callback
                    raw_bytes = download_page_as_pdf(current_url, None, method=method)
                    
                    # Shrink immediately to save memory
                    shrunk_bytes = service.shrink_pdf(
                        raw_bytes,
                        quality=quality,
                        max_dimension=max_dimension,
                    )
                    chapters_pdf_bytes.append(shrunk_bytes)
                    
                    # Offer individual download
                    with chapter_status:
                        st.download_button(
                            label=f"📥 Download Chapter {chapter_num}",
                            data=shrunk_bytes,
                            file_name=f"{series_title} - Chapter {chapter_num}.pdf",
                            mime="application/pdf",
                            key=f"dl_btn_{chapter_num}"
                        )

                progress_bar.progress(0.9, text="Merging chapters...")
                status_text.info("🔄 Merging all chapters into one PDF...")

                # Merge all chapters
                merged_pdf = service.merge_pdfs(chapters_pdf_bytes)

                progress_bar.progress(1.0, text="Done!")
                status_text.success("✅ Series processing complete!")

                merged_size_mb = len(merged_pdf) / (1024 * 1024)
                st.info(f"📊 Final PDF size: {merged_size_mb:.1f} MB")

                st.download_button(
                    label="📥 Download Series PDF",
                    data=merged_pdf,
                    file_name=filename,
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"❌ Error during series processing: {e}")
                st.exception(e)

    st.divider()
    st.caption("""
    **Note:** This tool uses a headless browser to print the page as PDF.
    Headers, footers, and ads are automatically hidden. Please respect copyright laws.
    """)

with tab4:
    st.header("📚 CBZ Creator")
    st.markdown("""
    Create CBZ (Comic Book Zip) archives from images. CBZ is the standard format for digital comics and manga readers.
    
    **Perfect for:** Converting image folders into a format readable by manga/comic apps.
    """)

    uploaded_images = st.file_uploader(
        "Upload images (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="Select multiple images to combine into a CBZ archive",
        key="cbz_uploader",
    )

    if uploaded_images:
        st.info(f"📷 {len(uploaded_images)} images uploaded")

        # Image format settings
        st.subheader("⚙️ Settings")
        col1, col2 = st.columns(2)
        with col1:
            image_format = st.selectbox(
                "Image Format",
                options=["JPEG", "WEBP", "PNG"],
                index=0,
                help="JPEG: Good compression. WEBP: Best compression. PNG: Lossless.",
                key="cbz_format",
            )
        with col2:
            quality = st.slider(
                "Quality",
                min_value=50,
                max_value=100,
                value=85,
                help="Quality for JPEG/WEBP (ignored for PNG)",
                key="cbz_quality",
            )

        # Filename
        default_name = "manga.cbz"
        if uploaded_images:
            # Try to get name from first file
            first_name = uploaded_images[0].name.rsplit(".", 1)[0]
            # Remove common suffixes like _001, -001, etc.
            base_name = re.sub(r"[-_]?\d+$", "", first_name)
            if base_name:
                default_name = f"{base_name}.cbz"

        filename = st.text_input("📁 Output filename", value=default_name, key="cbz_filename")
        if not filename.endswith(".cbz"):
            filename += ".cbz"

        if st.button("📦 Create CBZ", type="primary", use_container_width=True, key="cbz_create"):
            with st.spinner("Creating CBZ archive..."):
                try:
                    # Load images
                    images = []
                    for f in uploaded_images:
                        img = Image.open(f)
                        img.load()  # Ensure fully loaded
                        images.append(img)

                    # Create CBZ
                    service = ArchiveService(image_format=image_format, quality=quality)
                    cbz_bytes = service.create_cbz(images)

                    size_mb = len(cbz_bytes) / (1024 * 1024)
                    st.success(f"✅ CBZ created: {size_mb:.2f} MB")

                    st.download_button(
                        label="📥 Download CBZ",
                        data=cbz_bytes,
                        file_name=filename,
                        mime="application/x-cbz",
                        type="primary",
                        use_container_width=True,
                    )

                except Exception as e:
                    st.error(f"Failed to create CBZ: {e}")

with tab5:
    st.header("📖 EPUB Creator")
    st.markdown("""
    Create EPUB ebooks from text. Perfect for light novels and text-based content.
    
    **Features:**
    - Automatic chapter detection (split by `---` or blank lines)
    - Optional cover image
    - Clean, readable formatting
    """)

    # Book metadata
    st.subheader("📝 Book Information")
    col1, col2 = st.columns(2)
    with col1:
        book_title = st.text_input("Title", placeholder="My Light Novel", key="epub_title")
    with col2:
        book_author = st.text_input("Author", placeholder="Author Name", key="epub_author")

    col3, col4 = st.columns(2)
    with col3:
        book_language = st.selectbox(
            "Language",
            options=["en", "de", "ja", "fr", "es", "zh", "ko"],
            format_func=lambda x: {
                "en": "English",
                "de": "German",
                "ja": "Japanese",
                "fr": "French",
                "es": "Spanish",
                "zh": "Chinese",
                "ko": "Korean",
            }[x],
            key="epub_language",
        )
    with col4:
        chapter_separator = st.selectbox(
            "Chapter Separator",
            options=["---", "***", "===", "blank_lines"],
            format_func=lambda x: {
                "---": "Three dashes (---)",
                "***": "Three asterisks (***)",
                "===": "Three equals (===)",
                "blank_lines": "Double blank lines",
            }[x],
            help="How chapters are separated in your text",
            key="epub_separator",
        )

    # Cover image (optional)
    cover_file = st.file_uploader(
        "Cover Image (optional)",
        type=["jpg", "jpeg", "png", "webp"],
        help="Optional cover image for the ebook",
        key="epub_cover",
    )

    # Text content
    st.subheader("📄 Content")
    content_method = st.radio(
        "Input method",
        options=["text", "file"],
        format_func=lambda x: "Paste text" if x == "text" else "Upload text file",
        horizontal=True,
        key="epub_input_method",
    )

    book_content = ""
    if content_method == "text":
        book_content = st.text_area(
            "Book content",
            height=300,
            placeholder="Paste your light novel text here...\n\n---\n\nChapter 2 starts here...",
            help="Separate chapters with the selected separator",
            key="epub_content",
        )
    else:
        content_file = st.file_uploader(
            "Upload text file",
            type=["txt", "md"],
            help="Upload a .txt or .md file",
            key="epub_content_file",
        )
        if content_file:
            book_content = content_file.read().decode("utf-8")
            st.success(f"Loaded {len(book_content)} characters")

    # Preview
    if book_content:
        sep_map = {
            "---": "\n\n---\n\n",
            "***": "\n\n***\n\n",
            "===": "\n\n===\n\n",
            "blank_lines": "\n\n\n\n",
        }
        sep = sep_map[chapter_separator]
        chapters = book_content.split(sep)
        chapters = [c.strip() for c in chapters if c.strip()]
        st.info(f"📚 Detected {len(chapters)} chapter(s)")

    # Filename
    if book_title:
        default_epub_name = f"{book_title}.epub"
    else:
        default_epub_name = "book.epub"

    epub_filename = st.text_input(
        "📁 Output filename", value=default_epub_name, key="epub_filename"
    )
    if not epub_filename.endswith(".epub"):
        epub_filename += ".epub"

    # Create button
    can_create = book_title and book_author and book_content
    if st.button(
        "📖 Create EPUB",
        type="primary",
        use_container_width=True,
        disabled=not can_create,
        key="epub_create",
    ):
        with st.spinner("Creating EPUB..."):
            try:
                # Load cover if provided
                cover_image = None
                if cover_file:
                    cover_image = Image.open(cover_file)

                # Create EPUB
                service = EPUBService()
                sep_map = {
                    "---": "\n\n---\n\n",
                    "***": "\n\n***\n\n",
                    "===": "\n\n===\n\n",
                    "blank_lines": "\n\n\n\n",
                }
                sep = sep_map[chapter_separator]

                epub_bytes = service.create_epub_from_text(
                    title=book_title,
                    author=book_author,
                    text=book_content,
                    cover_image=cover_image,
                    language=book_language,
                    chapter_separator=sep,
                )

                size_kb = len(epub_bytes) / 1024
                st.success(f"✅ EPUB created: {size_kb:.1f} KB")

                st.download_button(
                    label="📥 Download EPUB",
                    data=epub_bytes,
                    file_name=epub_filename,
                    mime="application/epub+zip",
                    type="primary",
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"Failed to create EPUB: {e}")

    if not can_create:
        st.caption("Fill in title, author, and content to create EPUB")

with tab6:
    st.header("🔄 Format Converter")
    st.markdown("""
    Convert between different manga/ebook formats.
    """)

    conversion_type = st.selectbox(
        "Conversion Type",
        options=["pdf_to_cbz", "cbz_to_pdf", "images_to_pdf"],
        format_func=lambda x: {
            "pdf_to_cbz": "PDF → CBZ",
            "cbz_to_pdf": "CBZ → PDF",
            "images_to_pdf": "Images → PDF",
        }[x],
        key="convert_type",
    )

    if conversion_type == "pdf_to_cbz":
        st.subheader("PDF → CBZ")
        st.markdown("Convert a PDF file to CBZ format for better compatibility with manga readers.")

        pdf_file = st.file_uploader("Upload PDF", type=["pdf"], key="convert_pdf_to_cbz")

        if pdf_file:
            # Settings
            col1, col2 = st.columns(2)
            with col1:
                img_format = st.selectbox(
                    "Image Format",
                    options=["JPEG", "WEBP", "PNG"],
                    index=0,
                    key="convert_img_format",
                )
            with col2:
                img_quality = st.slider(
                    "Quality",
                    min_value=50,
                    max_value=100,
                    value=85,
                    key="convert_quality",
                )

            base_name = pdf_file.name.rsplit(".", 1)[0]
            output_name = st.text_input(
                "Output filename", value=f"{base_name}.cbz", key="convert_output_name"
            )
            if not output_name.endswith(".cbz"):
                output_name += ".cbz"

            if st.button("🔄 Convert to CBZ", type="primary", use_container_width=True):
                with st.spinner("Converting PDF to CBZ..."):
                    try:
                        # Load PDF
                        pdf_bytes = pdf_file.read()
                        pdf_service = PDFService()
                        page_count = pdf_service.load_from_bytes(pdf_bytes)

                        # Extract all pages as images
                        images = []
                        progress = st.progress(0)
                        for i in range(page_count):
                            progress.progress(
                                (i + 1) / page_count, f"Extracting page {i + 1}/{page_count}"
                            )
                            img = pdf_service.get_page_image(i)
                            images.append(img)

                        # Create CBZ
                        progress.progress(1.0, "Creating CBZ...")
                        archive_service = ArchiveService(
                            image_format=img_format, quality=img_quality
                        )
                        cbz_bytes = archive_service.create_cbz(images)

                        pdf_service.close()
                        progress.empty()

                        size_mb = len(cbz_bytes) / (1024 * 1024)
                        st.success(f"✅ Converted: {size_mb:.2f} MB ({page_count} pages)")

                        st.download_button(
                            label="📥 Download CBZ",
                            data=cbz_bytes,
                            file_name=output_name,
                            mime="application/x-cbz",
                            type="primary",
                            use_container_width=True,
                        )

                    except Exception as e:
                        st.error(f"Conversion failed: {e}")

    elif conversion_type == "cbz_to_pdf":
        st.subheader("CBZ → PDF")
        st.markdown("Convert a CBZ archive to PDF format.")

        cbz_file = st.file_uploader("Upload CBZ", type=["cbz", "zip"], key="convert_cbz_to_pdf")

        if cbz_file:
            col1, col2 = st.columns(2)
            with col1:
                pdf_quality = st.slider("JPEG Quality", 50, 95, 75, key="cbz_pdf_quality")
            with col2:
                pdf_max_dim = st.slider(
                    "Max Image Size", 800, 2400, 1400, step=100, key="cbz_pdf_maxdim"
                )

            base_name = cbz_file.name.rsplit(".", 1)[0]
            output_name = st.text_input(
                "Output filename", value=f"{base_name}.pdf", key="cbz_output_name"
            )
            if not output_name.endswith(".pdf"):
                output_name += ".pdf"

            if st.button("🔄 Convert to PDF", type="primary", use_container_width=True):
                with st.spinner("Converting CBZ to PDF..."):
                    try:
                        # Extract images from CBZ
                        cbz_bytes = cbz_file.read()
                        archive_service = ArchiveService()
                        images = archive_service.extract_cbz(cbz_bytes)

                        # Create PDF
                        pdf_service = PDFService()
                        pdf_bytes = pdf_service.create_pdf_from_images(
                            images,
                            quality=pdf_quality,
                            max_dimension=pdf_max_dim,
                        )

                        size_mb = len(pdf_bytes) / (1024 * 1024)
                        st.success(f"✅ Converted: {size_mb:.2f} MB ({len(images)} pages)")

                        st.download_button(
                            label="📥 Download PDF",
                            data=pdf_bytes,
                            file_name=output_name,
                            mime="application/pdf",
                            type="primary",
                            use_container_width=True,
                        )

                    except Exception as e:
                        st.error(f"Conversion failed: {e}")

    elif conversion_type == "images_to_pdf":
        st.subheader("Images → PDF")
        st.markdown("Combine multiple images into a PDF document.")

        image_files = st.file_uploader(
            "Upload images",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="convert_images_to_pdf",
        )

        if image_files:
            st.info(f"📷 {len(image_files)} images uploaded")

            col1, col2 = st.columns(2)
            with col1:
                pdf_quality = st.slider("JPEG Quality", 50, 95, 75, key="img_pdf_quality")
            with col2:
                pdf_max_dim = st.slider(
                    "Max Image Size", 800, 2400, 1400, step=100, key="img_pdf_maxdim"
                )

            base_name = image_files[0].name.rsplit(".", 1)[0]
            base_name = re.sub(r"[-_]?\d+$", "", base_name)
            output_name = st.text_input(
                "Output filename", value=f"{base_name or 'images'}.pdf", key="img_output_name"
            )
            if not output_name.endswith(".pdf"):
                output_name += ".pdf"

            if st.button("🔄 Create PDF", type="primary", use_container_width=True):
                with st.spinner("Creating PDF..."):
                    try:
                        # Load images
                        images = []
                        for f in image_files:
                            img = Image.open(f)
                            img.load()
                            images.append(img)

                        # Create PDF
                        pdf_service = PDFService()
                        pdf_bytes = pdf_service.create_pdf_from_images(
                            images,
                            quality=pdf_quality,
                            max_dimension=pdf_max_dim,
                        )

                        size_mb = len(pdf_bytes) / (1024 * 1024)
                        st.success(f"✅ Created: {size_mb:.2f} MB ({len(images)} pages)")

                        st.download_button(
                            label="📥 Download PDF",
                            data=pdf_bytes,
                            file_name=output_name,
                            mime="application/pdf",
                            type="primary",
                            use_container_width=True,
                        )

                    except Exception as e:
                        st.error(f"Failed to create PDF: {e}")

with tab7:
    st.header("👁️ CBZ Reader")
    st.markdown("""
    Read CBZ/CBR comic book archives directly in the browser.
    """)

    # Initialize session state for reader
    if "reader_images" not in st.session_state:
        st.session_state.reader_images = []
    if "reader_current_page" not in st.session_state:
        st.session_state.reader_current_page = 0
    if "reader_filename" not in st.session_state:
        st.session_state.reader_filename = ""

    cbz_file = st.file_uploader(
        "Upload CBZ/CBR file",
        type=["cbz", "cbr", "zip", "rar"],
        help="Upload a comic book archive to read",
        key="cbz_reader_upload",
    )

    if cbz_file:
        # Check if we need to load a new file
        if cbz_file.name != st.session_state.reader_filename:
            with st.spinner("Loading comic..."):
                try:
                    cbz_bytes = cbz_file.read()
                    archive_service = ArchiveService()

                    # Try CBZ first, then CBR
                    if cbz_file.name.lower().endswith((".cbr", ".rar")):
                        images = archive_service.extract_cbr(cbz_bytes)
                    else:
                        images = archive_service.extract_cbz(cbz_bytes)

                    st.session_state.reader_images = images
                    st.session_state.reader_current_page = 0
                    st.session_state.reader_filename = cbz_file.name

                except Exception as e:
                    st.error(f"Failed to load archive: {e}")
                    st.session_state.reader_images = []

    # Display reader if we have images
    if st.session_state.reader_images:
        images = st.session_state.reader_images
        total_pages = len(images)
        current_page = st.session_state.reader_current_page

        # Ensure current page is valid
        if current_page >= total_pages:
            current_page = 0
            st.session_state.reader_current_page = 0

        st.success(f"📖 **{st.session_state.reader_filename}** - {total_pages} pages")

        # Reading mode selection
        col_mode, col_fit = st.columns(2)
        with col_mode:
            reading_mode = st.radio(
                "Reading Mode",
                options=["single", "double", "vertical"],
                format_func=lambda x: {
                    "single": "Single Page",
                    "double": "Double Page (Manga)",
                    "vertical": "Vertical Scroll",
                }[x],
                horizontal=True,
                key="reader_mode",
            )
        with col_fit:
            fit_mode = st.radio(
                "Fit",
                options=["width", "height", "original"],
                format_func=lambda x: {
                    "width": "Fit Width",
                    "height": "Fit Height",
                    "original": "Original Size",
                }[x],
                horizontal=True,
                key="reader_fit",
            )

        # Navigation controls
        st.divider()

        if reading_mode == "vertical":
            # Vertical scroll mode - show all pages
            st.caption(f"Showing all {total_pages} pages (scroll down)")

            for idx, img in enumerate(images):
                st.image(img, caption=f"Page {idx + 1}", use_container_width=(fit_mode == "width"))

        else:
            # Single or double page mode with navigation
            nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 1, 2, 1, 1])

            with nav_col1:
                if st.button("⏮️ First", use_container_width=True, disabled=current_page == 0):
                    st.session_state.reader_current_page = 0
                    st.rerun()

            with nav_col2:
                step = 2 if reading_mode == "double" else 1
                if st.button("◀️ Prev", use_container_width=True, disabled=current_page == 0):
                    st.session_state.reader_current_page = max(0, current_page - step)
                    st.rerun()

            with nav_col3:
                # Page selector
                if reading_mode == "double":
                    page_display = f"Pages {current_page + 1}-{min(current_page + 2, total_pages)} of {total_pages}"
                else:
                    page_display = f"Page {current_page + 1} of {total_pages}"

                new_page = st.number_input(
                    "Go to page",
                    min_value=1,
                    max_value=total_pages,
                    value=current_page + 1,
                    key="reader_page_input",
                    label_visibility="collapsed",
                )
                if new_page - 1 != current_page:
                    st.session_state.reader_current_page = new_page - 1
                    st.rerun()

            with nav_col4:
                step = 2 if reading_mode == "double" else 1
                max_page = total_pages - 1 if reading_mode == "single" else total_pages - 2
                if st.button("Next ▶️", use_container_width=True, disabled=current_page >= max_page):
                    st.session_state.reader_current_page = min(max_page, current_page + step)
                    st.rerun()

            with nav_col5:
                max_page = total_pages - 1 if reading_mode == "single" else max(0, total_pages - 2)
                if st.button("Last ⏭️", use_container_width=True, disabled=current_page >= max_page):
                    st.session_state.reader_current_page = max_page
                    st.rerun()

            # Display current page(s)
            st.divider()

            if reading_mode == "single":
                # Single page view
                img = images[current_page]
                st.image(
                    img,
                    caption=f"Page {current_page + 1}",
                    use_container_width=(fit_mode == "width"),
                )

            elif reading_mode == "double":
                # Double page spread (manga style - right to left)
                col_right, col_left = st.columns(2)

                # Right page first (manga reading order)
                with col_right:
                    if current_page < total_pages:
                        st.image(
                            images[current_page],
                            caption=f"Page {current_page + 1}",
                            use_container_width=(fit_mode == "width"),
                        )

                # Left page second
                with col_left:
                    if current_page + 1 < total_pages:
                        st.image(
                            images[current_page + 1],
                            caption=f"Page {current_page + 2}",
                            use_container_width=(fit_mode == "width"),
                        )

            # Keyboard navigation hint
            st.caption("💡 Tip: Use the page number input to jump to any page")

        # Close/clear button
        st.divider()
        if st.button("🗑️ Close Comic", use_container_width=True):
            st.session_state.reader_images = []
            st.session_state.reader_current_page = 0
            st.session_state.reader_filename = ""
            st.rerun()

    else:
        st.info("📚 Upload a CBZ or CBR file to start reading")
