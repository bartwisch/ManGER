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

st.set_page_config(
    page_title="Tools - ManGER",
    page_icon="🛠️",
    layout="wide",
)

st.title("🛠️ Tools")

tab1, tab2, tab3, tab4 = st.tabs(["📄 PDF Combiner", "📉 PDF Shrinker", "🌐 Web Manga to PDF", "🚧 More Tools"])

with tab1:
    st.header("PDF Combiner")
    st.markdown("Upload multiple PDF files to combine them into a single document. **Drag to reorder!**")
    
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
        if "pdf_file_data" not in st.session_state or len(st.session_state.get("pdf_file_data", [])) != len(uploaded_files):
            st.session_state.pdf_file_data = []
            for f in uploaded_files:
                f.seek(0)
                st.session_state.pdf_file_data.append({
                    "name": f.name,
                    "data": f.read(),
                    "size": f.size,
                })
        
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
                        order[display_idx], order[display_idx - 1] = order[display_idx - 1], order[display_idx]
                        st.rerun()
                else:
                    st.write("")  # Empty placeholder
            
            with col4:
                if display_idx < len(order) - 1:
                    if st.button("⬇️", key=f"down_{file_idx}", help="Move down"):
                        # Swap with next
                        order[display_idx], order[display_idx + 1] = order[display_idx + 1], order[display_idx]
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
                        first_name = st.session_state.pdf_file_data[order[0]]["name"].rsplit(".", 1)[0]
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
        "Choose a PDF file to shrink",
        type="pdf",
        help="Upload a PDF file to reduce its size"
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
                        st.metric("Reduction", f"{reduction:.1f}%", delta=f"-{original_size_mb - shrunk_size_mb:.1f} MB")
                    
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
    
    **✨ Now with JavaScript support!** Uses a real browser to load pages.
    """)
    
    def extract_images_with_browser(url: str, progress_callback=None) -> list[Image.Image]:
        """Extract manga images using Playwright browser."""
        from playwright.sync_api import sync_playwright
        
        if progress_callback:
            progress_callback(0.05, "Starting browser...")
        
        images = []
        
        with sync_playwright() as p:
            # Launch headless browser
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = context.new_page()
            
            if progress_callback:
                progress_callback(0.1, "Loading page...")
            
            # Navigate to URL
            page.goto(url, wait_until="networkidle", timeout=60000)
            
            # Wait for images to load
            time.sleep(2)
            
            # Scroll down to trigger lazy loading
            if progress_callback:
                progress_callback(0.2, "Scrolling to load all images...")
            
            # Get page height and scroll incrementally
            prev_height = 0
            scroll_attempts = 0
            while scroll_attempts < 30:  # Max 30 scroll attempts
                page.evaluate("window.scrollBy(0, 800)")
                time.sleep(0.3)
                current_height = page.evaluate("document.documentElement.scrollHeight")
                if current_height == prev_height:
                    scroll_attempts += 1
                    if scroll_attempts > 3:
                        break
                else:
                    scroll_attempts = 0
                prev_height = current_height
            
            # Scroll back to top
            page.evaluate("window.scrollTo(0, 0)")
            time.sleep(0.5)
            
            if progress_callback:
                progress_callback(0.3, "Finding manga images...")
            
            # Find all images with various selectors
            img_elements = page.query_selector_all("img")
            
            # Collect image URLs
            img_urls = []
            for img in img_elements:
                try:
                    src = img.get_attribute("src") or img.get_attribute("data-src") or img.get_attribute("data-lazy-src") or ""
                    if not src or src.startswith("data:"):
                        continue
                    
                    # Get natural dimensions
                    natural_width = img.evaluate("el => el.naturalWidth") or 0
                    natural_height = img.evaluate("el => el.naturalHeight") or 0
                    
                    # Filter out small images
                    if natural_width > 200 and natural_height > 200:
                        # Check if it's likely a manga page (taller than wide, or large)
                        if natural_height > natural_width * 0.5 or (natural_width > 400 and natural_height > 400):
                            img_urls.append({
                                "src": src,
                                "width": natural_width,
                                "height": natural_height,
                            })
                except:
                    continue
            
            # Remove duplicates while preserving order
            seen = set()
            unique_imgs = []
            for img in img_urls:
                if img["src"] not in seen:
                    seen.add(img["src"])
                    unique_imgs.append(img)
            
            if not unique_imgs:
                browser.close()
                raise ValueError("No manga images found. The site may have anti-scraping protection.")
            
            if progress_callback:
                progress_callback(0.4, f"Found {len(unique_imgs)} images, downloading...")
            
            # Download each image
            for i, img_info in enumerate(unique_imgs):
                try:
                    src = img_info["src"]
                    
                    # Make absolute URL if needed
                    if src.startswith("//"):
                        src = "https:" + src
                    elif src.startswith("/"):
                        base = page.url.split("/")[0:3]
                        src = "/".join(base) + src
                    
                    # Download using the browser context (handles cookies, etc.)
                    response = page.request.get(src)
                    
                    if response.ok:
                        img_data = response.body()
                        pil_img = Image.open(io.BytesIO(img_data))
                        
                        # Convert to RGB
                        if pil_img.mode in ('RGBA', 'LA', 'P'):
                            background = Image.new('RGB', pil_img.size, (255, 255, 255))
                            if pil_img.mode == 'P':
                                pil_img = pil_img.convert('RGBA')
                            if pil_img.mode == 'RGBA':
                                background.paste(pil_img, mask=pil_img.split()[-1])
                            else:
                                background.paste(pil_img)
                            pil_img = background
                        elif pil_img.mode != 'RGB':
                            pil_img = pil_img.convert('RGB')
                        
                        images.append(pil_img)
                    
                    if progress_callback:
                        progress = 0.4 + (0.5 * (i + 1) / len(unique_imgs))
                        progress_callback(progress, f"Downloaded {i + 1}/{len(unique_imgs)} images")
                        
                except Exception as e:
                    st.warning(f"Failed to download image {i + 1}: {e}")
                    continue
            
            browser.close()
        
        if not images:
            raise ValueError("Failed to download any images.")
        
        # Sort images by height (manga pages are usually consistently sized)
        # This helps filter out any remaining non-manga images
        if len(images) > 3:
            avg_height = sum(img.height for img in images) / len(images)
            images = [img for img in images if img.height > avg_height * 0.5]
        
        return images
    
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
    
    # UI
    url = st.text_input(
        "🔗 Manga Chapter URL",
        placeholder="https://example.com/manga/title/chapter/1",
        help="Paste the URL of a manga chapter page",
    )
    
    col1, col2 = st.columns(2)
    with col1:
        quality = st.slider(
            "JPEG Quality",
            min_value=50,
            max_value=95,
            value=80,
            step=5,
            help="Higher = better quality but larger file",
            key="webmanga_quality",
        )
    with col2:
        max_dimension = st.slider(
            "Max Image Size",
            min_value=1000,
            max_value=2400,
            value=1600,
            step=100,
            help="Maximum width/height of images",
            key="webmanga_maxdim",
        )
    
    if url:
        title, chapter = extract_chapter_info(url)
        default_filename = f"{title} - Chapter {chapter}.pdf"
        
        filename = st.text_input("📁 Output filename", value=default_filename)
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        
        if st.button("📥 Download & Create PDF", type="primary", use_container_width=True):
            progress_bar = st.progress(0, text="Starting...")
            status_text = st.empty()
            
            def update_progress(progress, text):
                progress_bar.progress(progress, text=text)
                status_text.text(text)
            
            try:
                images = extract_images_with_browser(url, update_progress)
                
                update_progress(0.9, f"Creating PDF with {len(images)} pages...")
                
                service = PDFService()
                pdf_bytes = service.create_pdf_from_images(
                    images,
                    quality=quality,
                    max_dimension=max_dimension,
                )
                
                update_progress(1.0, "Done!")
                
                pdf_size_mb = len(pdf_bytes) / (1024 * 1024)
                st.success(f"✅ Successfully created PDF with {len(images)} pages ({pdf_size_mb:.1f} MB)")
                
                with st.expander("📖 Preview first page"):
                    st.image(images[0], width=300)
                
                st.download_button(
                    label="📥 Download PDF",
                    data=pdf_bytes,
                    file_name=filename,
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True,
                )
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.exception(e)
    
    st.divider()
    st.caption("""
    **Note:** This tool uses a headless browser to load JavaScript-rendered pages.
    It may take a moment to start. Please respect copyright laws.
    """)

with tab4:
    st.info("More tools coming soon!")
