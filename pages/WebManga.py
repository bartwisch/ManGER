import streamlit as st
import sys
import re
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from manger.services.pdf import PDFService

st.set_page_config(
    page_title="Web Manga to PDF - ManGER",
    page_icon="🌐",
    layout="wide",
)

col1, col2 = st.columns([4, 1])
with col1:
    st.title("🌐 Web Manga to PDF")
with col2:
    if st.button("📖 Back to Translator", use_container_width=True):
        st.switch_page("app.py")

st.markdown("""
Download manga chapters from websites and convert them to PDF.

**✨ Uses a real browser** to load JavaScript-rendered pages and print to PDF.
""")


def download_page_as_pdf(url: str, progress_callback=None) -> bytes:
    """Load a webpage and print it as PDF using Playwright."""
    from playwright.sync_api import sync_playwright
    from loguru import logger
    
    logger.info(f"Starting download for: {url}")
    
    if progress_callback:
        progress_callback(0.05, "Starting browser...")
    
    with sync_playwright() as p:
        # Launch headless browser
        logger.debug("Launching Chromium...")
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1200, "height": 800},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        
        if progress_callback:
            progress_callback(0.1, "Loading page...")
        
        # Navigate to URL - use domcontentloaded instead of networkidle (faster, more reliable)
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
                    // Timeout after 15 seconds
                    setTimeout(() => { clearInterval(timer); resolve(); }, 15000);
                });
            }
        """)
        
        if progress_callback:
            progress_callback(0.5, "Waiting for images to load...")
        
        # Wait for images to load after scrolling
        logger.debug("Waiting for images to load...")
        time.sleep(3)
        
        # Scroll back to top
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(0.5)
        
        if progress_callback:
            progress_callback(0.6, "Hiding non-essential elements...")
        
        logger.debug("Hiding non-content elements...")
        # Try to hide common non-content elements (headers, footers, ads, navigation)
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
                // Also remove fixed/sticky positioning that might cause issues
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
        # Print to PDF
        pdf_bytes = page.pdf(
            format="A4",
            print_background=True,
            margin={"top": "10mm", "bottom": "10mm", "left": "5mm", "right": "5mm"},
        )
        
        logger.info(f"Browser PDF generated: {len(pdf_bytes) / 1024 / 1024:.1f} MB")
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


# UI
url = st.text_input(
    "🔗 Manga Chapter URL",
    placeholder="https://example.com/manga/title/chapter/1",
    help="Paste the URL of a manga chapter page",
)

# Compression settings
st.subheader("⚙️ Compression Settings")
col1, col2 = st.columns(2)
with col1:
    quality = st.slider(
        "JPEG Quality",
        min_value=40,
        max_value=90,
        value=70,
        step=5,
        help="Lower = smaller file. 60-70 is usually good for manga.",
        key="webmanga_quality",
    )
with col2:
    max_dimension = st.slider(
        "Max Image Size (px)",
        min_value=800,
        max_value=2000,
        value=1400,
        step=100,
        help="Maximum width/height. 1200-1400 is readable.",
        key="webmanga_maxdim",
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

# Auto-translate option
auto_translate = st.checkbox(
    "🔄 Auto-translate after scraping",
    value=False,
    help="Automatically load the PDF into the translator and translate all pages",
)

# Button always visible, disabled if no URL
if st.button("📥 Download & Create PDF", type="primary", use_container_width=True, disabled=not url):
    progress_bar = st.progress(0, text="Starting...")
    status_text = st.empty()  # For detailed status updates
    
    def update_progress(progress, text):
        progress_bar.progress(min(progress, 0.99), text=text)
        status_text.info(f"🔄 {text}")
    
    try:
        # Step 1: Download page as PDF
        raw_pdf_bytes = download_page_as_pdf(url, update_progress)
        raw_size_mb = len(raw_pdf_bytes) / (1024 * 1024)
        
        # Step 2: Compress the PDF using our shrinker
        update_progress(0.85, f"Compressing PDF ({raw_size_mb:.1f} MB)...")
        
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
        
        st.success(f"✅ PDF created: {pdf_size_mb:.1f} MB (compressed from {raw_size_mb:.1f} MB, -{reduction:.0f}%)")
        
        if auto_translate:
            # Store PDF in session state for translator
            st.session_state.scraped_pdf_bytes = pdf_bytes
            st.session_state.scraped_pdf_filename = filename
            st.session_state.auto_translate_scraped = True
            st.info("🔄 Redirecting to translator...")
            st.switch_page("app.py")
        else:
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

st.divider()
st.caption("""
**Note:** This tool uses a headless browser to print the page as PDF.
Headers, footers, and ads are automatically hidden. Please respect copyright laws.
""")
