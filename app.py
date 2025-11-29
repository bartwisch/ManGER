"""Streamlit UI for ManGER - Manga Translation Application.

This is the main entry point for the web interface.
Run with: streamlit run app.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
from PIL import Image
import io
from loguru import logger

from manger.config import get_config, AppConfig
from manger.pipeline import MangaPipeline
from manger.services.ocr import DummyOCRService
from manger.services.translator import DummyTranslator
from manger.services.renderer import Renderer
from manger.services.pdf import PDFService, PDFError
from manger.domain.models import TextBlock, MangaPage

# Configure logger for Streamlit
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
)


# Page configuration
st.set_page_config(
    page_title="ManGER - Manga Translator",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    """Initialize session state variables."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "pdf_service" not in st.session_state:
        st.session_state.pdf_service = PDFService()
    if "pdf_loaded" not in st.session_state:
        st.session_state.pdf_loaded = False
    if "pdf_page_count" not in st.session_state:
        st.session_state.pdf_page_count = 0
    if "selected_pages" not in st.session_state:
        st.session_state.selected_pages = []
    if "page_thumbnails" not in st.session_state:
        st.session_state.page_thumbnails = []
    if "current_page_idx" not in st.session_state:
        st.session_state.current_page_idx = 0
    if "pages_data" not in st.session_state:
        st.session_state.pages_data = {}  # {page_num: {"image": img, "blocks": [], "result": None}}
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False


def get_pipeline() -> MangaPipeline:
    """Get or create the pipeline instance."""
    if st.session_state.pipeline is None:
        config = get_config()
        st.session_state.pipeline = MangaPipeline(
            config=config,
            ocr_service=DummyOCRService(config.ocr),
            translator=DummyTranslator(config.translation),
            renderer=Renderer(config.render),
        )
    return st.session_state.pipeline


def render_sidebar():
    """Render the sidebar with settings."""
    st.sidebar.title("⚙️ Settings")
    
    st.sidebar.subheader("OCR Settings")
    confidence = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence score to keep a text block",
    )
    
    st.sidebar.subheader("Translation Settings")
    provider = st.sidebar.selectbox(
        "Translation Provider",
        options=["dummy", "openai"],
        index=0,
        help="Select the translation backend",
    )
    
    source_lang = st.sidebar.selectbox(
        "Source Language",
        options=["ja", "ko", "zh"],
        index=0,
        format_func=lambda x: {"ja": "Japanese", "ko": "Korean", "zh": "Chinese"}[x],
    )
    
    target_lang = st.sidebar.selectbox(
        "Target Language",
        options=["en", "de", "fr", "es"],
        index=0,
        format_func=lambda x: {
            "en": "English",
            "de": "German",
            "fr": "French",
            "es": "Spanish",
        }[x],
    )
    
    st.sidebar.subheader("Render Settings")
    font_size = st.sidebar.slider(
        "Default Font Size",
        min_value=8,
        max_value=48,
        value=14,
        help="Default font size for rendered text",
    )
    
    render_dpi = st.sidebar.slider(
        "PDF Render DPI",
        min_value=72,
        max_value=300,
        value=150,
        step=10,
        help="DPI for rendering PDF pages (higher = better quality but slower)",
    )
    
    st.sidebar.divider()
    
    st.sidebar.info(
        "**Note:** This demo uses a dummy OCR and translator. "
        "For real translations, configure the OpenAI API key."
    )
    
    return {
        "confidence": confidence,
        "provider": provider,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "font_size": font_size,
        "render_dpi": render_dpi,
    }


def render_text_blocks(blocks: list[TextBlock], page_num: int = 0):
    """Render the text blocks table."""
    if not blocks:
        st.info("No text blocks detected yet. Click 'Detect Text' to process.")
        return
    
    st.subheader(f"📝 Text Blocks - Page {page_num + 1} ({len(blocks)} blocks)")
    
    # Create editable table data
    data = []
    for block in blocks:
        data.append({
            "ID": block.id,
            "Original": block.original_text,
            "Translation": block.translated_text or "",
            "Confidence": f"{block.confidence:.2%}",
            "Vertical": "✓" if block.is_vertical else "",
        })
    
    # Display as dataframe
    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True,
    )


def render_page_selector():
    """Render the page selection grid."""
    if not st.session_state.pdf_loaded:
        return
    
    st.subheader("📄 Select Pages to Translate")
    
    page_count = st.session_state.pdf_page_count
    
    # Quick selection buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Select All", use_container_width=True):
            st.session_state.selected_pages = list(range(page_count))
            st.rerun()
    with col2:
        if st.button("Clear Selection", use_container_width=True):
            st.session_state.selected_pages = []
            st.rerun()
    with col3:
        if st.button("Select Odd Pages", use_container_width=True):
            st.session_state.selected_pages = list(range(0, page_count, 2))
            st.rerun()
    with col4:
        if st.button("Select Even Pages", use_container_width=True):
            st.session_state.selected_pages = list(range(1, page_count, 2))
            st.rerun()
    
    st.write(f"**Selected:** {len(st.session_state.selected_pages)} of {page_count} pages")
    
    # Page thumbnail grid
    thumbnails = st.session_state.page_thumbnails
    if thumbnails:
        cols_per_row = 6
        for row_start in range(0, len(thumbnails), cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx, (page_num, thumb) in enumerate(
                thumbnails[row_start:row_start + cols_per_row]
            ):
                with cols[col_idx]:
                    is_selected = page_num in st.session_state.selected_pages
                    
                    # Show thumbnail
                    st.image(thumb, use_container_width=True)
                    
                    # Checkbox for selection
                    if st.checkbox(
                        f"Page {page_num + 1}",
                        value=is_selected,
                        key=f"page_select_{page_num}",
                    ):
                        if page_num not in st.session_state.selected_pages:
                            st.session_state.selected_pages.append(page_num)
                            st.session_state.selected_pages.sort()
                    else:
                        if page_num in st.session_state.selected_pages:
                            st.session_state.selected_pages.remove(page_num)


def render_page_viewer():
    """Render the page viewer for selected pages."""
    if not st.session_state.selected_pages:
        st.info("Select pages to view and translate.")
        return
    
    selected = st.session_state.selected_pages
    current_idx = st.session_state.current_page_idx
    
    if current_idx >= len(selected):
        st.session_state.current_page_idx = 0
        current_idx = 0
    
    current_page_num = selected[current_idx]
    
    # Navigation
    st.subheader(f"📖 Page {current_page_num + 1}")
    
    nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 1, 2, 1, 1])
    
    with nav_col1:
        if st.button("⏮️ First", disabled=current_idx == 0, use_container_width=True):
            st.session_state.current_page_idx = 0
            st.rerun()
    
    with nav_col2:
        if st.button("◀️ Prev", disabled=current_idx == 0, use_container_width=True):
            st.session_state.current_page_idx = current_idx - 1
            st.rerun()
    
    with nav_col3:
        st.write(f"Page {current_idx + 1} of {len(selected)} selected")
    
    with nav_col4:
        if st.button("Next ▶️", disabled=current_idx >= len(selected) - 1, use_container_width=True):
            st.session_state.current_page_idx = current_idx + 1
            st.rerun()
    
    with nav_col5:
        if st.button("Last ⏭️", disabled=current_idx >= len(selected) - 1, use_container_width=True):
            st.session_state.current_page_idx = len(selected) - 1
            st.rerun()
    
    # Load page image if not cached
    if current_page_num not in st.session_state.pages_data:
        with st.spinner(f"Loading page {current_page_num + 1}..."):
            try:
                img = st.session_state.pdf_service.get_page_image(current_page_num)
                st.session_state.pages_data[current_page_num] = {
                    "image": img,
                    "blocks": [],
                    "result": None,
                }
            except PDFError as e:
                st.error(f"Failed to load page: {e}")
                return
    
    page_data = st.session_state.pages_data[current_page_num]
    
    # Display original and processed images side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Original**")
        st.image(page_data["image"], use_container_width=True)
    
    with col2:
        st.write("**Translated**")
        if page_data["result"]:
            st.image(page_data["result"], use_container_width=True)
        else:
            st.info("Process page to see translation")
    
    # Action buttons for current page
    st.divider()
    
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    
    with btn_col1:
        if st.button("🔍 Detect Text", key=f"detect_{current_page_num}", use_container_width=True):
            with st.spinner("Detecting text..."):
                pipeline = get_pipeline()
                blocks = pipeline.ocr.process_image(page_data["image"])
                page_data["blocks"] = blocks
                st.success(f"Detected {len(blocks)} text blocks!")
                st.rerun()
    
    with btn_col2:
        if st.button(
            "🌐 Translate",
            key=f"translate_{current_page_num}",
            disabled=len(page_data["blocks"]) == 0,
            use_container_width=True,
        ):
            with st.spinner("Translating..."):
                pipeline = get_pipeline()
                texts = [b.original_text for b in page_data["blocks"]]
                translations = pipeline.translator.translate_batch(texts)
                
                for block, trans in zip(page_data["blocks"], translations):
                    block.translated_text = trans
                
                st.success("Translation complete!")
                st.rerun()
    
    with btn_col3:
        if st.button(
            "🎨 Render",
            key=f"render_{current_page_num}",
            disabled=len(page_data["blocks"]) == 0,
            use_container_width=True,
        ):
            with st.spinner("Rendering..."):
                pipeline = get_pipeline()
                translated_blocks = [
                    b for b in page_data["blocks"] if b.translated_text
                ]
                
                if translated_blocks:
                    result = pipeline.renderer.inpaint(page_data["image"], translated_blocks)
                    result = pipeline.renderer.typeset(result, translated_blocks)
                    page_data["result"] = result
                    st.success("Rendering complete!")
                    st.rerun()
                else:
                    st.warning("No translated blocks. Please translate first.")
    
    # Show text blocks
    if page_data["blocks"]:
        render_text_blocks(page_data["blocks"], current_page_num)


def render_batch_processing():
    """Render batch processing controls."""
    if not st.session_state.selected_pages:
        return
    
    st.divider()
    st.subheader("🚀 Batch Processing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(
            "Process All Selected Pages",
            type="primary",
            use_container_width=True,
        ):
            process_all_pages()
    
    with col2:
        # Check if any pages are processed
        processed_pages = [
            p for p in st.session_state.selected_pages
            if p in st.session_state.pages_data
            and st.session_state.pages_data[p].get("result") is not None
        ]
        
        if processed_pages:
            if st.button(
                f"📥 Download All ({len(processed_pages)} pages)",
                use_container_width=True,
            ):
                st.info("Download functionality requires additional implementation for multi-page PDFs.")


def process_all_pages():
    """Process all selected pages."""
    selected = st.session_state.selected_pages
    pipeline = get_pipeline()
    pdf_service = st.session_state.pdf_service
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, page_num in enumerate(selected):
        progress = (i + 1) / len(selected)
        progress_bar.progress(progress)
        status_text.text(f"Processing page {page_num + 1} ({i + 1}/{len(selected)})")
        
        # Load page if not cached
        if page_num not in st.session_state.pages_data:
            try:
                img = pdf_service.get_page_image(page_num)
                st.session_state.pages_data[page_num] = {
                    "image": img,
                    "blocks": [],
                    "result": None,
                }
            except PDFError as e:
                logger.error(f"Failed to load page {page_num}: {e}")
                continue
        
        page_data = st.session_state.pages_data[page_num]
        
        # Detect text
        if not page_data["blocks"]:
            blocks = pipeline.ocr.process_image(page_data["image"])
            page_data["blocks"] = blocks
        
        # Translate
        untranslated = [b for b in page_data["blocks"] if not b.translated_text]
        if untranslated:
            texts = [b.original_text for b in untranslated]
            translations = pipeline.translator.translate_batch(texts)
            for block, trans in zip(untranslated, translations):
                block.translated_text = trans
        
        # Render
        translated_blocks = [b for b in page_data["blocks"] if b.translated_text]
        if translated_blocks and not page_data["result"]:
            result = pipeline.renderer.inpaint(page_data["image"], translated_blocks)
            result = pipeline.renderer.typeset(result, translated_blocks)
            page_data["result"] = result
    
    progress_bar.progress(1.0)
    status_text.text("✅ Processing complete!")
    st.session_state.processing_complete = True
    st.rerun()


def main():
    """Main application entry point."""
    init_session_state()
    
    # Header
    st.title("📖 ManGER - Manga Translator")
    st.markdown(
        "Upload a PDF manga to detect and translate text. "
        "Select the pages you want to translate."
    )
    
    # Sidebar settings
    settings = render_sidebar()
    
    # Update PDF service DPI
    st.session_state.pdf_service.dpi = settings["render_dpi"]
    
    # File upload
    st.subheader("📤 Upload Manga PDF")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a manga PDF file",
    )
    
    if uploaded_file and not st.session_state.pdf_loaded:
        with st.spinner("Loading PDF..."):
            try:
                # Read the file bytes
                pdf_bytes = uploaded_file.read()
                page_count = st.session_state.pdf_service.load_from_bytes(pdf_bytes)
                st.session_state.pdf_loaded = True
                st.session_state.pdf_page_count = page_count
                
                # Generate thumbnails
                with st.spinner("Generating page previews..."):
                    thumbnails = st.session_state.pdf_service.get_all_thumbnails(
                        max_size=(120, 120)
                    )
                    st.session_state.page_thumbnails = thumbnails
                
                st.success(f"Loaded PDF with {page_count} pages!")
                st.rerun()
                
            except PDFError as e:
                st.error(f"Failed to load PDF: {e}")
    
    # Reset button
    if st.session_state.pdf_loaded:
        if st.button("🔄 Load New PDF"):
            st.session_state.pdf_service.close()
            st.session_state.pdf_loaded = False
            st.session_state.pdf_page_count = 0
            st.session_state.selected_pages = []
            st.session_state.page_thumbnails = []
            st.session_state.pages_data = {}
            st.session_state.current_page_idx = 0
            st.session_state.processing_complete = False
            st.rerun()
    
    # Page selection
    if st.session_state.pdf_loaded:
        render_page_selector()
        
        st.divider()
        
        # Tabs for different views
        tab1, tab2 = st.tabs(["📖 Page Viewer", "🚀 Batch Processing"])
        
        with tab1:
            render_page_viewer()
        
        with tab2:
            render_batch_processing()
            
            # Show results summary
            if st.session_state.processing_complete:
                st.subheader("📊 Results Summary")
                
                processed = 0
                failed = 0
                total_blocks = 0
                
                for page_num in st.session_state.selected_pages:
                    if page_num in st.session_state.pages_data:
                        page_data = st.session_state.pages_data[page_num]
                        if page_data.get("result"):
                            processed += 1
                            total_blocks += len(page_data["blocks"])
                        else:
                            failed += 1
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Pages Processed", processed)
                col2.metric("Pages Failed", failed)
                col3.metric("Total Text Blocks", total_blocks)
    
    # Footer
    st.divider()
    st.caption(
        "ManGER v0.1.0 | "
        "[GitHub](https://github.com/example/manger) | "
        "Built with Streamlit"
    )


if __name__ == "__main__":
    main()