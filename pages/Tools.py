import streamlit as st
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from manger.services.pdf import PDFService, PDFError

st.set_page_config(
    page_title="Tools - ManGER",
    page_icon="🛠️",
    layout="wide",
)

st.title("🛠️ Tools")

tab1, tab2, tab3 = st.tabs(["📄 PDF Combiner", "📉 PDF Shrinker", "🚧 More Tools"])

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
    st.info("More tools coming soon!")
