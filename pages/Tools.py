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
    st.markdown("Upload multiple PDF files to combine them into a single document.")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files", 
        type="pdf", 
        accept_multiple_files=True,
        help="Select multiple files to merge. They will be merged in the order shown."
    )
    
    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} files:")
        # Show list of files to confirm order
        for i, file in enumerate(uploaded_files):
            st.text(f"{i+1}. {file.name}")
            
        if st.button("Merge PDFs", type="primary"):
            with st.spinner("Merging PDFs..."):
                try:
                    # Read all files
                    pdf_bytes_list = [f.read() for f in uploaded_files]
                    
                    # Merge
                    service = PDFService()
                    merged_pdf = service.merge_pdfs(pdf_bytes_list)
                    
                    st.success("✅ PDFs merged successfully!")
                    
                    # Default filename based on first file
                    first_name = uploaded_files[0].name.rsplit(".", 1)[0]
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
    st.markdown("Reduce the file size of your PDF documents.")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file to compress",
        type="pdf",
        help="Upload a PDF file to reduce its size"
    )
    
    if uploaded_file:
        # Show original file info
        original_bytes = uploaded_file.read()
        original_size = len(original_bytes)
        original_size_mb = original_size / (1024 * 1024)
        
        st.info(f"📄 Original file: **{uploaded_file.name}** ({original_size_mb:.2f} MB)")
        
        if st.button("Shrink PDF", type="primary"):
            with st.spinner("Compressing PDF..."):
                try:
                    # Compress
                    service = PDFService()
                    compressed_pdf = service.compress_pdf(original_bytes)
                    
                    # Calculate stats
                    compressed_size = len(compressed_pdf)
                    compressed_size_mb = compressed_size / (1024 * 1024)
                    reduction = ((original_size - compressed_size) / original_size) * 100
                    
                    st.success("✅ PDF compressed successfully!")
                    
                    # Show stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Size", f"{original_size_mb:.2f} MB")
                    with col2:
                        st.metric("Compressed Size", f"{compressed_size_mb:.2f} MB")
                    with col3:
                        st.metric("Reduction", f"{reduction:.1f}%")
                    
                    # Default filename
                    base_name = uploaded_file.name.rsplit(".", 1)[0]
                    default_name = f"{base_name}_compressed.pdf"
                    
                    st.download_button(
                        label="📥 Download Compressed PDF",
                        data=compressed_pdf,
                        file_name=default_name,
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True,
                    )
                    
                except Exception as e:
                    st.error(f"Failed to compress PDF: {e}")

with tab3:
    st.info("More tools coming soon!")
