# ManGER - Manga Translation Application

A production-grade, modular manga translation application using AI-powered OCR and translation.

## Features

- **PDF Upload Support**: Upload manga PDFs and select specific pages to translate
- **Page Selection**: Visual page selector with thumbnails for easy navigation
- **Batch Processing**: Process multiple pages at once
- **Modular Architecture**: Separation of concerns with Domain, Service, Application, and Interface layers
- **Type Safety**: Full Pydantic v2 validation for all data models
- **Resolution Independence**: Normalized coordinates (0.0-1.0) for consistent processing across image sizes
- **Multiple Translation Backends**: Support for OpenAI, DeepL (planned), and dummy translator for testing
- **Smart Rendering**: Automatic text sizing and bubble detection
- **Streamlit UI**: Easy-to-use web interface

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or uv package manager

### Quick Start

1. **Clone the repository**
   ```bash
   cd ManGER
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e .
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** to http://localhost:8501

## Project Structure

```
ManGER/
├── app.py                      # Streamlit UI entry point
├── pyproject.toml              # Project configuration
├── concept.md                  # Design document
├── README.md                   # This file
└── src/
    └── manger/
        ├── __init__.py
        ├── config.py           # Configuration management
        ├── pipeline.py         # Main orchestration pipeline
        ├── domain/
        │   ├── __init__.py
        │   └── models.py       # Pydantic data models
        └── services/
            ├── __init__.py
            ├── ocr.py          # OCR service (Magi integration)
            ├── translator.py   # Translation services
            ├── renderer.py     # Image rendering/typesetting
            └── pdf.py          # PDF handling service
```

## Usage

### Web Interface

1. Launch the app with `streamlit run app.py`
2. **Upload a PDF** - Drag and drop or browse for a manga PDF file
3. **Select Pages** - Use the visual page selector to choose which pages to translate
   - Click thumbnails to select/deselect individual pages
   - Use "Select All", "Clear Selection", "Odd Pages", or "Even Pages" buttons
4. **Process Pages** - Use the Page Viewer tab for individual pages or Batch Processing for all selected pages
   - **Detect Text**: Run OCR to find text blocks
   - **Translate**: Translate detected text
   - **Render**: Generate the final image with translated text
5. **Download** - Download processed pages

### Programmatic Usage

```python
from manger.pipeline import MangaPipeline
from manger.services.pdf import PDFService
from manger.config import get_config

# Load a PDF
pdf_service = PDFService(dpi=200)
page_count = pdf_service.load("manga.pdf")
print(f"PDF has {page_count} pages")

# Get specific pages as images
for page_num in [0, 1, 2]:
    image = pdf_service.get_page_image(page_num)
    
    # Process with pipeline
    pipeline = MangaPipeline()
    result = pipeline.process_image(image, f"output/page_{page_num}.jpg")
    
    if result.success:
        print(f"Page {page_num}: {len(result.page.text_blocks)} text blocks")
```

### Configuration

Configuration can be set via environment variables or a `.env` file:

```bash
# Translation settings
MANGER_TRANSLATE_PROVIDER=openai
MANGER_TRANSLATE_OPENAI_API_KEY=your-api-key
MANGER_TRANSLATE_SOURCE_LANGUAGE=ja
MANGER_TRANSLATE_TARGET_LANGUAGE=en

# OCR settings
MANGER_OCR_CONFIDENCE_THRESHOLD=0.5

# Render settings
MANGER_RENDER_DEFAULT_FONT_SIZE=14

# App settings
MANGER_DEBUG=false
MANGER_LOG_LEVEL=INFO
```

## Architecture

### Domain Layer

- **BoundingBox**: Normalized coordinates (0-1) for resolution independence
- **TextBlock**: OCR results with original text, translation, and metadata
- **MangaPage**: Container for all text blocks in a page

### Service Layer

- **PDFService**: Handles PDF loading, page extraction, and thumbnail generation
- **OCRService**: Abstract interface for OCR backends (Magi, dummy)
- **Translator**: Abstract interface for translation (OpenAI, DeepL, dummy)
- **Renderer**: In-painting and typesetting

### Application Layer

- **MangaPipeline**: Orchestrates the full translation workflow

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest
```

### Code Style

```bash
ruff check src/
ruff format src/
```

## Current Limitations

- Uses dummy OCR (generates random text blocks for testing)
- Actual Magi integration requires `pip install manger[magi]`
- DeepL translator not yet implemented
- Vertical text rendering is basic
- Multi-page PDF download not yet implemented

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request