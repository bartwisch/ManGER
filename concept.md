1. Project Overview
Build a production-grade, modular, and robust Manga Translation application from scratch. The core philosophy is Separation of Concerns, Type Safety, and Reproducibility. The application will use Magi (The Manga Whisperer) as the primary OCR and detection engine.

2. Core Philosophy & Improvements
Unlike the previous iteration, this version will address specific pain points:

Coordinate System: Use normalized coordinates (0.0-1.0) internally to be resolution-independent, converting to absolute pixels only for rendering.
Type Safety: Use Pydantic for all data exchange (no loose dictionaries or tuples).
Modularity: The UI (Streamlit) should be completely decoupled from the logic. The core logic should be runnable as a CLI or API.
Error Handling: Proper logging and exception hierarchies instead of silent failures or print statements.
Configuration: Environment-based and file-based configuration using pydantic-settings.
3. Architecture
3.1. Layered Design
Domain Layer: Data models (Page, Panel, TextBlock) and business rules.
Service Layer: Interfaces and Implementations for OCR, Translation, and Image Processing.
Application Layer: Pipelines that orchestrate services (e.g., TranslationPipeline).
Interface Layer: Streamlit UI, CLI, or REST API.
3.2. Data Models (Pydantic)
from pydantic import BaseModel
from typing import List, Tuple
class BoundingBox(BaseModel):
    x_min: float  # Normalized 0-1
    y_min: float
    x_max: float
    y_max: float
    
    def to_pixels(self, width: int, height: int) -> Tuple[int, int, int, int]:
        # Helper to convert to pixels
        pass
class TextBlock(BaseModel):
    id: str
    bbox: BoundingBox
    original_text: str
    translated_text: str | None = None
    confidence: float
    speaker_id: int | None = None  # From Magi grouping
class MangaPage(BaseModel):
    page_number: int
    image_path: str
    resolution: Tuple[int, int]
    text_blocks: List[TextBlock]
4. Core Components
4.1. OCR Service (MagiService)
Responsibility: Wraps the Magi model.
Input: PIL.Image or np.ndarray.
Output: List of TextBlock objects.
Key Feature: Handles image resizing internally but always maps coordinates back to the original image's normalized space immediately. This eliminates "offset" bugs.
Optimization: Implements LRU caching for model loading and inference results.
4.2. Translation Service (Translator)
Interface: translate(text: str, context: str) -> str
Implementations:
OpenAITranslator: Uses GPT-4o.
DeepLTranslator: Uses DeepL API.
DummyTranslator: For testing without costs.
Robustness: Handles rate limits, retries, and partial failures.
4.3. Image Processing Service (Compositor)
Responsibility: In-painting (cleaning text bubbles) and Typesetting (drawing text).
Features:
Smart In-painting: Uses OpenCV or simple color filling to remove original text based on the bounding box.
Dynamic Typesetting: Calculates optimal font size to fit the BoundingBox. Supports text wrapping and vertical centering.
Bubble Analysis: Optional "Bubble Contour" detection to refine the rectangular box from Magi into a shape for better erasing.
4.4. Pipeline (MangaPipeline)
Orchestrates the flow:
Load Image
OCR (Magi) -> Get TextBlocks
Filter (Confidence check)
Group (Merge close blocks if needed)
Translate (Batch translation for context)
Render (Create final image)
5. Technology Stack
Language: Python 3.10+
Data Validation: Pydantic v2
Computer Vision: OpenCV, Pillow, PyTorch (for Magi)
UI: Streamlit (kept simple, acting only as a view)
Config: pydantic-settings
Logging: loguru (for structured, pretty logging)
6. Implementation Steps for LLM
Setup: Initialize project with poetry or uv. Define pyproject.toml.
Domain: Create src/domain/models.py.
OCR: Implement src/services/ocr.py with Magi. Ensure coordinate normalization tests are written first.
Translation: Implement src/services/translator.py.
Rendering: Implement src/services/renderer.py.
Pipeline: Create src/pipeline.py to glue it together.
UI: Build 
app.py
 that imports MangaPipeline.
7. Critical "Robustness" Rules
No Magic Numbers: All thresholds (confidence, padding) must be in a config object.
Fail Gracefully: If OCR fails on one page, the pipeline should log it and continue to the next, not crash the app.
Traceability: Every TextBlock should track its origin (raw OCR result) to debug offsets easily.
Resolution Independence: Never pass absolute pixel coordinates between services. Always use normalized (0-1) or pass the reference image size explicitly.