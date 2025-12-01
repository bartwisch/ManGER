"""Streamlit UI for ManGER - Manga Translation Application.

This is the main entry point for the web interface.
Run with: streamlit run app.py
"""

import sys
import os
import base64
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
from streamlit.components.v1 import html
from PIL import Image, ImageDraw, ImageFont
import io
from loguru import logger

# .env file path
ENV_FILE = Path(__file__).parent / ".env"

# JavaScript for localStorage API key management
LOCAL_STORAGE_JS = """
<script>
// Load API key from localStorage on page load
(function() {
    const key = localStorage.getItem('manger_api_key');
    if (key && window.parent) {
        // Send to Streamlit via query params workaround
        window.parent.postMessage({type: 'manger_api_key', key: key}, '*');
    }
})();

// Function to save API key to localStorage
function saveApiKey(key) {
    if (key) {
        localStorage.setItem('manger_api_key', key);
    } else {
        localStorage.removeItem('manger_api_key');
    }
}

// Listen for save requests from Streamlit
window.addEventListener('message', function(event) {
    if (event.data && event.data.type === 'save_manger_api_key') {
        saveApiKey(event.data.key);
    }
});
</script>
"""

# Notification sound (base64 encoded short beep)
NOTIFICATION_SOUND_HTML = """
<script>
(function() {
    try {
        var audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        
        // First beep
        var oscillator = audioCtx.createOscillator();
        var gainNode = audioCtx.createGain();
        oscillator.connect(gainNode);
        gainNode.connect(audioCtx.destination);
        oscillator.frequency.value = 880;
        oscillator.type = 'sine';
        gainNode.gain.value = 0.5;
        oscillator.start();
        oscillator.stop(audioCtx.currentTime + 0.15);
        
        // Second beep (higher pitch)
        setTimeout(function() {
            var oscillator2 = audioCtx.createOscillator();
            var gainNode2 = audioCtx.createGain();
            oscillator2.connect(gainNode2);
            gainNode2.connect(audioCtx.destination);
            oscillator2.frequency.value = 1100;
            oscillator2.type = 'sine';
            gainNode2.gain.value = 0.5;
            oscillator2.start();
            oscillator2.stop(audioCtx.currentTime + 0.2);
        }, 180);
    } catch(e) {
        console.log('Audio not supported:', e);
    }
})();
</script>
"""

# JavaScript to scroll down immediately
SCROLL_DOWN_JS = """
<script>
    window.parent.document.querySelector('section.main').scrollTo({top: 99999, behavior: 'instant'});
</script>
"""


def play_notification_sound():
    """Schedule notification sound to play on next render."""
    st.session_state._play_sound_pending = True


def scroll_to_results():
    """Schedule scroll to results on next render."""
    st.session_state._scroll_to_results_pending = True


def _render_pending_sound():
    """Render the sound if pending."""
    if st.session_state.get("_play_sound_pending", False):
        st.session_state._play_sound_pending = False
        st.components.v1.html(NOTIFICATION_SOUND_HTML, height=0)


def _render_pending_scroll():
    """Render scroll if pending."""
    if st.session_state.get("_scroll_to_results_pending", False):
        st.session_state._scroll_to_results_pending = False
        st.components.v1.html(SCROLL_DOWN_JS, height=0)


def load_api_key_from_env() -> str:
    """Load API key from .env file or environment."""
    # First check environment variable
    api_key = os.environ.get("MANGER_TRANSLATE_OPENAI_API_KEY", "")
    if api_key:
        return api_key
    
    # Then check .env file
    if ENV_FILE.exists():
        try:
            with open(ENV_FILE, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("MANGER_TRANSLATE_OPENAI_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
    return ""


def save_api_key_to_env(api_key: str) -> None:
    """Save API key to .env file."""
    lines = []
    key_found = False
    
    # Read existing .env if it exists
    if ENV_FILE.exists():
        try:
            with open(ENV_FILE, "r") as f:
                for line in f:
                    if line.strip().startswith("MANGER_TRANSLATE_OPENAI_API_KEY="):
                        if api_key:  # Only write if we have a key
                            lines.append(f'MANGER_TRANSLATE_OPENAI_API_KEY="{api_key}"\n')
                        key_found = True
                    else:
                        lines.append(line)
        except Exception:
            pass
    
    # Add key if not found and we have one
    if not key_found and api_key:
        lines.append(f'MANGER_TRANSLATE_OPENAI_API_KEY="{api_key}"\n')
    
    # Write back
    if lines or api_key:
        try:
            with open(ENV_FILE, "w") as f:
                f.writelines(lines)
        except Exception as e:
            logger.warning(f"Failed to save API key: {e}")


def smooth_polygon_for_display(polygon: list[tuple[int, int]], num_points: int = 100) -> list[tuple[int, int]]:
    """Smooth a polygon using Catmull-Rom spline interpolation for display.
    
    This creates a smooth curve through all polygon points, eliminating
    harsh 90-degree corners that would break the speech bubble appearance.
    
    Args:
        polygon: List of (x, y) points
        num_points: Number of points in the smoothed output
        
    Returns:
        Smoothed polygon with many more points for a curved appearance
    """
    if len(polygon) < 4:
        return polygon
    
    import math
    
    # Close the polygon by wrapping points
    pts = list(polygon) + [polygon[0], polygon[1], polygon[2]]
    
    result = []
    n = len(polygon)
    points_per_segment = max(3, num_points // n)
    
    for i in range(n):
        # Four control points for Catmull-Rom
        p0 = pts[i]
        p1 = pts[i + 1]
        p2 = pts[i + 2]
        p3 = pts[i + 3]
        
        # Generate points along the spline segment
        for j in range(points_per_segment):
            t = j / points_per_segment
            t2 = t * t
            t3 = t2 * t
            
            # Catmull-Rom spline formula
            x = 0.5 * (
                (2 * p1[0]) +
                (-p0[0] + p2[0]) * t +
                (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                (2 * p1[1]) +
                (-p0[1] + p2[1]) * t +
                (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
            )
            result.append((int(x), int(y)))
    
    return result


def group_text_blocks(blocks: list, distance_threshold: float = 0.05) -> list[list[int]]:
    """Group text blocks that are close together (likely in same speech bubble).
    
    Args:
        blocks: List of TextBlock objects
        distance_threshold: Maximum normalized distance to consider blocks as connected
        
    Returns:
        List of groups, where each group is a list of block indices
    """
    if not blocks:
        return []
    
    n = len(blocks)
    # Track which group each block belongs to
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Check each pair of blocks for proximity
    for i in range(n):
        for j in range(i + 1, n):
            bbox_i = blocks[i].bbox
            bbox_j = blocks[j].bbox
            
            # Calculate distance between boxes (minimum edge distance)
            # Horizontal distance
            if bbox_i.x_max < bbox_j.x_min:
                dx = bbox_j.x_min - bbox_i.x_max
            elif bbox_j.x_max < bbox_i.x_min:
                dx = bbox_i.x_min - bbox_j.x_max
            else:
                dx = 0  # Overlapping horizontally
            
            # Vertical distance
            if bbox_i.y_max < bbox_j.y_min:
                dy = bbox_j.y_min - bbox_i.y_max
            elif bbox_j.y_max < bbox_i.y_min:
                dy = bbox_i.y_min - bbox_j.y_max
            else:
                dy = 0  # Overlapping vertically
            
            # If close enough, group them
            distance = (dx ** 2 + dy ** 2) ** 0.5
            if distance < distance_threshold:
                union(i, j)
    
    # Build groups
    groups_dict = {}
    for i in range(n):
        root = find(i)
        if root not in groups_dict:
            groups_dict[root] = []
        groups_dict[root].append(i)
    
    # Only return groups with more than one block
    return [g for g in groups_dict.values() if len(g) > 1]


def draw_text_boxes(image: Image.Image, blocks: list, show_text: bool = True, show_polygons: bool = False) -> Image.Image:
    """Draw bounding boxes around detected text blocks.
    
    Args:
        image: Original PIL Image
        blocks: List of TextBlock objects with bounding boxes
        show_text: Whether to show the detected text above boxes
        show_polygons: Whether to show precise text polygons instead of rectangles
    
    Returns:
        Image with drawn bounding boxes
    """
    # Create a copy to draw on
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    
    width, height = image.size
    
    # Colors for different states
    colors = {
        "detected": (255, 0, 0),      # Red for detected only
        "translated": (0, 255, 0),    # Green for translated
        "group": (0, 100, 255),       # Blue for group boxes
        "polygon": (255, 0, 255),     # Pink/Magenta for text polygons
    }
    
    # First, draw group boxes (so they appear behind individual boxes)
    groups = group_text_blocks(blocks)
    for group_idx, group in enumerate(groups):
        # Calculate bounding box that encompasses all blocks in group
        min_x = min(blocks[i].bbox.x_min for i in group)
        min_y = min(blocks[i].bbox.y_min for i in group)
        max_x = max(blocks[i].bbox.x_max for i in group)
        max_y = max(blocks[i].bbox.y_max for i in group)
        
        # Add some padding
        padding = 0.01
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(1, max_x + padding)
        max_y = min(1, max_y + padding)
        
        # Convert to pixels
        gx1 = int(min_x * width)
        gy1 = int(min_y * height)
        gx2 = int(max_x * width)
        gy2 = int(max_y * height)
        
        # Draw dashed rectangle for group
        group_color = colors["group"]
        dash_length = 10
        gap_length = 5
        
        # Top edge
        x = gx1
        while x < gx2:
            draw.line([(x, gy1), (min(x + dash_length, gx2), gy1)], fill=group_color, width=3)
            x += dash_length + gap_length
        
        # Bottom edge
        x = gx1
        while x < gx2:
            draw.line([(x, gy2), (min(x + dash_length, gx2), gy2)], fill=group_color, width=3)
            x += dash_length + gap_length
        
        # Left edge
        y = gy1
        while y < gy2:
            draw.line([(gx1, y), (gx1, min(y + dash_length, gy2))], fill=group_color, width=3)
            y += dash_length + gap_length
        
        # Right edge
        y = gy1
        while y < gy2:
            draw.line([(gx2, y), (gx2, min(y + dash_length, gy2))], fill=group_color, width=3)
            y += dash_length + gap_length
        
        # Draw group label
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except:
            font = ImageFont.load_default()
        
        group_label = f"Gruppe {group_idx + 1}"
        label_bbox = draw.textbbox((gx1, gy1 - 25), group_label, font=font)
        draw.rectangle([label_bbox[0] - 2, label_bbox[1] - 2, label_bbox[2] + 2, label_bbox[3] + 2], fill=group_color)
        draw.text((gx1, gy1 - 25), group_label, fill=(255, 255, 255), font=font)
    
    # Draw polygons if requested (outline only, no fill)
    if show_polygons:
        renderer = Renderer()
        # Same shift values as for green OCR boxes
        shift_left = 12
        expand_width = 8
        for i, block in enumerate(blocks):
            polygon = renderer.extract_text_polygon(image, block.bbox)
            if len(polygon) >= 3:
                # Shift and expand polygon to match green OCR boxes
                shifted_polygon = []
                for (px, py) in polygon:
                    # Shift left and expand width (approximate based on bbox center)
                    bbox = block.bbox
                    bbox_center_x = (bbox.x_min + bbox.x_max) / 2 * width
                    # Points left of center get extra shift, right of center less
                    if px < bbox_center_x:
                        new_x = px - shift_left - expand_width
                    else:
                        new_x = px - shift_left + expand_width
                    shifted_polygon.append((new_x, py))
                
                # Smooth the polygon to eliminate harsh corners
                smoothed_polygon = smooth_polygon_for_display(shifted_polygon, num_points=150)
                
                # Draw smooth polygon outline - thick purple line
                for j in range(len(smoothed_polygon)):
                    p1 = smoothed_polygon[j]
                    p2 = smoothed_polygon[(j + 1) % len(smoothed_polygon)]
                    # Draw thick line
                    draw.line([p1, p2], fill=(255, 0, 255), width=3)
    
    # Then draw individual block boxes
    for i, block in enumerate(blocks):
        # Convert normalized coordinates to pixels
        bbox = block.bbox
        # Shift boxes left and make wider for better text coverage
        shift_left = 12
        expand_width = 8
        x1 = int(bbox.x_min * width) - shift_left - expand_width
        y1 = int(bbox.y_min * height)
        x2 = int(bbox.x_max * width) - shift_left + expand_width
        y2 = int(bbox.y_max * height)
        
        # Choose color based on translation status
        color = colors["translated"] if block.translated_text else colors["detected"]
        
        # Draw rectangle with thicker border
        for offset in range(3):  # 3px border
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=color
            )
        
        # Draw block number
        label = f"{i + 1}"
        if show_text and block.original_text:
            # Truncate long text
            text_preview = block.original_text[:20] + "..." if len(block.original_text) > 20 else block.original_text
            label = f"{i + 1}: {text_preview}"
        
        # Draw label background
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except:
            font = ImageFont.load_default()
        
        text_bbox = draw.textbbox((x1, y1 - 20), label, font=font)
        draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2], fill=color)
        draw.text((x1, y1 - 20), label, fill=(255, 255, 255), font=font)
    
    return img_with_boxes


from manger.config import get_config, AppConfig
from manger.pipeline import MangaPipeline
from manger.services.ocr import create_ocr_service, MagiOCRService, MAGI_AVAILABLE
from manger.services.translator import create_translator
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

# CSS for uniform thumbnail grid
st.markdown("""
<style>
/* Thumbnail grid container */
.thumbnail-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 10px;
    padding: 10px 0;
}

/* Each thumbnail cell */
.thumbnail-cell {
    aspect-ratio: 0.7;  /* Manga pages are typically taller than wide */
    overflow: hidden;
    border-radius: 4px;
    background: #1e1e1e;
    display: flex;
    align-items: center;
    justify-content: center;
}

.thumbnail-cell img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
}

/* Make streamlit image containers uniform in the sidebar/thumbnail area */
[data-testid="stImage"] {
    display: flex;
    justify-content: center;
    align-items: center;
}

/* Uniform height for thumbnail columns */
.stColumn [data-testid="stVerticalBlock"] {
    min-height: 180px;
    align-items: center;
}

/* Center checkbox in thumbnail grid */
[data-testid="stHorizontalBlock"] > [data-testid="column"] {
    display: flex;
    flex-direction: column;
    align-items: center;
}

[data-testid="stHorizontalBlock"] > [data-testid="column"] > [data-testid="stVerticalBlockBorderWrapper"] {
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
}

[data-testid="stCheckbox"] {
    width: auto !important;
}

/* Thumbnail image specific styling */
.thumbnail-container {
    height: 150px;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    background: #2d2d2d;
    border-radius: 4px;
    margin-bottom: 5px;
}

.thumbnail-container img {
    max-height: 100%;
    max-width: 100%;
    object-fit: contain;
}

/* Compact the button row layout */
.stHorizontalBlock {
    gap: 0.5rem !important;
}

/* Target the specific vertical block gap */
[data-testid="stVerticalBlock"] {
    gap: 0.5rem !important;
}

/* Remove extra wrapper margins */
[data-testid="stLayoutWrapper"] {
    margin: 0 !important;
}

.stColumn {
    padding: 0 !important;
}

/* Compact element containers */
.stElementContainer {
    margin: 0 !important;
}

/* Reduce button spacing */
.stButton {
    margin: 0 !important;
}

/* Reduce spacing after subheader */
h3 {
    margin-bottom: 0.25rem !important;
}

/* Compact toggle/checkbox */
.stCheckbox {
    margin-bottom: 0 !important;
}

/* Compact info alerts */
[data-testid="stAlert"] {
    margin: 0.25rem 0 !important;
    padding: 0.5rem !important;
}

/* Style file uploader button as primary (red) */
[data-testid="stFileUploader"] button {
    background-color: rgb(255, 75, 75) !important;
    color: white !important;
    border: none !important;
}

[data-testid="stFileUploader"] button:hover {
    background-color: rgb(255, 100, 100) !important;
}

/* Align columns to bottom for manga options */
[data-testid="stHorizontalBlock"]:has([data-testid="stFileUploader"]) {
    align-items: flex-end !important;
}

[data-testid="stHorizontalBlock"]:has([data-testid="stFileUploader"]) > [data-testid="column"] {
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
}
</style>
""", unsafe_allow_html=True)


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
    if "show_download_dialog" not in st.session_state:
        st.session_state.show_download_dialog = False
    if "pdf_bytes" not in st.session_state:
        st.session_state.pdf_bytes = None
    if "original_filename" not in st.session_state:
        st.session_state.original_filename = "manga.pdf"
    if "range_select_mode" not in st.session_state:
        st.session_state.range_select_mode = False
    if "last_clicked_page" not in st.session_state:
        st.session_state.last_clicked_page = None


def dismiss_download_dialog():
    """Dismiss the download dialog if open."""
    if st.session_state.get("show_download_dialog", False):
        st.session_state.show_download_dialog = False



def get_pipeline(settings: dict | None = None) -> MangaPipeline:
    """Get or create the pipeline instance.
    
    Args:
        settings: Settings dict from sidebar (api_key, provider, etc.)
    """
    # Check if we need to recreate the pipeline due to settings change
    needs_recreate = False
    if settings:
        current_settings = st.session_state.get("_pipeline_settings", {})
        if (settings.get("api_key") != current_settings.get("api_key") or
            settings.get("provider") != current_settings.get("provider") or
            settings.get("source_lang") != current_settings.get("source_lang") or
            settings.get("target_lang") != current_settings.get("target_lang") or
            settings.get("ocr_engine") != current_settings.get("ocr_engine") or
            settings.get("magi_version") != current_settings.get("magi_version")):
            needs_recreate = True
            st.session_state._pipeline_settings = settings.copy()
    
    if st.session_state.pipeline is None or needs_recreate:
        from manger.config import TranslationConfig
        
        config = get_config()
        
        # Override translation config with sidebar settings
        if settings:
            trans_config = TranslationConfig(
                provider=settings.get("provider", "openai"),
                openai_api_key=settings.get("api_key") or None,
                source_language=settings.get("source_lang", "en"),
                target_language=settings.get("target_lang", "de"),
            )
        else:
            trans_config = config.translation
        
        # Create OCR service based on settings
        use_manga_ocr = settings.get("ocr_engine", "magi") == "manga-ocr" if settings else False
        magi_version = settings.get("magi_version", "v1") if settings else "v1"
        if MAGI_AVAILABLE:
            ocr_service = MagiOCRService(config.ocr, use_manga_ocr=use_manga_ocr, model_version=magi_version)
        else:
            ocr_service = create_ocr_service(config.ocr)
        
        st.session_state.pipeline = MangaPipeline(
            config=config,
            ocr_service=ocr_service,
            translator=create_translator(trans_config),
            renderer=Renderer(config.render),
        )
    return st.session_state.pipeline


def render_sidebar():
    """Render the sidebar with settings."""
    st.sidebar.title("⚙️ Settings")
    
    st.sidebar.subheader("OCR Settings")
    
    magi_version = st.sidebar.selectbox(
        "Magi Model",
        options=["v1", "v2"],
        index=0,
        help="v1: Better OCR accuracy (KILL not FILL). v2: Faster, more features.",
    )
    
    ocr_engine = st.sidebar.selectbox(
        "OCR Engine",
        options=["magi", "manga-ocr"],
        index=0,
        help="magi: Best for English text. manga-ocr: Best for Japanese text only.",
    )
    
    confidence = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence score to keep a text block",
    )
    
    st.sidebar.subheader("Translation Settings")
    
    # Load saved API key
    if "_saved_api_key" not in st.session_state:
        st.session_state._saved_api_key = load_api_key_from_env()
    
    # API Key input
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        value=st.session_state._saved_api_key,
        type="password",
        help="Enter your OpenAI API key for translation (auto-saved)",
    )
    
    # Save API key if changed
    if api_key != st.session_state._saved_api_key:
        st.session_state._saved_api_key = api_key
        if api_key:
            save_api_key_to_env(api_key)
            st.sidebar.success("✓ API key saved", icon="🔐")
    
    provider = st.sidebar.selectbox(
        "Translation Provider",
        options=["dummy", "openai"],
        index=1,
        help="Select the translation backend",
    )
    
    source_lang = st.sidebar.selectbox(
        "Source Language",
        options=["en", "ja", "ko", "zh"],
        index=0,
        format_func=lambda x: {"en": "English", "ja": "Japanese", "ko": "Korean", "zh": "Chinese"}[x],
    )
    
    target_lang = st.sidebar.selectbox(
        "Target Language",
        options=["en", "de", "fr", "es"],
        index=1,
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
    
    st.sidebar.subheader("PDF Output Settings")
    pdf_quality = st.sidebar.slider(
        "JPEG Quality",
        min_value=30,
        max_value=95,
        value=75,
        step=5,
        help="JPEG quality for images in PDF (lower = smaller file size)",
    )
    
    pdf_max_dimension = st.sidebar.slider(
        "Max Image Size",
        min_value=800,
        max_value=2400,
        value=1400,
        step=100,
        help="Maximum image dimension in pixels (lower = smaller file size)",
    )
    
    st.sidebar.subheader("Display Settings")
    show_polygons = st.sidebar.checkbox(
        "Show Text Polygons",
        value=True,
        help="Show precise text polygons instead of bounding boxes (orange overlay)",
    )
    
    play_sound = st.sidebar.checkbox(
        "🔔 Play Sound When Ready",
        value=True,
        help="Play a notification sound when processing is complete",
    )
    
    st.sidebar.divider()
    
    # Check if we have any processed pages to download
    has_results = False
    if "pages_data" in st.session_state:
        for page_data in st.session_state.pages_data.values():
            if page_data.get("result"):
                has_results = True
                break
    
    if has_results:
        if st.sidebar.button("📥 Download PDF", type="primary", use_container_width=True):
            st.session_state.show_download_dialog = True
            st.rerun()
        st.sidebar.divider()
    
    st.sidebar.info(
        "**Note:** Enter your OpenAI API key above to use GPT-4o-mini for translation. "
        "Use 'dummy' provider for testing without an API key."
    )
    
    return {
        "confidence": confidence,
        "provider": provider,
        "api_key": api_key,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "font_size": font_size,
        "render_dpi": render_dpi,
        "show_polygons": show_polygons,
        "ocr_engine": ocr_engine,
        "magi_version": magi_version,
        "play_sound": play_sound,
        "pdf_quality": pdf_quality,
        "pdf_max_dimension": pdf_max_dimension,
    }


def _process_all_selected_pages(settings: dict):
    """Process all selected pages (OCR → Translate → Render) and offer PDF download."""
    selected_pages = st.session_state.selected_pages
    total = len(selected_pages)
    
    if total == 0:
        st.warning("No pages selected!")
        return
    
    pipeline = get_pipeline(settings)
    progress_bar = st.progress(0, text="Processing pages...")
    
    for idx, page_num in enumerate(selected_pages):
        progress_bar.progress((idx) / total, text=f"Processing page {page_num + 1} ({idx + 1}/{total})...")
        
        # Load page if not cached
        if page_num not in st.session_state.pages_data:
            try:
                img = st.session_state.pdf_service.get_page_image(page_num)
                st.session_state.pages_data[page_num] = {
                    "image": img,
                    "blocks": [],
                    "result": None,
                }
            except Exception as e:
                st.warning(f"Failed to load page {page_num + 1}: {e}")
                continue
        
        page_data = st.session_state.pages_data[page_num]
        
        # Step 1: OCR
        blocks = pipeline.ocr.process_image(page_data["image"])
        page_data["blocks"] = blocks
        
        if not blocks:
            # No text found - use original image as result
            page_data["result"] = page_data["image"]
            continue
        
        # Step 2: Translate
        texts = [b.original_text for b in blocks]
        translations = pipeline.translator.translate_batch(texts)
        
        for block, trans in zip(blocks, translations):
            block.translated_text = trans
        
        # Step 3: Render
        translated_blocks = [b for b in blocks if b.translated_text]
        
        if translated_blocks:
            result = pipeline.renderer.inpaint(page_data["image"], translated_blocks)
            result = pipeline.renderer.typeset(result, translated_blocks)
            page_data["result"] = result
        else:
            page_data["result"] = page_data["image"]
    
    progress_bar.progress(1.0, text="Complete!")
    
    # Play notification sound if enabled
    if settings.get("play_sound", True):
        play_notification_sound()
    
    # Mark processing as complete and trigger PDF generation
    st.session_state.processing_complete = True
    st.session_state.show_download_dialog = True
    
    st.rerun()


@st.dialog("📥 Download Translated PDF")
def _show_download_dialog():
    """Show a dialog to download the translated PDF."""
    # Collect all result images from pages_data (not relying on selected_pages)
    result_images = []
    processed_pages = []
    for page_num, page_data in sorted(st.session_state.pages_data.items()):
        if page_data and page_data.get("result"):
            result_images.append(page_data["result"])
            processed_pages.append(page_num)
    
    if not result_images:
        st.error("No translated pages available!")
        if st.button("Close"):
            st.session_state.show_download_dialog = False
            st.rerun()
        return
    
    st.success(f"✅ Successfully processed {len(result_images)} pages!")
    
    # Get PDF settings from session state
    current_settings = st.session_state.get("_current_settings", {})
    pdf_quality = current_settings.get("pdf_quality", 75)
    pdf_max_dimension = current_settings.get("pdf_max_dimension", 1400)
    
    # Generate PDF with current settings
    try:
        pdf_bytes = st.session_state.pdf_service.create_pdf_from_images(
            result_images,
            quality=pdf_quality,
            max_dimension=pdf_max_dimension,
        )
        st.session_state.pdf_bytes = pdf_bytes
    except Exception as e:
        st.error(f"Failed to create PDF: {e}")
        if st.button("Close"):
            st.session_state.show_download_dialog = False
            st.rerun()
        return
    
    # Generate default filename: original_name + page_range + GER
    original_name = st.session_state.get("original_filename", "manga.pdf")
    # Remove .pdf extension
    base_name = original_name.rsplit(".", 1)[0] if "." in original_name else original_name
    
    # Get page range from processed pages
    if processed_pages:
        min_page = min(processed_pages) + 1  # 1-indexed for display
        max_page = max(processed_pages) + 1
        if min_page == max_page:
            page_range = f"{min_page}"
        else:
            page_range = f"{min_page}-{max_page}"
    else:
        page_range = "all"
    
    default_name = f"{base_name} {page_range} GER.pdf"
    file_name = st.text_input("File name:", value=default_name)
    
    if not file_name.endswith(".pdf"):
        file_name += ".pdf"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📥 Download PDF",
            data=pdf_bytes,
            file_name=file_name,
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )
    
    with col2:
        if st.button("Close", use_container_width=True):
            st.session_state.show_download_dialog = False
            st.rerun()


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
        width="stretch",
        hide_index=True,
    )



def render_page_selector():
    """Render the page selection grid."""
    if not st.session_state.pdf_loaded:
        return
    
    st.subheader("📄 Select Pages to Translate")
    
    page_count = st.session_state.pdf_page_count
    
    # Sync selected_pages from checkbox states at the start (for accurate count display)
    # Only sync if checkbox states exist (i.e., thumbnails have been rendered before)
    first_checkbox_key = "page_select_0"
    
    # Range Selection Logic
    if "range_select_mode" not in st.session_state:
        st.session_state.range_select_mode = False
    
    # Detect which checkbox changed
    changed_page = None
    new_selected = []
    
    if first_checkbox_key in st.session_state:
        for page_num in range(page_count):
            checkbox_key = f"page_select_{page_num}"
            is_checked = st.session_state.get(checkbox_key, False)
            
            # Check if this specific checkbox changed since last run (we need to track previous state ideally,
            # but here we can infer from user interaction if we had previous state.
            # Streamlit doesn't give us "what changed" easily without callbacks.
            # So we will rely on the callback approach or just check against internal tracking?)
            
            # Actually, the simplest way for range select is to use the callback on the checkbox itself.
            # But we can't pass arguments easily to on_change in a loop without partials.
            # Let's stick to the plan: Checkbox state is source of truth.
            
            if is_checked:
                new_selected.append(page_num)
        
        st.session_state.selected_pages = new_selected
    
    # Range Selection Toggle
    col_mode, col_info = st.columns([1, 3])
    with col_mode:
        range_mode = st.toggle("Range Selection Mode", value=st.session_state.range_select_mode, help="When ON, clicking a page selects all pages between it and the last clicked page.")
        if range_mode != st.session_state.range_select_mode:
            st.session_state.range_select_mode = range_mode
            # Reset last clicked when mode changes
            st.session_state.last_clicked_page = None
            st.rerun()
            
    with col_info:
        if st.session_state.range_select_mode:
            st.info("👉 Click a start page, then click an end page to select the range.")
    
    # Quick selection buttons and selected count on same row
    col_btns, col_count = st.columns([2, 1])
    with col_btns:
        btn1, btn2 = st.columns(2)
        with btn1:
            if st.button("Select All", use_container_width=True, on_click=dismiss_download_dialog):
                st.session_state.selected_pages = list(range(page_count))
                for i in range(page_count):
                    st.session_state[f"page_select_{i}"] = True
                st.rerun()
        with btn2:
            if st.button("Clear Selection", use_container_width=True, on_click=dismiss_download_dialog):
                st.session_state.selected_pages = []
                for i in range(page_count):
                    st.session_state[f"page_select_{i}"] = False
                st.rerun()
    with col_count:
        st.markdown(f"**{len(st.session_state.selected_pages)}/{page_count}** pages")
    
    # Start button for all selected pages - always visible
    settings = st.session_state.get("_current_settings", {})
    provider = settings.get("provider", "openai")
    api_key = settings.get("api_key", "")
    needs_api_key = provider == "openai" and not api_key
    no_pages_selected = len(st.session_state.selected_pages) == 0
    
    if st.button(
        f"▶️ Start All Selected Pages ({len(st.session_state.selected_pages)} pages)" if st.session_state.selected_pages else "▶️ Start (select pages first)",
        key="start_all_selected",
        type="primary",
        use_container_width=True,
        disabled=needs_api_key or no_pages_selected,
        on_click=dismiss_download_dialog,
    ):
        _process_all_selected_pages(settings)
    
    if needs_api_key:
        st.caption("⚠️ Enter OpenAI API key in sidebar or use 'dummy' provider")
    
    # Show download dialog after processing is complete
    if st.session_state.get("show_download_dialog", False):
        _show_download_dialog()
    
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
                    # Show thumbnail in uniform container
                    # Resize thumbnail to uniform size while maintaining aspect ratio
                    thumb_display = thumb.copy()
                    target_height = 150
                    aspect = thumb_display.width / thumb_display.height
                    target_width = int(target_height * aspect)
                    thumb_display = thumb_display.resize((target_width, target_height), Image.Resampling.LANCZOS)
                    
                    # Create uniform canvas
                    canvas = Image.new('RGB', (120, 150), (45, 45, 45))
                    # Center the thumbnail on canvas
                    x_offset = (120 - thumb_display.width) // 2
                    y_offset = (150 - thumb_display.height) // 2
                    # Paste (handling if thumbnail is wider than canvas)
                    if thumb_display.width > 120:
                        thumb_display = thumb_display.resize((120, int(120 / aspect)), Image.Resampling.LANCZOS)
                        y_offset = (150 - thumb_display.height) // 2
                        x_offset = 0
                    canvas.paste(thumb_display, (max(0, x_offset), max(0, y_offset)))
                    
                    st.image(canvas, width="stretch")
                    
                    # Checkbox for selection - use callback to update selected_pages
                    checkbox_key = f"page_select_{page_num}"
                    
                    # Initialize checkbox state if not exists
                    if checkbox_key not in st.session_state:
                        st.session_state[checkbox_key] = page_num in st.session_state.selected_pages
                    
                    def on_checkbox_change(p_num=page_num):
                        dismiss_download_dialog()
                        
                        # Handle Range Selection
                        if st.session_state.range_select_mode:
                            is_checked = st.session_state.get(f"page_select_{p_num}", False)
                            
                            if is_checked:
                                last_page = st.session_state.last_clicked_page
                                
                                if last_page is not None and last_page != p_num:
                                    # Select range
                                    start = min(last_page, p_num)
                                    end = max(last_page, p_num)
                                    
                                    for i in range(start, end + 1):
                                        st.session_state[f"page_select_{i}"] = True
                                    
                                    # Update selected pages list immediately
                                    new_sel = []
                                    for i in range(st.session_state.pdf_page_count):
                                        if st.session_state.get(f"page_select_{i}", False):
                                            new_sel.append(i)
                                    st.session_state.selected_pages = new_sel
                                    
                                # Update last clicked
                                st.session_state.last_clicked_page = p_num
                            else:
                                # If unchecked, just reset last clicked? Or keep it?
                                # Usually range select implies clicking to ADD to selection.
                                # If unchecking, maybe we just treat it as a new anchor?
                                st.session_state.last_clicked_page = p_num
                        else:
                            # Normal mode - just track this as potential anchor if mode is switched on later?
                            # Or just ignore.
                            st.session_state.last_clicked_page = p_num

                    st.checkbox(
                        f"{page_num + 1}",
                        key=checkbox_key,
                        on_change=on_checkbox_change,
                    )
        
        # Start button at bottom of thumbnails
        st.divider()
        if st.button(
            f"▶️ Start All Selected Pages ({len(st.session_state.selected_pages)} pages)" if st.session_state.selected_pages else "▶️ Start (select pages first)",
            key="start_all_selected_bottom",
            type="primary",
            use_container_width=True,
            disabled=needs_api_key or no_pages_selected,
            on_click=dismiss_download_dialog,
        ):
            _process_all_selected_pages(settings)


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
        if st.button("⏮️ First", disabled=current_idx == 0, use_container_width=True, on_click=dismiss_download_dialog):
            st.session_state.current_page_idx = 0
            st.rerun()
    
    with nav_col2:
        if st.button("◀️ Prev", disabled=current_idx == 0, use_container_width=True, on_click=dismiss_download_dialog):
            st.session_state.current_page_idx = current_idx - 1
            st.rerun()
    
    with nav_col3:
        st.write(f"Page {current_idx + 1} of {len(selected)} selected")
    
    with nav_col4:
        if st.button("Next ▶️", disabled=current_idx >= len(selected) - 1, use_container_width=True, on_click=dismiss_download_dialog):
            st.session_state.current_page_idx = current_idx + 1
            st.rerun()
    
    with nav_col5:
        if st.button("Last ⏭️", disabled=current_idx >= len(selected) - 1, use_container_width=True, on_click=dismiss_download_dialog):
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
    
    # Get settings from session state
    settings = st.session_state.get("_current_settings", {})
    show_polygons = settings.get("show_polygons", False)
    
    # Start button at top of preview
    if st.button(
        "▶️ Process Current Page Only",
        key=f"start_top_{current_page_num}",
        type="primary",
        use_container_width=True,
    ):
        provider = settings.get("provider", "openai")
        api_key = settings.get("api_key", "")
        needs_api_key = provider == "openai" and not api_key
        
        if needs_api_key:
            st.error("⚠️ Enter OpenAI API key in sidebar or use 'dummy' provider")
        else:
            pipeline = get_pipeline(settings)
            
            # Step 1: OCR
            with st.spinner("Step 1/3: Detecting text..."):
                blocks = pipeline.ocr.process_image(page_data["image"])
                page_data["blocks"] = blocks
            
            if not blocks:
                st.warning("No text detected.")
            else:
                st.success(f"✓ Detected {len(blocks)} text blocks")
                
                # Step 2: Translate
                with st.spinner("Step 2/3: Translating..."):
                    texts = [b.original_text for b in blocks]
                    translations = pipeline.translator.translate_batch(texts)
                    
                    for block, trans in zip(blocks, translations):
                        block.translated_text = trans
                
                st.success("✓ Translation complete")
                
                # Step 3: Render
                with st.spinner("Step 3/3: Rendering..."):
                    translated_blocks = [b for b in blocks if b.translated_text]
                    
                    if translated_blocks:
                        result = pipeline.renderer.inpaint(page_data["image"], translated_blocks)
                        result = pipeline.renderer.typeset(result, translated_blocks)
                        page_data["result"] = result
                
                st.success("✓ Rendering complete!")
                if settings.get("play_sound", True):
                    play_notification_sound()
                scroll_to_results()
                st.rerun()
    
    # Display original and processed images side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Original**")
        # If blocks detected, show image with bounding boxes
        if page_data["blocks"]:
            img_with_boxes = draw_text_boxes(page_data["image"], page_data["blocks"], show_polygons=show_polygons)
            st.image(img_with_boxes, width="stretch")
            groups = group_text_blocks(page_data["blocks"])
            num_groups = len(groups)
            caption = "🔴 Red = detected, 🟢 Green = translated, 🔵 Blue = grouped"
            if show_polygons:
                caption += ", 🟠 Orange = text polygons"
            caption += f" ({len(page_data['blocks'])} blocks, {num_groups} groups)"
            st.caption(caption)
        else:
            st.image(page_data["image"], width="stretch")
    
    with col2:
        st.write("**Translated**")
        if page_data["result"]:
            st.image(page_data["result"], width="stretch")
        else:
            st.info("Process page to see translation")
    
    # Action buttons for current page
    st.divider()
    
    # Start button - runs all steps
    if st.button(
        "▶️ Process Current Page Only",
        key=f"start_{current_page_num}",
        type="primary",
        use_container_width=True,
    ):
        provider = settings.get("provider", "openai")
        api_key = settings.get("api_key", "")
        needs_api_key = provider == "openai" and not api_key
        
        if needs_api_key:
            st.error("⚠️ Enter OpenAI API key in sidebar or use 'dummy' provider")
        else:
            pipeline = get_pipeline(settings)
            
            # Step 1: OCR
            with st.spinner("Step 1/3: Detecting text..."):
                blocks = pipeline.ocr.process_image(page_data["image"])
                page_data["blocks"] = blocks
            
            if not blocks:
                st.warning("No text detected.")
            else:
                st.success(f"✓ Detected {len(blocks)} text blocks")
                
                # Step 2: Translate
                with st.spinner("Step 2/3: Translating..."):
                    texts = [b.original_text for b in blocks]
                    translations = pipeline.translator.translate_batch(texts)
                    
                    for block, trans in zip(blocks, translations):
                        block.translated_text = trans
                
                st.success("✓ Translation complete")
                
                # Step 3: Render
                with st.spinner("Step 3/3: Rendering..."):
                    translated_blocks = [b for b in blocks if b.translated_text]
                    
                    if translated_blocks:
                        result = pipeline.renderer.inpaint(page_data["image"], translated_blocks)
                        result = pipeline.renderer.typeset(result, translated_blocks)
                        page_data["result"] = result
                
                st.success("✓ Rendering complete!")
                if settings.get("play_sound", True):
                    play_notification_sound()
                scroll_to_results()
                st.rerun()
    
    st.divider()
    
    # Individual step buttons
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    
    with btn_col1:
        if st.button("🔍 Detect Text", key=f"detect_{current_page_num}", use_container_width=True):
            with st.spinner("Detecting text..."):
                pipeline = get_pipeline(settings)
                blocks = pipeline.ocr.process_image(page_data["image"])
                page_data["blocks"] = blocks
                st.success(f"Detected {len(blocks)} text blocks!")
                st.rerun()
    
    with btn_col2:
        # Check if API key is provided when using OpenAI
        provider = settings.get("provider", "openai")
        api_key = settings.get("api_key", "")
        needs_api_key = provider == "openai" and not api_key
        
        if st.button(
            "🌐 Translate",
            key=f"translate_{current_page_num}",
            disabled=len(page_data["blocks"]) == 0 or needs_api_key,
            use_container_width=True,
            help="Enter OpenAI API key in sidebar" if needs_api_key else None,
        ):
            with st.spinner("Translating..."):
                pipeline = get_pipeline(settings)
                texts = [b.original_text for b in page_data["blocks"]]
                translations = pipeline.translator.translate_batch(texts)
                
                for block, trans in zip(page_data["blocks"], translations):
                    block.translated_text = trans
                
                st.success("Translation complete!")
                st.rerun()
        
        if needs_api_key and len(page_data["blocks"]) > 0:
            st.caption("⚠️ Enter OpenAI API key in sidebar or use 'dummy' provider")
    
    with btn_col3:
        if st.button(
            "🎨 Render",
            key=f"render_{current_page_num}",
            disabled=len(page_data["blocks"]) == 0,
            use_container_width=True,
        ):
            with st.spinner("Rendering..."):
                pipeline = get_pipeline(settings)
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
                st.session_state.show_download_dialog = True
                st.rerun()


def process_all_pages():
    """Process all selected pages."""
    selected = st.session_state.selected_pages
    settings = st.session_state.get("_current_settings", {})
    
    # Check if API key is needed
    provider = settings.get("provider", "openai")
    api_key = settings.get("api_key", "")
    if provider == "openai" and not api_key:
        st.error("⚠️ OpenAI API key required. Enter it in the sidebar or switch to 'dummy' provider.")
        return
    
    pipeline = get_pipeline(settings)
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
    st.session_state.show_download_dialog = True
    st.rerun()


def main():
    """Main application entry point."""
    init_session_state()
    
    # Play any pending notification sound
    _render_pending_sound()
    
    # Header with navigation
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("📖 ManGER - Manga Translator")
    with col2:
        st.link_button("🛠️ Tools", "/Tools", use_container_width=True)
    
    st.markdown("Detect and translate text in manga. Select pages and start translating.")
    
    # Sidebar settings
    settings = render_sidebar()
    
    # Store settings in session state for other functions to access
    st.session_state._current_settings = settings
    
    # Update PDF service DPI
    st.session_state.pdf_service.dpi = settings["render_dpi"]
    
    # Two options: Scrape or Upload
    st.subheader("📥 Get Your Manga")
    
    col_scrape, col_or, col_upload = st.columns([2, 1, 2])
    with col_scrape:
        if st.button("🌐 Scrape Web Manga", use_container_width=True, type="primary"):
            st.switch_page("pages/WebManga.py")
        st.caption("Download from manga websites")
    with col_or:
        st.markdown("<div style='text-align: center; padding-top: 0.5rem; color: #888;'>OR</div>", unsafe_allow_html=True)
    with col_upload:
        uploaded_file = st.file_uploader(
            "📄 Upload PDF",
            type=["pdf"],
            help="Upload a manga PDF file",
            label_visibility="collapsed",
        )
        st.caption("Upload a manga PDF file")
    
    # Check for scraped PDF from WebManga tool
    scraped_pdf_bytes = st.session_state.pop("scraped_pdf_bytes", None)
    scraped_pdf_filename = st.session_state.pop("scraped_pdf_filename", None)
    auto_translate_scraped = st.session_state.pop("auto_translate_scraped", False)
    
    # Load scraped PDF if available
    if scraped_pdf_bytes and not st.session_state.pdf_loaded:
        with st.spinner("Loading scraped PDF..."):
            try:
                st.session_state.original_filename = scraped_pdf_filename or "scraped_manga.pdf"
                page_count = st.session_state.pdf_service.load_from_bytes(scraped_pdf_bytes)
                st.session_state.pdf_loaded = True
                st.session_state.pdf_page_count = page_count
                
                # Select all pages by default
                st.session_state.selected_pages = list(range(page_count))
                for i in range(page_count):
                    st.session_state[f"page_select_{i}"] = True
                
                # Generate thumbnails
                with st.spinner("Generating page previews..."):
                    thumbnails = st.session_state.pdf_service.get_all_thumbnails(max_size=(120, 120))
                    st.session_state.page_thumbnails = thumbnails
                
                st.success(f"Loaded scraped PDF with {page_count} pages!")
                
                # Auto-start translation if requested
                if auto_translate_scraped:
                    st.session_state.start_auto_translate = True
                
                st.rerun()
                
            except PDFError as e:
                st.error(f"Failed to load scraped PDF: {e}")
    
    if uploaded_file and not st.session_state.pdf_loaded:
        with st.spinner("Loading PDF..."):
            try:
                # Store original filename
                st.session_state.original_filename = uploaded_file.name
                
                # Read the file bytes
                pdf_bytes = uploaded_file.read()
                page_count = st.session_state.pdf_service.load_from_bytes(pdf_bytes)
                st.session_state.pdf_loaded = True
                st.session_state.pdf_page_count = page_count
                
                # Select all pages by default
                st.session_state.selected_pages = list(range(page_count))
                # Set all checkbox states to True
                for i in range(page_count):
                    st.session_state[f"page_select_{i}"] = True
                
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
        if st.button("🔄 Load New PDF", on_click=dismiss_download_dialog):
            st.session_state.pdf_service.close()
            st.session_state.pdf_loaded = False
            st.session_state.pdf_page_count = 0
            st.session_state.selected_pages = []
            st.session_state.page_thumbnails = []
            st.session_state.pages_data = {}
            st.session_state.current_page_idx = 0
            st.session_state.processing_complete = False
            st.rerun()
        
        # Start All button at the very top (always visible)
        provider = settings.get("provider", "openai")
        api_key = settings.get("api_key", "")
        needs_api_key = provider == "openai" and not api_key
        has_pages = len(st.session_state.selected_pages) > 0
        
        # Check for auto-translate trigger from WebManga scraper
        auto_translate_now = st.session_state.pop("start_auto_translate", False)
        
        page_count = len(st.session_state.selected_pages)
        
        # If auto-translating, show info instead of clickable button
        if auto_translate_now and has_pages and not needs_api_key:
            st.info(f"🔄 Auto-translating {page_count} pages...")
            _process_all_selected_pages(settings)
        else:
            button_label = f"▶️ Start All Selected Pages ({page_count} pages)" if has_pages else "▶️ Start (Select pages first)"
            
            if st.button(
                button_label,
                key="start_all_top",
                type="primary",
                use_container_width=True,
                disabled=needs_api_key or not has_pages,
                on_click=dismiss_download_dialog,
            ):
                _process_all_selected_pages(settings)
        
        if needs_api_key:
            st.caption("⚠️ Enter OpenAI API key in sidebar or use 'dummy' provider")
        
        # Check if we have any processed pages to download (Top Button)
        has_results = False
        if "pages_data" in st.session_state:
            for page_data in st.session_state.pages_data.values():
                if page_data.get("result"):
                    has_results = True
                    break
        
        if has_results:
            if st.button("📥 Download Translated PDF", key="download_top", type="secondary", use_container_width=True):
                st.session_state.show_download_dialog = True
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
    
    # Bottom Download Button
    has_results_bottom = False
    if "pages_data" in st.session_state:
        for page_data in st.session_state.pages_data.values():
            if page_data.get("result"):
                has_results_bottom = True
                break
    
    if has_results_bottom:
        if st.button("📥 Download Translated PDF", key="download_bottom", type="primary", use_container_width=True):
            st.session_state.show_download_dialog = True
            st.rerun()

    # Footer
    st.divider()
    st.caption(
        "ManGER v0.1.0 | "
        "[GitHub](https://github.com/example/manger) | "
        "Built with Streamlit"
    )
    
    # Scroll to results at the very end (after all content is rendered)
    _render_pending_scroll()


if __name__ == "__main__":
    main()