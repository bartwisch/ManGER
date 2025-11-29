"""Configuration management using pydantic-settings.

All magic numbers and thresholds are centralized here.
Configuration can be loaded from environment variables or .env files.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OCRConfig(BaseSettings):
    """Configuration for OCR service."""
    
    model_config = SettingsConfigDict(env_prefix="MANGER_OCR_")
    
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score to keep a text block",
    )
    merge_distance_threshold: float = Field(
        default=0.02,
        ge=0.0,
        le=0.5,
        description="Maximum normalized distance to merge adjacent blocks",
    )
    min_text_length: int = Field(
        default=1,
        ge=0,
        description="Minimum text length to keep a block",
    )
    max_image_dimension: int = Field(
        default=1600,
        ge=256,
        description="Maximum image dimension for OCR processing",
    )


class TranslationConfig(BaseSettings):
    """Configuration for translation service."""
    
    model_config = SettingsConfigDict(env_prefix="MANGER_TRANSLATE_")
    
    provider: Literal["openai", "deepl", "dummy"] = Field(
        default="openai",
        description="Translation provider to use",
    )
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model to use for translation",
    )
    deepl_api_key: str | None = Field(
        default=None,
        description="DeepL API key",
    )
    source_language: str = Field(
        default="en",
        description="Source language code",
    )
    target_language: str = Field(
        default="de",
        description="Target language code",
    )
    batch_size: int = Field(
        default=10,
        ge=1,
        description="Number of text blocks to translate in one batch",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retry attempts for failed translations",
    )
    timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        description="Timeout for translation API calls",
    )


class RenderConfig(BaseSettings):
    """Configuration for rendering/typesetting."""
    
    model_config = SettingsConfigDict(env_prefix="MANGER_RENDER_")
    
    font_path: str | None = Field(
        default=None,
        description="Path to custom font file",
    )
    default_font_size: int = Field(
        default=14,
        ge=6,
        description="Default font size in pixels",
    )
    min_font_size: int = Field(
        default=8,
        ge=4,
        description="Minimum font size in pixels",
    )
    max_font_size: int = Field(
        default=48,
        ge=12,
        description="Maximum font size in pixels",
    )
    text_color: tuple[int, int, int] = Field(
        default=(0, 0, 0),
        description="Default text color as RGB tuple",
    )
    background_color: tuple[int, int, int] = Field(
        default=(255, 255, 255),
        description="Default bubble background color as RGB tuple",
    )
    padding_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Padding inside text bubbles as ratio of box size",
    )
    line_spacing: float = Field(
        default=1.2,
        ge=1.0,
        le=3.0,
        description="Line spacing multiplier",
    )
    inpaint_method: Literal["simple", "opencv"] = Field(
        default="simple",
        description="Method for removing original text",
    )


class AppConfig(BaseSettings):
    """Main application configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="MANGER_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )
    
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )
    output_dir: Path = Field(
        default=Path("output"),
        description="Directory for output files",
    )
    temp_dir: Path = Field(
        default=Path("temp"),
        description="Directory for temporary files",
    )
    
    # Nested configurations
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    
    def ensure_directories(self) -> None:
        """Create output and temp directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)


# Global configuration instance
_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Get the global configuration instance.
    
    Returns:
        AppConfig instance (creates one if not exists)
    """
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None