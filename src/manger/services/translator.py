"""Translation Service for text translation.

Provides multiple backend implementations for translating manga text.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from loguru import logger

from manger.config import TranslationConfig, get_config

if TYPE_CHECKING:
    pass


class TranslationError(Exception):
    """Base exception for translation-related errors."""
    pass


class TranslationAPIError(TranslationError):
    """Raised when API call fails."""
    pass


class TranslationRateLimitError(TranslationError):
    """Raised when rate limit is hit."""
    pass


class BaseTranslator(ABC):
    """Abstract base class for translation services.
    
    Implementations should handle:
    - Rate limiting
    - Retries
    - Batch processing
    """
    
    def __init__(self, config: TranslationConfig | None = None):
        """Initialize the translator.
        
        Args:
            config: Translation configuration
        """
        self.config = config or get_config().translation
    
    @abstractmethod
    def translate(self, text: str, context: str = "") -> str:
        """Translate a single text string.
        
        Args:
            text: Text to translate
            context: Optional context for better translation
            
        Returns:
            Translated text
            
        Raises:
            TranslationError: If translation fails
        """
        pass
    
    def translate_batch(
        self, texts: list[str], contexts: list[str] | None = None
    ) -> list[str]:
        """Translate multiple texts in batch.
        
        Default implementation translates one by one.
        Subclasses can override for more efficient batching.
        
        Args:
            texts: List of texts to translate
            contexts: Optional list of contexts (same length as texts)
            
        Returns:
            List of translated texts
        """
        if contexts is None:
            contexts = [""] * len(texts)
        
        if len(texts) != len(contexts):
            raise ValueError("texts and contexts must have same length")
        
        results = []
        for text, context in zip(texts, contexts):
            try:
                result = self.translate(text, context)
                results.append(result)
            except TranslationError as e:
                logger.error(f"Failed to translate '{text[:20]}...': {e}")
                results.append(f"[Translation Error: {text}]")
        
        return results


class DummyTranslator(BaseTranslator):
    """Dummy translator for testing without API costs.
    
    Returns placeholder translations.
    """
    
    # Sample Japanese to English translations
    TRANSLATIONS = {
        "こんにちは": "Hello",
        "ありがとう": "Thank you",
        "お願いします": "Please",
        "大丈夫": "It's okay",
        "すごい！": "Amazing!",
        "何？": "What?",
        "分かった": "Got it",
        "行こう！": "Let's go!",
        "待って": "Wait",
        "助けて！": "Help!",
    }
    
    def translate(self, text: str, context: str = "") -> str:
        """Return a placeholder translation.
        
        Uses predefined translations for known text,
        otherwise creates a placeholder.
        """
        if text in self.TRANSLATIONS:
            translated = self.TRANSLATIONS[text]
        else:
            # Create a readable placeholder
            translated = f"[EN: {text}]"
        
        logger.debug(f"Dummy translation: '{text}' -> '{translated}'")
        return translated


class OpenAITranslator(BaseTranslator):
    """Translator using OpenAI's GPT models.
    
    Requires MANGER_TRANSLATE_OPENAI_API_KEY environment variable.
    """
    
    def __init__(self, config: TranslationConfig | None = None):
        super().__init__(config)
        self._client = None
    
    def _get_client(self):
        """Lazily initialize the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise TranslationError(
                    "OpenAI package not installed. Install with: pip install openai"
                )
            
            if not self.config.openai_api_key:
                raise TranslationError(
                    "OpenAI API key not configured. "
                    "Set MANGER_TRANSLATE_OPENAI_API_KEY environment variable."
                )
            
            self._client = OpenAI(api_key=self.config.openai_api_key)
        
        return self._client
    
    def translate(self, text: str, context: str = "") -> str:
        """Translate text using OpenAI GPT.
        
        Args:
            text: Japanese text to translate
            context: Optional context (e.g., previous dialogue)
            
        Returns:
            English translation
        """
        client = self._get_client()
        
        system_prompt = (
            f"You are a professional manga translator from "
            f"{self.config.source_language} to {self.config.target_language}. "
            "Translate the following text naturally, preserving tone and meaning. "
            "Only output the translation, nothing else."
        )
        
        user_prompt = text
        if context:
            user_prompt = f"Context: {context}\n\nTranslate: {text}"
        
        try:
            response = client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )
            
            translated = response.choices[0].message.content.strip()
            logger.debug(f"OpenAI translation: '{text}' -> '{translated}'")
            return translated
            
        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower():
                raise TranslationRateLimitError(f"Rate limit exceeded: {e}")
            raise TranslationAPIError(f"OpenAI API error: {e}")
    
    def translate_batch(
        self, texts: list[str], contexts: list[str] | None = None
    ) -> list[str]:
        """Translate batch using combined prompts for efficiency.
        
        Combines multiple texts into a single API call when possible.
        """
        if not texts:
            return []
        
        if contexts is None:
            contexts = [""] * len(texts)
        
        # For small batches, use combined translation
        if len(texts) <= self.config.batch_size:
            return self._translate_combined(texts, contexts)
        
        # For larger batches, process in chunks
        results = []
        for i in range(0, len(texts), self.config.batch_size):
            chunk_texts = texts[i:i + self.config.batch_size]
            chunk_contexts = contexts[i:i + self.config.batch_size]
            chunk_results = self._translate_combined(chunk_texts, chunk_contexts)
            results.extend(chunk_results)
        
        return results
    
    def _translate_combined(
        self, texts: list[str], contexts: list[str]
    ) -> list[str]:
        """Translate multiple texts in a single API call."""
        client = self._get_client()
        
        system_prompt = (
            f"You are a professional manga translator from "
            f"{self.config.source_language} to {self.config.target_language}. "
            "Translate each numbered line, preserving tone and meaning. "
            "Output only the translations, numbered to match the input."
        )
        
        # Format texts as numbered list
        numbered_texts = "\n".join(
            f"{i+1}. {text}" for i, text in enumerate(texts)
        )
        
        try:
            response = client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": numbered_texts},
                ],
                temperature=0.3,
                max_tokens=1000,
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse numbered results
            results = []
            for line in result_text.split("\n"):
                line = line.strip()
                if line and line[0].isdigit():
                    # Remove number prefix
                    parts = line.split(".", 1)
                    if len(parts) > 1:
                        results.append(parts[1].strip())
                    else:
                        results.append(line)
            
            # Pad or trim to match input length
            while len(results) < len(texts):
                results.append("[Translation missing]")
            results = results[:len(texts)]
            
            return results
            
        except Exception as e:
            logger.error(f"Batch translation failed: {e}")
            # Fall back to individual translations
            return super().translate_batch(texts, contexts)


def create_translator(config: TranslationConfig | None = None) -> BaseTranslator:
    """Factory function to create the appropriate translator.
    
    Args:
        config: Translation configuration
        
    Returns:
        Appropriate translator instance based on config
    """
    config = config or get_config().translation
    
    if config.provider == "openai":
        return OpenAITranslator(config)
    elif config.provider == "deepl":
        # DeepL implementation would go here
        raise NotImplementedError("DeepL translator not yet implemented")
    else:
        return DummyTranslator(config)