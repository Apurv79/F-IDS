"""Provider selection, fallback, and graceful degradation."""

import logging

from ai.providers.base import BaseAIProvider
from ai.providers.gemini import GeminiProvider
from ai.providers.huggingface import HuggingFaceProvider
from ai.providers.ollama import OllamaProvider
from shared.config import get_settings

logger = logging.getLogger(__name__)


class AIProviderManager:
    """Manage AI provider selection and fallback."""

    def __init__(
        self,
        primary: BaseAIProvider | None = None,
        fallback: BaseAIProvider | None = None,
    ):
        settings = get_settings()
        # If Gemini API Key is provided, prioritize Gemini Flash (fast & generous free tier)
        if primary:
            self.primary = primary
        elif settings.gemini_api_key:
            self.primary = GeminiProvider()
        else:
            self.primary = HuggingFaceProvider()

        self.fallback = fallback or OllamaProvider()

    async def generate(self, prompt: str) -> tuple[str, str, bool]:
        """Return (generated_text, provider_name, degraded)."""
        if isinstance(self.primary, GeminiProvider):
            primary_name = "gemini"
        elif isinstance(self.primary, HuggingFaceProvider):
            primary_name = "huggingface"
        elif isinstance(self.primary, OllamaProvider):
            primary_name = "ollama"
        else:
            primary_name = "huggingface"

        try:
            result = await self.primary.generate(prompt)
            return result, primary_name, False
        except Exception:
            logger.warning(f"Primary AI provider ({primary_name}) failed", exc_info=True)

        try:
            result = await self.fallback.generate(prompt)
            return result, "ollama", True
        except Exception:
            logger.warning("Fallback AI provider failed", exc_info=True)

        raise RuntimeError("All configured AI providers failed")
