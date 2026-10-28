"""Provider selection, fallback, and graceful degradation."""

import logging

from ai.providers.base import BaseAIProvider
from ai.providers.huggingface import HuggingFaceProvider
from ai.providers.ollama import OllamaProvider

logger = logging.getLogger(__name__)


class AIProviderManager:
    """Manage AI provider selection and fallback."""

    def __init__(
        self,
        primary: BaseAIProvider | None = None,
        fallback: BaseAIProvider | None = None,
    ):
        self.primary = primary or HuggingFaceProvider()
        self.fallback = fallback or OllamaProvider()

    async def generate(self, prompt: str) -> tuple[str, str, bool]:
        """Return (generated_text, provider_name, degraded).

        The caller decides what safe response to use if both providers fail.
        """
        try:
            result = await self.primary.generate(prompt)
            return result, "huggingface", False
        except Exception:
            # Log the provider failure without logging secrets or full prompts.
            logger.warning("Primary AI provider failed", exc_info=True)

        try:
            result = await self.fallback.generate(prompt)
            return result, "ollama", True
        except Exception:
            logger.warning("Fallback AI provider failed", exc_info=True)

        raise RuntimeError("All configured AI providers failed")
