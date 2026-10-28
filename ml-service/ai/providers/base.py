"""Common interface for hosted AI providers."""

from abc import ABC, abstractmethod


class BaseAIProvider(ABC):
    """Interface implemented by each AI provider."""

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        raise NotImplementedError
