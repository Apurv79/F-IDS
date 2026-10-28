"""Test provider fallback without making real network calls."""

import pytest

from ai.providers.manager import AIProviderManager


class SuccessfulProvider:

    async def generate(self, prompt: str) -> str:
        return "Generated response"


class FailingProvider:

    async def generate(self, prompt: str) -> str:
        raise RuntimeError("Provider unavailable")


@pytest.mark.asyncio
async def test_primary_provider_is_used():
    manager = AIProviderManager(
        primary=SuccessfulProvider(),
        fallback=FailingProvider(),
    )
    text, provider, degraded = await manager.generate("Test")
    assert text == "Generated response"
    assert provider == "huggingface"
    assert degraded is False


@pytest.mark.asyncio
async def test_fallback_provider_is_used():
    manager = AIProviderManager(
        primary=FailingProvider(),
        fallback=SuccessfulProvider(),
    )
    text, provider, degraded = await manager.generate("Test")
    assert text == "Generated response"
    assert provider == "ollama"
    assert degraded is True


@pytest.mark.asyncio
async def test_both_providers_failing_raises():
    manager = AIProviderManager(
        primary=FailingProvider(),
        fallback=FailingProvider(),
    )
    with pytest.raises(RuntimeError):
        await manager.generate("Test")
