"""Hugging Face hosted inference provider."""

import httpx

from ai.providers.base import BaseAIProvider
from shared.config import get_settings


class HuggingFaceProvider(BaseAIProvider):
    """Generate text using Hugging Face Inference Providers API."""

    API_URL = "https://router.huggingface.co/v1/chat/completions"

    async def generate(self, prompt: str) -> str:
        settings = get_settings()
        if not settings.huggingface_api_token:
            raise RuntimeError("Hugging Face API token is not configured")
        if not settings.huggingface_model:
            raise RuntimeError("Hugging Face model is not configured")

        payload = {
            "model": settings.huggingface_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a defensive cybersecurity assistant. "
                        "Explain security events accurately and provide "
                        "safe defensive recommendations."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "max_tokens": 700,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self.API_URL,
                headers={
                    "Authorization": (
                        f"Bearer {settings.huggingface_api_token}"
                    ),
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "Unexpected Hugging Face response format"
            ) from exc

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Hugging Face returned an empty response")

        return content.strip()
