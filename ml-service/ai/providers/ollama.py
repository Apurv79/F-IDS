"""Ollama Cloud hosted inference provider."""

import httpx

from ai.providers.base import BaseAIProvider
from shared.config import get_settings


class OllamaProvider(BaseAIProvider):
    """Generate text using the hosted Ollama API."""

    async def generate(self, prompt: str) -> str:
        settings = get_settings()
        if not settings.ollama_api_key:
            raise RuntimeError("Ollama API key is not configured")
        if not settings.ollama_model:
            raise RuntimeError("Ollama model is not configured")

        # Use https://ollama.com in .env for cloud inference.
        # The API endpoint is appended here.
        base_url = settings.ollama_base_url.rstrip("/")
        if base_url.endswith("/api"):
            endpoint = f"{base_url}/chat"
        else:
            endpoint = f"{base_url}/api/chat"

        payload = {
            "model": settings.ollama_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a defensive cybersecurity assistant. "
                        "Explain security events and suggest defensive "
                        "remediation steps."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {settings.ollama_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        try:
            content = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                "Unexpected Ollama response format"
            ) from exc

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Ollama returned an empty response")

        return content.strip()
