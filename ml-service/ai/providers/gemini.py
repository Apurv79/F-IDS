"""Google Gemini Flash hosted inference provider (Free Tier)."""

import httpx
from ai.providers.base import BaseAIProvider
from shared.config import get_settings


class GeminiProvider(BaseAIProvider):
    """Generate text using Google Gemini 2.0 / 1.5 Flash API."""

    API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    async def generate(self, prompt: str) -> str:
        settings = get_settings()
        api_key = getattr(settings, "gemini_api_key", "")
        model = getattr(settings, "gemini_model", "gemini-2.0-flash")

        if not api_key:
            raise RuntimeError("Gemini API key is not configured")

        url = f"{self.API_BASE}/{model}:generateContent?key={api_key}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                "You are a defensive cybersecurity assistant. "
                                "Explain security events accurately and provide "
                                "practical defensive remediation recommendations.\n\n"
                                f"{prompt}"
                            )
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 700,
            },
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        try:
            candidates = data.get("candidates", [])
            if not candidates:
                raise RuntimeError("No candidates returned from Gemini API")
            content = candidates[0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Unexpected Gemini response structure") from exc

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Gemini returned an empty response")

        return content.strip()
