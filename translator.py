import os
import re
import asyncio
import httpx
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")

AVAILABLE_MODELS = [
    "gemini-flash-latest",
    "gemini-pro-latest",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash-001",
    "gemma-3-27b-it",
]

DEFAULT_MODEL = "gemini-flash-latest"

def get_gemini_url(model: str) -> str:
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

PROMPT_TEMPLATE = """You are a professional multilingual translator.
Translate the given text into: Tamil, English, Kannada, Malayalam, Arabic, Telugu, Thanglish, Hindi, French, and German.

Tone: {tone}

Rules:
- Preserve meaning and intent
- Maintain natural grammar
- Match the tone (formal/casual)
- No explanations
- Thanglish = Tamil in English letters
- Arabic = proper Arabic script

INPUT:
{text}

Return strictly in this format:
Tamil:...
English:...
Kannada:...
Malayalam:...
Arabic:...
Telugu:...
Thanglish:...
Hindi:...
French:...
German:..."""


def parse_translations(raw: str) -> dict:
    keys = ["Tamil", "English", "Kannada", "Malayalam", "Arabic",
            "Telugu", "Thanglish", "Hindi", "French", "German"]
    result = {}
    for i, key in enumerate(keys):
        next_key = keys[i + 1] if i + 1 < len(keys) else None
        pattern = (
            rf"{key}:\s*(.*?)(?=\n{next_key}:)" if next_key
            else rf"{key}:\s*(.*)"
        )
        match = re.search(pattern, raw, re.DOTALL)
        result[key] = match.group(1).strip() if match else ""
    return result


async def translate_text(text: str, tone: str, model: str = DEFAULT_MODEL, retries: int = 3) -> dict:
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set in .env file")

    if model not in AVAILABLE_MODELS:
        model = DEFAULT_MODEL

    prompt = PROMPT_TEMPLATE.format(tone=tone, text=text)
    url = get_gemini_url(model)

    for attempt in range(retries):
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                url,
                params={"key": GOOGLE_API_KEY},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.3},
                },
            )

        if response.status_code == 429:
            wait = 2 ** attempt
            if attempt < retries - 1:
                await asyncio.sleep(wait)
                continue
            raise RuntimeError("Google API rate limit exceeded. Wait a moment and try again.")

        if response.status_code in (401, 403):
            raise RuntimeError(f"Invalid or unauthorized Google API key (HTTP {response.status_code}).")

        if not response.is_success:
            raise RuntimeError(f"Gemini API error {response.status_code}: {response.text[:200]}")

        data = response.json()
        content = data["candidates"][0]["content"]["parts"][0]["text"]
        return parse_translations(content)

    raise RuntimeError("Translation failed after multiple retries.")
