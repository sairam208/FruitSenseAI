"""Speech service module for FruitSenseAI.

Provides text-to-speech synthesis using the official sarvamai SDK.
Supports en-IN, ta-IN, and hi-IN audio generation.
"""

from __future__ import annotations

import base64
from pathlib import Path
from sarvamai import SarvamAI

from src.config import SARVAM_TTS_MODEL, SUPPORTED_LANGUAGES, get_sarvam_api_key

def generate_speech(
    text: str,
    language_code: str = "en-IN",
    output_path: Path | str | None = None,
) -> bytes:
    """Generate spoken audio for the given explanation text using Sarvam AI.

    Args:
        text: Text to synthesize (max 2400 chars).
        language_code: One of 'en-IN', 'ta-IN', 'hi-IN'.
        output_path: Optional file path to save the generated WAV file.

    Returns:
        Raw audio bytes (WAV format).

    Raises:
        EnvironmentError: If SARVAM_API_KEY is missing.
        ValueError: If language_code is unsupported.
        RuntimeError: If Sarvam API fails or returns no audio.
    """
    if language_code not in SUPPORTED_LANGUAGES:
        valid = ", ".join(SUPPORTED_LANGUAGES.keys())
        raise ValueError(f"Unsupported language code '{language_code}'. Valid codes: {valid}")

    api_key = get_sarvam_api_key()
    if not api_key:
        raise EnvironmentError(
            "SARVAM_API_KEY is not configured in the environment or .env file.\n"
            "Set it in PowerShell with: $env:SARVAM_API_KEY=\"your_key_here\""
        )

    clean_text = text.strip()
    if not clean_text:
        raise ValueError("Cannot synthesize speech for empty text.")

    client = SarvamAI(api_subscription_key=api_key)
    try:
        response = client.text_to_speech.convert(
            model=SARVAM_TTS_MODEL,
            text=clean_text[:2400],
            language_code=language_code,
        )
    except Exception as exc:
        raise RuntimeError(f"Sarvam AI TTS API request failed: {exc}") from exc

    audio_list = getattr(response, "audios", None)
    if not audio_list or not audio_list[0]:
        raise RuntimeError("Sarvam AI returned no audio content in the response.")

    try:
        audio_bytes = base64.b64decode(audio_list[0])
    except Exception as exc:
        raise RuntimeError(f"Failed to decode base64 audio response: {exc}") from exc

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio_bytes)

    return audio_bytes
