"""utils.py — I/O helpers: JSON cleanup, TTS voice mapping, async edge-tts wrapper."""

import asyncio
import json
import logging
import re

import edge_tts  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Voice map (BCP-47 locale → edge-tts voice name)
# ---------------------------------------------------------------------------
_VOICE_MAP: dict[str, str] = {
    "kn": "kn-IN-GaganNeural",
    "hi": "hi-IN-MadhurNeural",
    "en": "en-IN-NeerjaNeural",
}
_FALLBACK_VOICE = "en-IN-NeerjaNeural"


def get_tts_voice(lang_code: str) -> str:
    """Map a language code (kn/hi/en) to its edge-tts voice name; falls back to en-IN-NeerjaNeural."""
    return _VOICE_MAP.get(lang_code.lower().strip(), _FALLBACK_VOICE)


# ---------------------------------------------------------------------------
# Async TTS generator
# ---------------------------------------------------------------------------

async def _tts_coroutine(text: str, voice: str, output_path: str) -> None:
    """Internal coroutine: synthesise speech and save to output_path."""
    communicator = edge_tts.Communicate(text, voice)
    await communicator.save(output_path)


def generate_tts_async(text: str, voice: str, output_path: str) -> None:
    """Synthesise TTS audio synchronously using asyncio; saves MP3 to output_path."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Running inside an existing event loop (e.g. Streamlit): schedule via thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, _tts_coroutine(text, voice, output_path))
                future.result(timeout=30)
        else:
            loop.run_until_complete(_tts_coroutine(text, voice, output_path))
    except RuntimeError:
        # Fallback: create a brand-new event loop
        asyncio.run(_tts_coroutine(text, voice, output_path))


# ---------------------------------------------------------------------------
# JSON cleanup / validation
# ---------------------------------------------------------------------------

def clean_json(raw_text: str) -> dict:
    """Strip markdown code fences, parse JSON, log warning on failure, raise ValueError if unparseable."""
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw_text, flags=re.IGNORECASE).strip()
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}.")
        return parsed
    except json.JSONDecodeError as exc:
        logger.warning("clean_json(): JSONDecodeError — %s", exc)
        logger.debug("Offending raw text: %s", raw_text)
        raise ValueError(f"LLM output could not be parsed as JSON: {exc}") from exc
