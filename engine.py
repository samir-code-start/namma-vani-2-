"""engine.py â€” Namma Vanni AI pipeline: STT, LLM analysis, TTS, feedback logging, mock mode."""

import asyncio
import csv
import json
import logging
import os
import re
import requests
from datetime import datetime
from typing import Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MOCK_MODE: bool = os.getenv("MOCK_MODE", "False").lower() == "true"
GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")

SARVAM_API_KEY = "sk_5o334a0d_2rh4kcqytkfCtDXgGDWOMyCT"
SARVAM_URL = "https://api.sarvam.ai/transcription/kannada-english-hindi/v1"

LLM_MODEL = "llama-3.3-70b-versatile"

KANADA_FIXES = {
    "à²¨à²®à³à²µ": "à²¨à²®à³à²®",
    "à²µà²£à²¿": "à²µà²¾à²£à²¿",
    "à²¤à³à²‚à²¬à²¾": "à²¤à³à²‚à²¬à²¾",
    "à²¹à²¾à²³à²¾à²—à²¿à²¦à³†": "à²¹à²¾à²³à²¾à²—à²¿à²¦à³†",
    "à²°à²¸à³à²¤à³†": "à²°à²¸à³à²¤à³†",
    "à²ªà²¾à²£à²¿": "à²ªà²¾à²£à²¿",
    "à²•à³‡à²‚à²¦à³à²°": "à²•à³‡à²‚à²¦à³à²°",
    "à²«à³‹à²¨à³": "à²«à³‹à²¨à³"
}

FEEDBACK_FILE = "feedback.csv"
FEEDBACK_HEADERS = [
    "timestamp", "language", "raw_text", "ai_issue",
    "confidence", "sentiment", "citizen_response",
    "agent_correction", "handover",
]

TTS_VOICE_MAP: dict[str, str] = {
    "kn": "kn-IN-GaganNeural",
    "hi": "hi-IN-MadhurNeural",
    "en": "en-IN-NeerjaNeural",
}
TTS_FALLBACK_VOICE = "en-IN-NeerjaNeural"
TTS_OUTPUT_PATH = "verify.mp3"

# ---------------------------------------------------------------------------
# Mock payloads
# ---------------------------------------------------------------------------
_MOCK_TRANSCRIPT = "à²¨à²®à³à²® à²Šà²°à²¿ à²°à²¸à³à²¤à³† à²¤à³à²‚à²¬à²¾ à²•à³†à²Ÿà³à²Ÿà²¿à²¦à³†, à²…à²§à²¿à²•à²¾à²°à²¿à²—à²³à²¨à³à²¨à³ à²•à²³à³à²¹à²¿à²¸à²¿"

_MOCK_ANALYSIS: dict = {
    "language": "kn",
    "normalized_issue": "The road in the caller's village is severely damaged and needs official inspection.",
    "confidence": 0.92,
    "sentiment": "urgent",
    "verification_prompt": "Your village road is badly damaged and you need officials to visit. Did I understand correctly? Say Yes or No.",
    "handover": False,
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are Namma Vanni, an expert AI analyst for Karnataka's 1092 Civic Helpline.

ðŸŽ¯ CORE TASKS:
1. DOMAIN TAXONOMY: Citizens report issues like water leakage/stagnation, street lights/faulty wiring, garbage/potholes, encroachments, public sanitation, or civic emergencies. Map fragmented speech to these categories.
2. SEMANTIC EXTRACTION: Even with poor grammar, regional slang, or mixed languages, extract the PRIMARY issue. Do NOT output "could not be determined" unless audio is silent/nonsensical. Partial understanding > failure.
3. CONFIDENCE RUBRIC:
   - High (0.85-1.0): Clear issue + location/context mentioned
   - Medium (0.60-0.84): Clear issue, missing context/location
   - Low (0.30-0.59): Ambiguous phrasing, overlapping topics, or dialect-heavy
   - Critical (<0.30): Incoherent, repetitive, or severe distress
4. DYNAMIC VERIFICATION: Generate a SPECIFIC clarification question tailored to the extracted issue. Never use generic templates.
5. SENTIMENT DETECTION: Analyze tone, urgency markers, repetition, and emotional vocabulary.

OUTPUT STRICT JSON ONLY:
{
  "language": "en|kn|hi",
  "confidence": float(0.0-1.0),
  "sentiment": "calm|confused|urgent|distressed|angry",
  "normalized_issue": "Clean 1-line summary in standard English/Kannada/Hindi",
  "verification_prompt": "Natural question clarifying the specific issue. Max 20 words.",
  "handover": true|false
}

GUARDRAILS:
- If confidence < 0.7 â†’ handover=true
- If sentiment in [distressed, angry] â†’ handover=true
- Normalized_issue must never contain placeholders like "unclear" or "pending". Use factual framing: "Caller reports [issue], needs [department/action]."
- Never hallucinate departments. Stick to reported facts."""

# ---------------------------------------------------------------------------
# Groq client (lazy, cached)
# ---------------------------------------------------------------------------
_groq_client = None


def parse_confirmation(transcript: str) -> bool | None:
    """Multilingual yes/no detector for Kannada/Hindi/English + dialects."""
    t = transcript.strip().lower()
    yes_tokens = {"yes","yeah","yep","correct","right","exactly","true","agreed","sounds right","okay","ok",
                  "haan","han","hān","ji","theek","sahi","haa","haanji","bilkul","sahī hai",
                  "sari","shari","saru","hana","hanna","sha","sharangalya","thikiddu","thike"}
    no_tokens = {"no","nahi","nahin","naahi","galat","bhool","phir se","wapos","na",
                 "not correct","wrong","incorrect","missed","repeat","again","try again","nope",
                 "illa","illai","alla","daana","thappilla","muddu","kadliya","mukhyamaga kadaliya"}
    words = re.findall(r'\b\w+\b', t)
    for w in words:
        if any(y.startswith(w) or w.startswith(y) for y in yes_tokens): return True
        if any(n.startswith(w) or w.startswith(n) for n in no_tokens): return False
    if any(y in t for y in yes_tokens): return True
    if any(n in t for n in no_tokens): return False
    return None


def _get_groq_client():
    """Return a cached Groq client; raises RuntimeError if API key is missing."""
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set. Add it to .env or enable MOCK_MODE=True.")
        try:
            from groq import Groq  # type: ignore
            _groq_client = Groq(api_key=GROQ_API_KEY)
            logger.info("Groq client initialised.")
        except ImportError as exc:
            raise RuntimeError("groq package not installed. Run: pip install groq") from exc
    return _groq_client


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
_VALID_LANGUAGES = {"kn", "hi", "en"}
_VALID_SENTIMENTS = {"calm", "confused", "urgent", "distressed", "angry"}


def _strip_fences(raw: str) -> str:
    """Remove markdown/code fences from a raw string."""
    return re.sub(r"```(?:json)?\s*|\s*```", "", raw, flags=re.IGNORECASE).strip()


def _enforce_guardrails(data: dict) -> dict:
    """Validate schema fields and enforce handover guardrail rules."""
    language = data.get("language", "en")
    if language not in _VALID_LANGUAGES:
        language = "en"

    sentiment = data.get("sentiment", "calm")
    if sentiment not in _VALID_SENTIMENTS:
        sentiment = "calm"

    try:
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5

    normalized_issue = str(data.get("normalized_issue", "Unclear report. Needs agent clarification.")).strip()
    if not normalized_issue or normalized_issue.lower() == "issue could not be determined":
        normalized_issue = "Unclear report. Needs agent clarification."

    verification_prompt = str(
        data.get(
            "verification_prompt",
            "Could you please repeat your issue? Did I understand correctly? Say Yes or No.",
        )
    ).strip()

    handover: bool = bool(data.get("handover", False))
    # Enforce handover logic:
    # 1. true if confidence < 0.7
    # 2. true if distressed/angry
    if confidence < 0.7:
        handover = True
    elif sentiment in {"distressed", "angry"} or normalized_issue == "Unclear report. Needs agent clarification.":
        handover = True

    return {
        "language": language,
        "normalized_issue": normalized_issue,
        "confidence": confidence,
        "sentiment": sentiment,
        "verification_prompt": verification_prompt,
        "handover": handover,
    }


async def _tts_coroutine(text: str, voice: str, output_path: str) -> None:
    """Async coroutine: synthesise speech with edge-tts and save to disk."""
    import edge_tts  # type: ignore
    communicator = edge_tts.Communicate(text, voice)
    await communicator.save(output_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_kannada(text: str) -> str:
    """Normalize Kannada text using KANADA_FIXES."""
    for wrong, right in KANADA_FIXES.items():
        text = text.replace(wrong, right)
    return text


def normalize_transcript(text: str, lang: str) -> str:
    """Cross-language ASR drift correction for Kannada/Hindi/English."""
    if not text: return ""
    lang = lang.lower()[:2]
    KN_FIXES = {"ನಮ್ವ":"ನಮ್ಮ", "ವಣಿ":"ವಾಣಿ", "ತುಂಬಾ":"ತುಂಬಾ", "ಹಾಳಾಗಿದೆ":"ಹಾಳಾಗಿದೆ", "ರಸ್ತೆ":"ರಸ್ತೆ", "ಪಾಣಿ":"ಪಾಣಿ", "ಕೇಂದ್ರ":"ಕೇಂದ್ರ", "ಫೋನ್":"ಫೋನ್", "ಬೇಕು":"ಬೇಕು", "ಸೇವೆ":"ಸೇವೆ"}
    HI_FIXES = {"क्यो":"क्यों", "कहा":"कहाँ", "ठिक":"ठीक", "सही":"सही", "रस्ता":"रास्ता", "बात":"बात", "दिया":"दिया", "लिए":"लिए", "में":"में", "पानि":"पानी", "सफाय":"सफाई"}
    EN_FIXES = {"teh":"the", "plz":"please", "thk":"thank", "recieve":"receive", "adress":"address", "wont":"won't", "cant":"can't", "im":"I'm", "waterline":"water line", "paniline":"pani line", "bandh kr do":"shut off"}
    fixes = {"kn": KN_FIXES, "hi": HI_FIXES, "en": EN_FIXES}
    current = fixes.get(lang, EN_FIXES)
    out = text
    for k, v in current.items(): out = out.replace(k, v)
    return out.strip().replace("  ", " ").replace("\n", " ")


def transcribe_audio(audio_path: str) -> tuple[str, str]:
    """Layer 1: Sarvam AI -> Layer 2: Groq Whisper Fallback. Zero UI impact."""
    if MOCK_MODE:
        return normalize_transcript("ನಮ್ಮ ಊರಿ ರಸ್ತೆ ತುಂಬಾ ಕೆಟ್ಟಿದೆ, ಅಧಿಕಾರಿಗಳನ್ನು ಕಳುಹಿಸಿ", "kn"), "kn"

    print(f"[STT] Checking: {audio_path}", flush=True)
    if not os.path.isfile(audio_path) or os.path.getsize(audio_path) < 50:
        print("[STT] EMPTY/MISSING FILE", flush=True)
        return "", "en"

    try:
        print("[STT] Provider 1: Sarvam AI", flush=True)
        with open(audio_path, "rb") as f:
            res = requests.post(SARVAM_URL, files={"file": ("audio.wav", f, "audio/wav")},
                                headers={"subscription-key": SARVAM_API_KEY}, timeout=15)
        res.raise_for_status()
        payload = res.json()
        data = payload.get("data") or payload
        raw_text = (data.get("text") or data.get("transcript") or "").strip()
        raw_lang = (data.get("detected_language") or data.get("language") or "kn-IN").split("-")[0][:2].lower()
        lang_map = {"kn":"kn","hi":"hi","en":"en","ml":"kn","ta":"kn","te":"kn"}
        lang = lang_map.get(raw_lang, "kn")
        clean = normalize_transcript(raw_text, lang)
        print(f"[STT SARVAM OK] {raw_lang} | Len:{len(clean)}", flush=True)
        return clean, lang
    except Exception as e:
        print(f"[STT SARVAM FAIL] {type(e).__name__}: {e}", flush=True)

    try:
        print("[STT] Provider 2: Groq Whisper API", flush=True)
        client = _get_groq_client()
        with open(audio_path, "rb") as f:
            res = client.audio.transcriptions.create(model="whisper-large-v3", file=f, temperature=0.0)
        raw_text = getattr(res, "text", "").strip().replace('\n', ' ')
        raw_lang = getattr(res, "language", "en")[:2].lower()
        lang_map = {"kn":"kn","hi":"hi","en":"en","ml":"kn","ta":"kn","te":"kn"}
        lang = lang_map.get(raw_lang, "en")
        clean = normalize_transcript(raw_text, lang)
        print(f"[STT GROQ OK] {raw_lang} | Len:{len(clean)}", flush=True)
        return clean, lang
    except Exception as e:
        print(f"[STT FINAL FAIL] {type(e).__name__}: {e}", flush=True)
        return "", "en"


def extract_json(text: str) -> dict:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try: return json.loads(match.group())
        except json.JSONDecodeError as e: raise ValueError(f"Invalid JSON: {e}")
    raise ValueError("No JSON object found in LLM response")

def analyze_transcript(text: str) -> dict:
    if os.getenv("MOCK_MODE", "").lower() == "true":
        return {"language": "kn", "normalized_issue": "ನಮ್ಮ ಊರಿ ರಸ್ತೆ ತುಂಬಾ ಕೆಟ್ಟಿದೆ", 
                "confidence": 0.95, "sentiment": "urgent", 
                "verification_prompt": "ಊರಿನ ರಸ್ತೆ ಹಾಳಾಗಿದೆ ಅಂತ ಹೇಳಿದಿರಾ? ಸರಿ ಎಂದು ಹೇಳಿ.", 
                "handover": False}
    
    SYSTEM_PROMPT = """You are Namma Vanni, AI for Karnataka's 1092 helpline.
CRITICAL RULES:
- PARSE THE EXACT TRANSCRIPT PROVIDED BELOW. NEVER invent issues or use generic phrases like "I see you need help".
- Normalize regional/dialect phrases to standard civic meaning.
- Detect sentiment: calm|confused|urgent|distressed|angry
- Set handover=true IF confidence < 0.7 OR sentiment ∈ [distressed, angry] OR issue ambiguous
- Output ONLY strict JSON matching this schema:
{"language":"kn|hi|en","normalized_issue":"1-sentence summary grounded in transcript","confidence":0.0-1.0,"sentiment":"calm|confused|urgent|distressed|angry","verification_prompt":"Clear restatement in detected language. End with: 'Did I understand correctly? Say Yes or No.'","handover":true|false}
- Never output markdown, code fences, or explanations. Raw JSON only."""

    logging.info(f"[LLM INPUT] {text[:80]}{'...' if len(text)>80 else ''}")
    
    try:
        client = _get_groq_client()
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"CITIZEN TRANSCRIPT: '{text}'"}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=250
        )
        raw = res.choices[0].message.content.strip()
        parsed = extract_json(raw)
        
        required = {"language", "normalized_issue", "confidence", "sentiment", "verification_prompt", "handover"}
        missing = required - set(parsed.keys())
        if missing: raise ValueError(f"Missing keys: {missing}")
        
        parsed["confidence"] = max(0.0, min(1.0, float(parsed["confidence"])))
        if parsed.get("language", "")[:2] not in ["kn", "hi", "en"]: parsed["language"] = "kn"
        if parsed["confidence"] < 0.7 or parsed.get("sentiment") in ["distressed", "angry"]: parsed["handover"] = True
        return parsed
    except Exception as e:
        logging.error(f"[LLM FAIL] {e}")
        return {"language": "en", "normalized_issue": "Unable to parse request. Please repeat.", 
                "confidence": 0.1, "sentiment": "confused", 
                "verification_prompt": "I didn't catch that. Could you please say it again?", 
                "handover": True}


def generate_tts(text: str, lang_code: str) -> str:
    """Synthesise verification prompt to verify.mp3 via edge-tts; returns file path."""
    if MOCK_MODE:
        logger.info("[MOCK] generate_tts() â†’ writing stub verify.mp3.")
        with open(TTS_OUTPUT_PATH, "wb") as f:
            f.write(b"")  # zero-byte stub so os.path.exists() returns True
        return TTS_OUTPUT_PATH

    voice = TTS_VOICE_MAP.get(lang_code.lower().strip(), TTS_FALLBACK_VOICE)
    logger.info("TTS: voice=%s, chars=%d", voice, len(text))

    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, _tts_coroutine(text, voice, TTS_OUTPUT_PATH))
                    future.result(timeout=30)
            else:
                loop.run_until_complete(_tts_coroutine(text, voice, TTS_OUTPUT_PATH))
        except RuntimeError:
            asyncio.run(_tts_coroutine(text, voice, TTS_OUTPUT_PATH))

        logger.info("TTS saved to %s", TTS_OUTPUT_PATH)
        return TTS_OUTPUT_PATH

    except Exception as exc:
        logger.warning("TTS failed with voice %s: %s â€” retrying with fallback.", voice, exc)
        try:
            asyncio.run(_tts_coroutine(text, TTS_FALLBACK_VOICE, TTS_OUTPUT_PATH))
            logger.info("TTS fallback succeeded.")
            return TTS_OUTPUT_PATH
        except Exception as fallback_exc:
            logger.error("TTS fallback also failed: %s", fallback_exc)
            return TTS_OUTPUT_PATH  # return path even if file wasn't created; caller handles


def process_audio(audio_path: str) -> dict:
    logging.info(f"[PROCESS] Starting for {audio_path}")
    raw_text, lang = transcribe_audio(audio_path)
    logging.info(f"[STT OUTPUT] Lang: {lang}, Text Length: {len(raw_text)}")
    
    if not raw_text.strip():
        logging.warning("[STT] Empty transcript. Returning fallback.")
        return {"language": lang, "normalized_issue": "I couldn't hear clearly. Please speak again.",
                "confidence": 0.2, "sentiment": "confused", 
                "verification_prompt": "Please try recording again.", "handover": False, 
                "raw_text": "", "verify_tts_path": "verify.mp3"}
                
    ai_data = analyze_transcript(raw_text)
    tts_path = generate_tts(ai_data.get("verification_prompt", ""), ai_data.get("language", "kn"))
    
    final = {**ai_data, "raw_text": raw_text, "verify_tts_path": tts_path}
    logging.info(f"[PIPELINE OK] Confidence: {final['confidence']}, Handover: {final['handover']}")
    return final


def log_feedback(data: dict) -> None:
    """Append a feedback row to feedback.csv; creates file with headers if it does not exist."""
    file_exists = os.path.isfile(FEEDBACK_FILE)
    try:
        with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=FEEDBACK_HEADERS, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
                logger.info("Created %s with headers.", FEEDBACK_FILE)
            row = {
                "timestamp": data.get("timestamp", datetime.utcnow().isoformat()),
                "language": data.get("language", ""),
                "raw_text": data.get("raw_text", ""),
                "ai_issue": data.get("normalized_issue", ""),
                "confidence": data.get("confidence", ""),
                "sentiment": data.get("sentiment", ""),
                "citizen_response": data.get("citizen_response", ""),
                "agent_correction": data.get("agent_correction", ""),
                "handover": data.get("handover", ""),
            }
            writer.writerow(row)
            logger.info("Feedback logged to %s.", FEEDBACK_FILE)
    except Exception as exc:
        logger.error("log_feedback() failed: %s", exc)

