"""engine.py — Namma Vanni AI pipeline: STT, LLM analysis, TTS, feedback logging, mock mode."""

import asyncio
import concurrent.futures
import csv
import json
import logging
import os
import re
import requests
from datetime import datetime

import edge_tts
from groq import Groq
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def translate_to_english(text: str) -> str:
    """Translates any text to English via Groq. Safe fallback to original."""
    if not text.strip(): return ""
    try:
        client = _get_groq_client()
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": f"Translate to English ONLY: '{text}'"}],
            temperature=0.1, max_tokens=100
        )
        return res.choices[0].message.content.strip()
    except Exception: return text # Fail-safe

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MOCK_MODE: bool = os.getenv("MOCK_MODE", "False").lower() == "true"
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
SARVAM_API_KEY: str | None = os.getenv("SARVAM_API_KEY")
SARVAM_URL = "https://api.sarvam.ai/transcription/kannada-english-hindi/v1"
LLM_MODEL = "llama-3.3-70b-versatile"

KANADA_FIXES = {
    "ನಮ್ವ": "ನಮ್ಮ",
    "ವಣಿ": "ವಾಣಿ",
    "ತುಂಬಾ": "ತುಂಬಾ",
    "ಹಾಳಾಗಿದೆ": "ಹಾಳಾಗಿದೆ",
    "ರಸ್ತೆ": "ರಸ್ತೆ",
    "ಪಾಣಿ": "ಪಾಣಿ",
    "ಕೇಂದ್ರ": "ಕೇಂದ್ರ",
    "ಫೋನ್": "ಫೋನ್"
}

FEEDBACK_FILE = "feedback.csv"
FEEDBACK_HEADERS = [
    "timestamp", "language", "raw_text", "ai_issue",
    "confidence", "sentiment", "citizen_response",
    "agent_correction", "handover",
]

TTS_VOICE_MAP: dict[str, str] = {
    "kn": "kn-IN-VarunNeural",
    "hi": "hi-IN-MadhurNeural",
    "en": "en-IN-NeerjaNeural",
}
TTS_FALLBACK_VOICE = "en-IN-NeerjaNeural"
TTS_OUTPUT_PATH = "verify.mp3"

# ---------------------------------------------------------------------------
# Mock payloads
# ---------------------------------------------------------------------------
_MOCK_TRANSCRIPT = "ನಮ್ಮ ಊರಿ ರಸ್ತೆ ತುಂಬಾ ಕೆಟ್ಟಿದೆ, ಅಧಿಕಾರಿಗಳನ್ನು ಕಳುಹಿಸಿ"

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

🎯 CORE TASKS:
1. DOMAIN TAXONOMY: Map fragmented speech to issues like water leakage, street lights, garbage, etc.
2. SEMANTIC EXTRACTION: Extract PRIMARY issue. Partial understanding > failure.
3. CONFIDENCE RUBRIC:
   - High (0.85-1.0): Clear issue + location
   - Medium (0.60-0.84): Clear issue, missing location
   - Low (0.30-0.59): Ambiguous or dialect-heavy
   - Critical (<0.30): Incoherent
4. DYNAMIC VERIFICATION: Generate a SPECIFIC clarification question tailored to the issue.
5. SENTIMENT DETECTION: Analyze tone, urgency.

OUTPUT STRICT JSON ONLY:
{
  "language": "en|kn|hi",
  "confidence": 0.0-1.0,
  "sentiment": "calm|confused|urgent|distressed|angry",
  "normalized_issue": "Clean 1-line summary",
  "verification_prompt": "Natural question clarifying the specific issue. End with: 'Did I understand correctly? Say Yes or No.' Max 20 words.",
  "handover": true|false
}

GUARDRAILS:
- If confidence < 0.7 -> handover=true
- If sentiment in [distressed, angry] -> handover=true"""

# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_groq_client() -> Groq:
    """Return a cached Groq client; raises RuntimeError if API key is missing."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set. Add it to .env or enable MOCK_MODE=True.")
    client = Groq(api_key=key)
    logger.info("Groq client initialised.")
    return client

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
    communicator = edge_tts.Communicate(text, voice)
    await communicator.save(output_path)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_confirmation(transcript: str) -> dict:
    """Robustly parses multilingual yes/no & summarizes longer inputs."""
    t = transcript.strip().lower()
    if not t: return {"intent": "unclear", "summary": ""}
    
    # Multi-language affirmation/negation patterns
    yes_tokens = ["yes","yeah","yep","correct","right","exactly","agreed","okay","ok",
                  "haan","han","hān","ji","theek","sahi","haa","bilkul","sari","sha","hana"]
    no_tokens = ["no","nahi","nahin","naahi","galat","bhool","phir se","wapos","na",
                 "wrong","incorrect","missed","repeat","again","try again",
                 "illa","illai","alla","muddu","kadliya","galti","bharosa nahi"]
                
    yes_hits = [w for w in yes_tokens if w in t]
    no_hits = [w for w in no_tokens if w in t]
    
    if len(yes_hits) > 0 and len(no_hits) == 0:
        return {"intent": "confirmed", "summary": t[:80]}
    if len(no_hits) > 0 and len(yes_hits) == 0:
        return {"intent": "denied", "summary": t[:80]}
    if len(t) > 20 or (len(yes_hits) > 0 and len(no_hits) > 0):
        # Longer/mixed input → route to quick LLM intent extraction
        try:
            client = _get_groq_client()
            system = "You are a helpline assistant. Respond ONLY with JSON: {\"intent\":\"confirmed\"|\"denied\"|\"unclear\",\"summary\":\"one-line clarification of user's exact meaning\"}. Analyze this spoken reply:"
            res = client.chat.completions.create(model="llama-3.3-70b-versatile",
                messages=[{"role":"system","content":system},{"role":"user","content":t}],
                temperature=0.1, max_tokens=80)
            import re, json
            m = re.search(r'\{.*\}', res.choices[0].message.content, re.DOTALL)
            if m: return json.loads(m.group())
        except: pass
        return {"intent": "unclear", "summary": t[:80]}
    return {"intent": "unclear", "summary": t[:80]}

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
    """Layer 1: Sarvam → Layer 2: Groq. Forces Indic language routing."""
    if os.getenv("MOCK_MODE", "").lower() == "true":
        return normalize_transcript("ನಮ್ಮ ಊರಿ ರಸ್ತೆ ತುಂಬಾ ಕೆಟ್ಟಿದೆ, ಅಧಿಕಾರಿಗಳನ್ನು ಕಳುಹಿಸಿ", "kn"), "kn"
    
    print(f"[STT] Checking: {audio_path}", flush=True)
    if not os.path.isfile(audio_path) or os.path.getsize(audio_path) < 50:
        return "", "en"
        
    # ── LAYER 1: SARVAM ──
    try:
        import requests
        with open(audio_path, "rb") as f:
            res = requests.post("https://api.sarvam.ai/transcription/kannada-english-hindi/v1",
                                files={"file": ("audio.wav", f, "audio/wav")},
                                headers={"subscription-key": os.getenv("SARVAM_API_KEY")}, timeout=15)
                                
        if res.status_code != 200:
            raise ValueError(f"Sarvam error: {res.text}")
            
        res.raise_for_status()
        data = res.json().get("data") or res.json()
        raw_text = (data.get("text") or data.get("transcript") or "").strip()
        
        if len(raw_text) < 3:
            raise ValueError("Transcript too short or empty")
            
        print(f"[DEBUG] SARVAM SUCCESS CHECK: Status={res.status_code}, Len={len(raw_text)}", flush=True)
        
        raw_lang = (data.get("detected_language") or data.get("language") or "kn-IN").split("-")[0][:2]
        
        # Force correct language if Unicode Indic detected
        if any(ord(c) > 127 for c in raw_text):
            raw_lang = "kn" if any(chr(2400) <= ord(c) <= chr(2815) for c in raw_text) else "hi"
            
        clean = normalize_transcript(raw_text, raw_lang)
        lang_map = {"kn":"kn","hi":"hi","en":"en","ml":"kn","ta":"kn","te":"kn"}
        print(f"[STT SARVAM OK] {raw_lang} → {lang_map.get(raw_lang,'kn')} | Len:{len(clean)}", flush=True)
        return clean, lang_map.get(raw_lang, "kn")
    except Exception as e:
        print(f"[WARN] SARVAM SKIPPED: {e} -> Switching to Groq", flush=True)
        
    # ── LAYER 2: GROQ ──
    try:
        g_client = _get_groq_client()
        with open(audio_path, "rb") as f:
            res = g_client.audio.transcriptions.create(model="whisper-large-v3", file=f, temperature=0.0)
        raw_text = getattr(res, "text", "").strip().replace('\n', ' ')
        raw_lang = getattr(res, "language", "en")[:2].lower()
        if any(ord(c) > 127 for c in raw_text): raw_lang = "kn"
        clean = normalize_transcript(raw_text, raw_lang)
        lang_map = {"kn":"kn","hi":"hi","en":"en","ml":"kn","ta":"kn","te":"kn"}
        print(f"[STT GROQ OK] {raw_lang} → {lang_map.get(raw_lang,'kn')} | Len:{len(clean)}", flush=True)
        return clean, lang_map.get(raw_lang, "en")
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
    if MOCK_MODE:
        return _MOCK_ANALYSIS
    
    logging.info(f"[LLM INPUT] {text[:80]}{'...' if len(text)>80 else ''}")
    
    try:
        client = _get_groq_client()
        res = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"CITIZEN TRANSCRIPT: '{text}'"}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=250
        )
        raw = res.choices[0].message.content.strip()
        clean_raw = _strip_fences(raw)
        parsed = extract_json(clean_raw)
        return _enforce_guardrails(parsed)
    except Exception as e:
        logging.error(f"[LLM FAIL] {e}")
        return _enforce_guardrails({
            "language": "en",
            "normalized_issue": "Unable to parse request. Please repeat.",
            "confidence": 0.1,
            "sentiment": "confused",
            "verification_prompt": "I didn't catch that. Could you please say it again?",
            "handover": True
        })

def generate_tts(text: str, lang: str) -> str:
    """Synthesise verification prompt to verify.mp3 via edge-tts; returns file path."""
    if MOCK_MODE:
        logger.info("[MOCK] generate_tts() -> writing stub verify.mp3.")
        with open(TTS_OUTPUT_PATH, "wb") as f:
            f.write(b"")  # zero-byte stub
        return TTS_OUTPUT_PATH

    voice = TTS_VOICE_MAP.get(lang.lower().strip(), TTS_FALLBACK_VOICE)
    logger.info("TTS: voice=%s, chars=%d", voice, len(text))

    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
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
        logger.warning("TTS failed with voice %s: %s — retrying with fallback.", voice, exc)
        try:
            asyncio.run(_tts_coroutine(text, TTS_FALLBACK_VOICE, TTS_OUTPUT_PATH))
            logger.info("TTS fallback succeeded.")
            return TTS_OUTPUT_PATH
        except Exception as fallback_exc:
            logger.error("TTS fallback also failed: %s", fallback_exc)
            return TTS_OUTPUT_PATH

def process_audio(audio_path: str) -> dict:
    logging.info(f"[PROCESS] Starting for {audio_path}")
    raw_text, lang = transcribe_audio(audio_path)
    logging.info(f"[STT OUTPUT] Lang: {lang}, Text Length: {len(raw_text)}")
    
    if not raw_text.strip():
        logging.warning("[STT] Empty transcript. Returning fallback.")
        fallback = _enforce_guardrails({
            "language": lang, 
            "normalized_issue": "I couldn't hear clearly. Please speak again.",
            "confidence": 0.2, 
            "sentiment": "confused", 
            "verification_prompt": "Please try recording again.", 
            "handover": False
        })
        return {**fallback, "raw_text": "", "verify_tts_path": "verify.mp3"}
                
    english_raw = translate_to_english(raw_text)
    ai_data = analyze_transcript(english_raw)
    tts_path = generate_tts(ai_data.get("verification_prompt", ""), ai_data.get("language", "kn"))
    
    final = {**ai_data, "raw_text": english_raw, "verify_tts_path": tts_path}
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
