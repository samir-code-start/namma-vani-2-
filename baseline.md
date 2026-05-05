# 📘 Namma Vanni | AI Helpline Assistant - BASELINE SPECIFICATION

## 🎯 CORE OBJECTIVE
Build a voice-to-voice AI assistive layer for Karnataka's 1092 helpline that **ensures accurate understanding before enabling response**. The system must interpret multilingual/dialect-rich citizen speech, verify understanding explicitly, detect sentiment/urgency, and safely hand over to human agents when confidence is low or distress is detected.

## 🚫 STRICT CONSTRAINTS
- ✅ **ALLOWED**: Python, Streamlit, Groq API (`whisper-large-v3` + `llama-3.3-70b-versatile`), `edge-tts`, `python-dotenv`, `pandas`/`csv`, `asyncio`
- ❌ **FORBIDDEN**: Google Cloud, Hugging Face, Twilio/telephony, external TTS APIs, placeholder code, TODO comments, incomplete functions, hardcoded paths outside `./`
- 💰 **COST MODEL**: Single `GROQ_API_KEY`. All other components free/unlimited.
- 🔄 **DEVELOPMENT MODE**: Implement in strict chunks. Never merge features. Validate after each chunk before proceeding.

## 🛠 TECH STACK
| Component | Tool | Purpose |
|-----------|------|---------|
| STT | `groq` (`whisper-large-v3`) | Speech-to-text with Indian accent/dialect support |
| LLM | `groq` (`llama-3.3-70b-versatile`) | Dialect normalization, sentiment detection, verification prompt generation |
| TTS | `edge-tts` | Free neural TTS (`kn-IN`, `hi-IN`, `en-IN` voices) |
| UI/Orchestrator | `streamlit` | Web-based voice interface, state machine, agent dashboard |
| Config/Logs | `python-dotenv`, `csv`/`pandas` | Secure key loading, append-only feedback logging |

## 📁 PROJECT STRUCTURE (EXACT)

namma-vanni/
├── BASELINE.md ← AI rulebook (read first)
├── .env ← GROQ_API_KEY=... & MOCK_MODE=True
├── requirements.txt ← Exact dependencies
├── engine.py ← STT, LLM, TTS, validation, logging
├── app.py ← Streamlit UI, state machine, handover, dashboard
├── utils.py ← Audio helpers, JSON cleanup, retry logic, logging
├── feedback.csv ← Auto-generated, append-only learning log
└── .streamlit/
└── secrets.toml ← Deploy-ready secrets (Streamlit Cloud)


## 🧱 CODING STANDARDS (AI MUST FOLLOW)
1. **Type Hints & Docstrings**: Every function must have explicit types and a 1-line docstring.
2. **No Placeholders**: Output complete, runnable code. Skip nothing. No `pass` or `# TODO`.
3. **Error Handling**: Wrap all API/TTS calls in `try/except`. Log warnings, never crash. Provide fallbacks.
4. **Strict JSON Enforcement**: LLM outputs must be cleaned via regex (`r'```json\s*|\s*```'`) + fallback `json.JSONDecodeError` handler.
5. **Mock-First Design**: `MOCK_MODE=True` bypasses network calls. Mock responses MUST match production schema exactly.
6. **Modular Separation**: 
   - `engine.py` → AI pipeline only
   - `utils.py` → I/O, validation, logging, async wrappers
   - `app.py` → UI & state flow only
7. **Performance**: Cache Groq client with `@st.cache_resource`. Use async TTS. Non-blocking audio playback. Target <4s/turn.

## 🔗 DATA FLOW & STATE MACHINE

[CITIZEN SPEAKS] → st.audio_input → input.wav
↓
[STT] → Groq Whisper → raw_text
↓
[ANALYZE] → Groq LLM → {language, normalized_issue, confidence, sentiment, verification_prompt, handover}
↓
[VERIFY] → edge-tts → verify.mp3 (autoplay) → Citizen says Yes/No
↓
[DECIDE] →
✅ Confirmed → Agent Dashboard + CSV Log
❌ Failed ×2 OR handover=True → Human Handover UI + Audio Cue + CSV Log
🔁 Partial → Retry (max 2)


## 📊 STRICT JSON SCHEMA (LLM OUTPUT)
```json
{
  "language": "kn|hi|en",
  "normalized_issue": "1-sentence clear summary",
  "confidence": 0.0-1.0,
  "sentiment": "calm|confused|urgent|distressed|angry",
  "verification_prompt": "Clear restatement in detected language. Ends with: 'Did I understand correctly? Say Yes or No.'",
  "handover": true|false
}

GUARDRAIL RULES:
handover = true IF confidence < 0.7 OR sentiment ∈ [distressed, angry] OR issue is ambiguous
verification_prompt MUST be ≤18 words
System prompt MUST enforce dialect normalization, cultural context awareness, and explicit confirmation request
🛡️ UI STATE RULES (Streamlit)
Use st.session_state exclusively for flow control
Valid States: input → verify → decision → agent_ready | handover
Never mutate state outside explicit transitions
Preserve audio files between reruns (input.wav, verify.mp3)
Use @st.cache_resource for Groq client initialization
Auto-play TTS: st.audio(path, autoplay=True, format="audio/mpeg")
Use visual badges for sentiment: 🟢 Calm, 🟡 Confused, 🟠 Urgent, 🔴 Distressed, 🟥 Angry
📝 FEEDBACK PIPELINE
File: feedback.csv
Headers: timestamp,language,raw_text,ai_issue,confidence,sentiment,citizen_response,agent_correction,handover
Append-only, UTF-8 encoded, thread-safe
Missing file → auto-create with headers on first log
Real-time viewer in Agent Dashboard via st.dataframe(pd.read_csv("feedback.csv"))
Data structured for future prompt optimization or supervised fine-tuning
🧪 CHUNKED EXECUTION PROTOCOL
AI will implement in strict sequence. DO NOT skip, merge, or preview future chunks.
CHUNK 1 → engine.py foundation (STT + LLM + JSON schema + mock mode)
CHUNK 2 → engine.py + utils.py (TTS pipeline + async wrappers + JSON validation)
CHUNK 3 → app.py (Record → Verify → Confirm loop + state transitions)
CHUNK 4 → app.py (Handover UI + Agent Dashboard + emotional routing)
CHUNK 5 → engine.py + app.py + README.md (CSV logging + live viewer + deploy guide)
VALIDATION GATE: After each chunk, output exact terminal commands to verify functionality. User runs them. Proceed only on success.

✅ COMPLIANCE MAPPING (1092 Problem Statement)
Requirement
Implementation
Voice-to-Voice
st.audio_input → Groq STT → edge-tts → st.audio(autoplay)
Verified Understanding
Explicit LLM-generated prompt + verbal Yes/No loop
Dialect & Cultural
System prompt enforces normalization + language auto-detect
Sentiment/Emotion
LLM detects → surfaces in UI → triggers guardrails
Guardrails/Handover
Confidence <0.7, distress, or 2x fail → red banner + audio cue
Human-in-Loop
Agent dashboard shows summary, allows edits, logs corrections
Learning Loop
Append-only feedback.csv + live viewer + structured for fine-tuning
Free/No GCP
Groq-only API + edge-tts + Streamlit Cloud