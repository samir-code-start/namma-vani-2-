import os
import pandas as pd
import streamlit as st

from engine import (
    MOCK_MODE,
    FEEDBACK_FILE,
    log_feedback,
    parse_confirmation,
    process_audio,
    transcribe_audio,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Namma Vanni — 1092 AI Helpline", page_icon="📞", layout="wide")

# ---------------------------------------------------------------------------
# Global Theme Injection
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    :root {
        --primary-blue: #0d6efd;
        --success-green: #198754;
        --warning-orange: #fd7e14;
        --danger-red: #dc3545;
    }

    /* Clean background & Typography */
    [data-testid="stAppViewContainer"] {
        background-color: #f0f2f6;
        font-family: 'Inter', sans-serif;
    }

    /* Spacing: Increased padding on the main container */
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
        padding-left: 5rem !important;
        padding-right: 5rem !important;
    }

    /* Card-like elements */
    .info-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        color: #374151;
    }
    
    .info-card h4 {
        margin-top: 0;
        margin-bottom: 1rem;
        color: #1f2937;
        font-size: 1.1rem;
        font-weight: 600;
    }

    /* Status Pills */
    .pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    .pill-calm { background-color: #e8f5e9; color: var(--success-green); border: 1px solid #c8e6c9; }
    .pill-confused { background-color: #fff3e0; color: var(--warning-orange); border: 1px solid #ffe0b2; }
    .pill-urgent { background-color: #ffebee; color: var(--danger-red); border: 1px solid #ffcdd2; }
    .pill-distressed { background-color: #dc3545; color: #ffffff; border: 1px solid #b02a37; }
    .pill-angry { background-color: #8b0000; color: #ffffff; border: 1px solid #600000; }
    
    .pill-confidence { background-color: #e3f2fd; color: var(--primary-blue); border: 1px solid #bbdefb; }
    .pill-language { background-color: #f3e5f5; color: #8e24aa; border: 1px solid #e1bee7; }

    /* Buttons */
    .stButton > button {
        width: 100% !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Headers Overlay */
    h1.branded-header {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #1f2937;
        border-bottom: 3px solid var(--primary-blue);
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    h2.branded-subheader {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #374151;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 0.3rem;
        margin-bottom: 1.5rem;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

SENTIMENT_BADGES = {
    "calm": "🟢 Calm",
    "confused": "🟡 Confused",
    "urgent": "🟠 Urgent",
    "distressed": "🔴 Distressed",
    "angry": "🟥 Angry",
}

def get_sentiment_pill(sentiment):
    classes = {
        "calm": "pill-calm",
        "confused": "pill-confused",
        "urgent": "pill-urgent",
        "distressed": "pill-distressed",
        "angry": "pill-angry"
    }
    css_class = classes.get(sentiment, "pill-calm")
    label = SENTIMENT_BADGES.get(sentiment, sentiment)
    return f"<span class='pill {css_class}'>{label}</span>"

def get_confidence_pill(conf):
    return f"<span class='pill pill-confidence'>{conf * 100:.0f}%</span>"

def get_language_pill(lang):
    return f"<span class='pill pill-language'>{lang.upper()}</span>"

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
_DEFAULTS = {"stage": "input_record", "ai_data": None, "attempts": 0}
for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

def _reset():
    """Reset session state to initial values."""
    for key, val in _DEFAULTS.items():
        st.session_state[key] = val

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 class='branded-header'>📞 Namma Vanni — 1092 AI Helpline</h1>"
    "<p style='text-align:center;opacity:0.6;font-family:Inter,sans-serif;margin-top:-1rem;margin-bottom:2rem;'>Voice-to-voice citizen assistant for Karnataka</p>",
    unsafe_allow_html=True,
)
if MOCK_MODE:
    st.caption("⚙️ Running in **MOCK MODE** — no API calls.")

# ═══════════════════════════════════════════════════════════════════════════
# STAGE: input_record
# ═══════════════════════════════════════════════════════════════════════════
if st.session_state.stage == "input_record":
    st.markdown("<h2 class='branded-subheader'>🎤 Start Call</h2>", unsafe_allow_html=True)
    audio_bytes = st.audio_input("Speak your concern or upload audio")

    if audio_bytes is not None:
        wav_path = "input.wav"
        with open(wav_path, "wb") as f:
            f.write(audio_bytes.getvalue())

        with st.spinner("🔄 Processing your voice — transcribing & analysing…"):
            result = process_audio(wav_path)

        st.session_state.ai_data = result
        st.session_state.attempts = 0
        st.session_state.stage = "verify"
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# STAGE: verify
# ═══════════════════════════════════════════════════════════════════════════
<<<<<<< HEAD
import time
if st.session_state.stage == "verify":
    data = st.session_state.ai_data or {}
    lang = data.get("language", "en")
    tts_path = data.get("verify_tts_path", "")
    
    # Initialize phase tracking once
    if "_ver_start" not in st.session_state:
        st.session_state._ver_start = time.time()
        st.session_state._ver_phase = 0
        st.session_state._mic_initialized = False
        
    elapsed_1 = time.time() - st.session_state._ver_start
    
    # Play AI verification summary
    st.markdown(f"**AI Summary:** {data.get('normalized_issue', '—')}")
    st.caption(f"🌐 `{lang.upper()}` | 📊 `{data.get('confidence', 0):.0%}`")
    if os.path.isfile(tts_path): st.audio(tts_path, autoplay=True)
    
    # PHASE 0: Single-click mic initialization (browser security requirement)
    if not st.session_state._mic_initialized:
        st.warning("⚠️ Click ONCE to activate live microphone. Keep device close.")
        if st.button("🎤 Activate Live Microphone", key="btn_init_mic", type="primary"):
            st.session_state._mic_initialized = True
            st.rerun()
=======
elif st.session_state.stage == "verify":
    data = st.session_state.ai_data
    if data is None:
        st.error("No analysis data found. Returning to start.")
        _reset()
        st.rerun()

    st.markdown("<h2 class='branded-subheader'>🔍 Verification</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"""
            <div class='info-card'>
                <h4>Analysis</h4>
                <div style='margin-bottom: 0.8rem;'>{get_language_pill(data.get("language", "—"))}</div>
                <div style='margin-bottom: 0.8rem;'>{get_confidence_pill(data.get('confidence', 0))}</div>
                <div>{get_sentiment_pill(data.get("sentiment", "calm"))}</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class='info-card'>
                <h4>AI Summary</h4>
                <p style='font-size: 1.1rem; margin-bottom: 0;'>{data.get('normalized_issue', '—')}</p>
            </div>
        """, unsafe_allow_html=True)

    base_prompt = data.get("verification_prompt", "")
    normalized_issue = data.get("normalized_issue", "")
    appended_prompt = f"{base_prompt} I heard you say '{normalized_issue}'. Is this correct? Say Yes or No."
    
    st.markdown(f"""
        <div class='info-card' style='background-color: #f8f9fa; border-left: 4px solid var(--primary-blue);'>
            <strong>🗣️ AI Voice Prompt:</strong> <em>"{appended_prompt}"</em>
        </div>
    """, unsafe_allow_html=True)

    # Auto-play TTS
    tts_path = data.get("verify_tts_path", "verify.mp3")
    if os.path.isfile(tts_path) and os.path.getsize(tts_path) > 0:
        st.audio(tts_path, autoplay=True, format="audio/mpeg")
>>>>>>> f454089 (Refactor UI/UX: Modern SaaS dashboard style)
    else:
        # PHASE 1: Initial 10-second auto-listening window
        if st.session_state._ver_phase == 0:
            st.info("🔴 LISTENING... Speak Yes/No/Han/Sari/Illa now.")
            conf_audio = st.audio_input("🎙️ Respond", key="mic_live", disabled=False)
            
            if conf_audio:
                with open("confirm.wav", "wb") as f: f.write(conf_audio.getvalue())
                with st.spinner("Processing..."):
                    raw_conf, _ = transcribe_audio("confirm.wav")
                    parsed = parse_confirmation(raw_conf)
                
                st.session_state.attempts += 1
                if parsed["intent"] == "confirmed":
                    st.session_state.stage = "agent_ready"
                    st.success("✅ Verified! Routing to agent...")
                    st.rerun()
                elif parsed["intent"] == "denied":
                    if st.session_state.attempts >= 2 or data.get("handover", False):
                        st.session_state.stage = "handover"
                        st.error("🔄 Handover triggered. Connecting to agent...")
                    else:
                        st.warning("⚠️ Not understood. Try again.")
                        st.session_state.stage = "input_record"
                    st.rerun()
                else:
                    st.info(f"📝 AI heard: '{parsed['summary']}'")
                    st.warning("❓ Did you mean Yes or No?")
                    st.session_state.stage = "input_record"
                    st.rerun()
                    
            if elapsed_1 >= 10:
                st.session_state._ver_phase = 1
                st.rerun()
                
        # PHASE 2: Auto-Replay TTS + Final 3-Second Safety Window
        elif st.session_state._ver_phase == 1:
            if "_rep_start" not in st.session_state:
                st.session_state._rep_start = time.time()
                st.info("🔊 Replay: Do I understand your problem? Reply now.")
                if os.path.isfile(tts_path): st.audio(tts_path, autoplay=True)
                
            elapsed_2 = time.time() - st.session_state._rep_start
            st.caption(f"⏳ Final response window: {max(0, int(3 - elapsed_2))}s")
            
            conf_audio_2 = st.audio_input("🎙️ Final Response", key="mic_final", disabled=False)
            if conf_audio_2:
                with open("confirm_2.wav", "wb") as f: f.write(conf_audio_2.getvalue())
                with st.spinner("Processing..."):
                    raw_conf, _ = transcribe_audio("confirm_2.wav")
                    parsed = parse_confirmation(raw_conf)
                
                st.session_state.attempts += 1
                if parsed["intent"] == "confirmed":
                    st.session_state.stage = "agent_ready"
                    st.success("✅ Verified!")
                    st.rerun()
                elif parsed["intent"] == "denied":
                    if st.session_state.attempts >= 2 or data.get("handover", False):
                        st.session_state.stage = "handover"
                        st.error("🔄 Handover triggered.")
                    else:
                        st.session_state.stage = "input_record"
                    st.rerun()
                else:
                    st.session_state.stage = "input_record"
                    st.rerun()
                    
            if elapsed_2 >= 3:
                st.warning("⏰ No further response detected. Transferring to human agent with available context...")
                st.session_state.stage = "handover"
<<<<<<< HEAD
                st.rerun()
=======
            else:
                st.warning("⚠️ Please speak clearly again.")
                st.session_state.stage = "input_record"
            st.rerun()
        else:
            st.markdown(f"<div class='info-card' style='border-left: 4px solid var(--warning-orange);'>📝 <strong>AI heard:</strong> '{parsed['summary']}'</div>", unsafe_allow_html=True)
            st.warning("❓ Did you mean 'Yes' or 'No'?")
            st.session_state.stage = "input_record"
            st.rerun()
>>>>>>> f454089 (Refactor UI/UX: Modern SaaS dashboard style)

# ═══════════════════════════════════════════════════════════════════════════
# STAGE: decision (transient — routes immediately)
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "decision":
    st.session_state.attempts += 1
    data = st.session_state.ai_data or {}

    if st.session_state.attempts >= 2 or data.get("handover", False):
        st.session_state.stage = "handover"
    else:
        st.warning("⚠️ Not understood. Please try again.")
        st.session_state.stage = "input_record"
    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# STAGE: agent_ready
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "agent_ready":
    data = st.session_state.ai_data or {}
    st.markdown("<h2 class='branded-subheader'>✅ Agent Dashboard — Verified Issue</h2>", unsafe_allow_html=True)

    sentiment = data.get("sentiment", "calm")
    conf = data.get("confidence", 0)
    lang = data.get("language", "—")
    issue = data.get('normalized_issue', '—')

    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"""
            <div class='info-card' style='height: 100%;'>
                <h4>Metadata</h4>
                <div style='margin-bottom: 0.8rem;'>{get_sentiment_pill(sentiment)}</div>
                <div style='margin-bottom: 0.8rem;'>{get_confidence_pill(conf)}</div>
                <div>{get_language_pill(lang)}</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class='info-card' style='height: 100%; border-left: 4px solid var(--success-green);'>
                <h4>Verified Issue</h4>
                <p style='font-size: 1.1rem; color: #1f2937; margin-bottom: 0;'>{issue}</p>
            </div>
        """, unsafe_allow_html=True)

    with st.expander("📝 Raw Transcript", expanded=False):
        st.code(data.get("raw_text", "—"), language=None)

    st.markdown("<br>", unsafe_allow_html=True)
    agent_note = st.text_area(
        "Agent Notes / Corrections",
        value=data.get("normalized_issue", ""),
        help="Edit if the AI summary needs correction.",
    )

    if st.button("✅ Log & Close Call", use_container_width=True):
        feedback = dict(data)
        feedback["citizen_response"] = "Confirmed"
        feedback["agent_correction"] = agent_note if agent_note != data.get("normalized_issue", "") else ""
        log_feedback(feedback)
        st.toast("📋 Feedback logged!", icon="✅")
        _reset()
        st.rerun()

    st.markdown("<h2 class='branded-subheader'>📊 Feedback Log</h2>", unsafe_allow_html=True)
    if os.path.isfile(FEEDBACK_FILE):
        try:
            df = pd.read_csv(FEEDBACK_FILE)
            st.dataframe(df, use_container_width=True)
        except Exception:
            st.caption("Feedback file is empty or unreadable.")
    else:
        st.caption("No feedback recorded yet.")

# ═══════════════════════════════════════════════════════════════════════════
# STAGE: handover
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "handover":
    data = st.session_state.ai_data or {}

    st.markdown("<h2 class='branded-subheader' style='border-bottom-color: var(--danger-red); color: var(--danger-red);'>⚠️ HUMAN AGENT TAKING OVER</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"""
            <div class='info-card'>
                <h4>Metadata</h4>
                <div style='margin-bottom: 0.8rem;'>{get_sentiment_pill(data.get("sentiment", "calm"))}</div>
                <div>{get_confidence_pill(data.get('confidence', 0))}</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class='info-card' style='border-left: 4px solid var(--danger-red);'>
                <p style='color: var(--danger-red); font-weight: 600; margin-bottom: 1rem;'>
                    The AI could not confidently verify the caller's issue.<br>
                    A human agent will now handle this call directly.
                </p>
                <h4>Last AI Summary</h4>
                <p style='font-size: 1.1rem; margin-bottom: 0;'>{data.get('normalized_issue', '—')}</p>
            </div>
        """, unsafe_allow_html=True)

    with st.expander("📝 Raw Transcript"):
        st.code(data.get("raw_text", "—"), language=None)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 End Call", use_container_width=True):
        feedback = dict(data)
        feedback["citizen_response"] = "Handover"
        feedback["agent_correction"] = ""
        log_feedback(feedback)
        st.toast("📋 Call logged as handover.", icon="🔄")
        _reset()
        st.rerun()
