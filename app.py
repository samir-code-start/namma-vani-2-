"""app.py — Namma Vanni Streamlit UI: Record → Verify → Confirm/Handover → Agent Dashboard."""

import os
import streamlit as st
import pandas as pd
from engine import process_audio, log_feedback, MOCK_MODE, FEEDBACK_FILE, parse_confirmation, transcribe_audio

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Namma Vanni — 1092 AI Helpline", page_icon="📞", layout="wide")

SENTIMENT_BADGES = {
    "calm": "🟢 Calm",
    "confused": "🟡 Confused",
    "urgent": "🟠 Urgent",
    "distressed": "🔴 Distressed",
    "angry": "🟥 Angry",
}

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
    "<h1 style='text-align:center'>📞 Namma Vanni — 1092 AI Helpline</h1>"
    "<p style='text-align:center;opacity:0.6'>Voice-to-voice citizen assistant for Karnataka</p>",
    unsafe_allow_html=True,
)
if MOCK_MODE:
    st.caption("⚙️ Running in **MOCK MODE** — no API calls.")

# ═══════════════════════════════════════════════════════════════════════════
# STAGE: input_record
# ═══════════════════════════════════════════════════════════════════════════
if st.session_state.stage == "input_record":
    st.subheader("🎤 Start Call")
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
elif st.session_state.stage == "verify":
    data = st.session_state.ai_data
    if data is None:
        st.error("No analysis data found. Returning to start.")
        _reset()
        st.rerun()

    st.subheader("🔍 Verification")

    col1, col2, col3 = st.columns(3)
    col1.metric("Language", data.get("language", "—").upper())
    col2.metric("Confidence", f"{data.get('confidence', 0) * 100:.0f}%")
    sentiment = data.get("sentiment", "calm")
    col3.markdown(f"**Sentiment:** {SENTIMENT_BADGES.get(sentiment, sentiment)}")

    st.info(f"**AI Summary:** {data.get('normalized_issue', '—')}")
    base_prompt = data.get("verification_prompt", "")
    normalized_issue = data.get("normalized_issue", "")
    appended_prompt = f"{base_prompt} I heard you say '{normalized_issue}'. Is this correct? Say Yes or No."
    st.markdown(f"🗣️ *\"{appended_prompt}\"*")

    # Auto-play TTS
    tts_path = data.get("verify_tts_path", "verify.mp3")
    if os.path.isfile(tts_path) and os.path.getsize(tts_path) > 0:
        st.audio(tts_path, autoplay=True, format="audio/mpeg")
    else:
        st.caption("🔇 TTS audio unavailable.")

    YES_KEYWORDS = ["yes", "ಹೌದು", "correct", "ಸರಿ", "right", "yeah", "yep", "ಹಾ", "ಹಾಂ"]
    NO_KEYWORDS = ["no", "ಇಲ್ಲ", "wrong", "ತಪ್ಪು", "nope", "ಬೇಡ"]

    def parse_yes_no(text: str):
        parsed = parse_confirmation(text)
        if parsed is not None:
            return parsed
        text = text.lower().strip()
        if any(kw in text for kw in YES_KEYWORDS):
            return True
        if any(kw in text for kw in NO_KEYWORDS):
            return False
        return None

    user_resp = st.text_input("Type Yes or No", key="verify_input")
    if user_resp:
        is_yes = parse_yes_no(user_resp)
        if is_yes is True:
            st.session_state.stage = "handover" if data.get("handover", False) else "agent_ready"
            st.rerun()
        else:
            st.session_state.stage = "decision"
            st.rerun()

    voice_resp = st.audio_input("Speak Yes or No", key="verify_voice_input")
    if voice_resp is not None:
        verify_path = "verify_response.wav"
        with open(verify_path, "wb") as f:
            f.write(voice_resp.getvalue())

        with st.spinner("ðŸ”„ Verifying your responseâ€¦"):
            verify_text, _ = transcribe_audio(verify_path)
            is_yes = parse_confirmation(verify_text)

        if is_yes is True:
            st.session_state.stage = "handover" if data.get("handover", False) else "agent_ready"
            st.rerun()
        else:
            st.session_state.stage = "decision"
            st.rerun()

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
    st.subheader("✅ Agent Dashboard — Verified Issue")

    sentiment = data.get("sentiment", "calm")
    st.success(
        f"**Issue:** {data.get('normalized_issue', '—')}  \n"
        f"**Sentiment:** {SENTIMENT_BADGES.get(sentiment, sentiment)} &nbsp;|&nbsp; "
        f"**Confidence:** {data.get('confidence', 0) * 100:.0f}% &nbsp;|&nbsp; "
        f"**Language:** {data.get('language', '—').upper()}"
    )

    with st.expander("📝 Raw Transcript", expanded=False):
        st.code(data.get("raw_text", "—"), language=None)

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

    st.divider()
    st.subheader("📊 Feedback Log")
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

    st.error("⚠️ **HUMAN AGENT TAKING OVER**")
    st.markdown(
        "The AI could not confidently verify the caller's issue.  \n"
        "A human agent will now handle this call directly.  \n\n"
        f"**Last AI Summary:** {data.get('normalized_issue', '—')}  \n"
        f"**Sentiment:** {SENTIMENT_BADGES.get(data.get('sentiment', ''), '—')}  \n"
        f"**Confidence:** {data.get('confidence', 0) * 100:.0f}%"
    )

    with st.expander("📝 Raw Transcript"):
        st.code(data.get("raw_text", "—"), language=None)

    if st.button("🔄 End Call", use_container_width=True):
        feedback = dict(data)
        feedback["citizen_response"] = "Handover"
        feedback["agent_correction"] = ""
        log_feedback(feedback)
        st.toast("📋 Call logged as handover.", icon="🔄")
        _reset()
        st.rerun()
