import os
import time
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
# Dark Command Center — Global Design System
# ---------------------------------------------------------------------------
def init_global_styles():
    st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
:root {
  --bg-base: #0d0f14; --bg-surface: #161921; --bg-surface-2: #1e2230;
  --border-subtle: rgba(255,255,255,0.07); --border-medium: rgba(255,255,255,0.12);
  --text-primary: #f0f2f8; --text-secondary: #9da5be; --text-muted: #6b7080;
  --accent-red: #e8445a; --accent-green: #1a9e5c; --accent-blue: #3b8af5; --accent-yellow: #d4a017;
}
[data-testid="stAppViewContainer"] { background-color: var(--bg-base) !important; color: var(--text-primary) !important; }
[data-testid="stHeader"] { background-color: var(--bg-base) !important; }
[data-testid="stVerticalBlock"] > div { padding-top: 0 !important; }
.block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; max-width: 900px; }
h1, h2, h3, h4, label, p, span, li { font-family: 'Syne', sans-serif !important; color: var(--text-primary) !important; }
.font-mono { font-family: 'Space Mono', monospace !important; letter-spacing: -0.5px; }

/* Nav Bar */
.nav-bar { display:flex; justify-content:space-between; align-items:center; padding:16px 0; border-bottom:1px solid var(--border-subtle); margin-bottom:24px; margin-top:8px; }
.nav-brand-group { display:flex; flex-direction:column; }
.nav-icon-bg { width:38px; height:38px; background:linear-gradient(135deg,#e8445a,#ff7b54); border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:18px; }
.nav-badge { background:rgba(232,68,90,0.15); color:var(--accent-red); border:1px solid rgba(232,68,90,0.3); font-size:11px; font-family:'Space Mono'; padding:4px 10px; border-radius:6px; text-transform:uppercase; letter-spacing:1px; }

/* Cards */
.card { background:var(--bg-surface); border:1px solid var(--border-subtle); border-radius:12px; padding:18px; margin-bottom:12px; }
.section-label { font-size:11px; text-transform:uppercase; letter-spacing:0.8px; color:var(--text-muted) !important; font-weight:700; margin-bottom:8px; display:block; }

/* Pills */
.pill { display:inline-flex; align-items:center; gap:6px; padding:5px 12px; border-radius:100px; font-size:12px; font-weight:600; }
.lang-pill { background:rgba(96,108,200,0.18); color:#8896e8 !important; border:1px solid rgba(96,108,200,0.25); }
.conf-pill { background:rgba(59,164,243,0.15); color:#58aef5 !important; border:1px solid rgba(59,164,243,0.25); }
.emotion-pill { border-radius:100px; font-size:12px; font-weight:600; padding:5px 12px; display:inline-flex; align-items:center; gap:6px; }
.emotion-calm { background:rgba(61,214,140,0.15); color:#3dd68c !important; border:1px solid rgba(61,214,140,0.25); }
.emotion-confused { background:rgba(212,160,23,0.15); color:#d4a017 !important; border:1px solid rgba(212,160,23,0.25); }
.emotion-urgent { background:rgba(255,123,84,0.15); color:#ff7b54 !important; border:1px solid rgba(255,123,84,0.25); }
.emotion-distressed { background:rgba(232,68,90,0.15); color:#e8445a !important; border:1px solid rgba(232,68,90,0.3); }
.emotion-angry { background:rgba(200,30,30,0.2); color:#ff4444 !important; border:1px solid rgba(200,30,30,0.35); }
.dot-blink::before { content:''; width:6px; height:6px; border-radius:50%; background:currentColor; display:inline-block; animation:blink 1s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
@keyframes pulse-ring { 0%{transform:scale(1);opacity:0.6} 100%{transform:scale(1.4);opacity:0} }

/* Buttons */
.stButton > button { width:100% !important; border-radius:10px !important; font-weight:700 !important; font-family:'Syne' !important; border:1px solid var(--border-medium) !important; background:var(--bg-surface) !important; color:var(--text-secondary) !important; transition:all 0.2s ease; }
.stButton > button:hover { background:var(--bg-surface-2) !important; border-color:var(--accent-blue) !important; color:var(--text-primary) !important; }
.stButton > button[kind="primary"] { background:var(--accent-red) !important; color:white !important; border-color:var(--accent-red) !important; }
.stButton > button[kind="primary"]:hover { background:#c7374b !important; }

/* Inputs & Expanders */
[data-testid="stAudioInput"] > div { background:var(--bg-surface) !important; border:1px solid var(--border-subtle) !important; border-radius:10px !important; }
[data-testid="stTextArea"] textarea { background:var(--bg-surface) !important; color:var(--text-secondary) !important; border:1px solid var(--border-subtle) !important; border-radius:10px !important; font-family:'Syne' !important; }
[data-testid="stExpander"] { background:var(--bg-surface) !important; border:1px solid var(--border-subtle) !important; border-radius:10px !important; }
[data-testid="stExpander"] summary span { color:var(--text-secondary) !important; }
[data-testid="stExpander"] .stCodeBlock code { background:var(--bg-base) !important; color:var(--text-secondary) !important; }

/* Alerts */
.stAlert { border-radius:10px !important; }
div[data-testid="stCaptionContainer"] p { color:var(--text-muted) !important; }

/* Toast */
[data-testid="stToast"] { background:var(--bg-surface) !important; color:var(--text-primary) !important; border:1px solid var(--border-subtle) !important; }

/* Dataframe */
[data-testid="stDataFrame"] { background:var(--bg-surface) !important; border:1px solid var(--border-subtle) !important; border-radius:10px !important; }

/* Spinner */
.stSpinner > div { color:var(--text-muted) !important; }
</style>""", unsafe_allow_html=True)

init_global_styles()

# ---------------------------------------------------------------------------
# UI Helpers
# ---------------------------------------------------------------------------
SENTIMENT_MAP = {
    "calm": ("emotion-calm", "🟢", "Calm"),
    "confused": ("emotion-confused", "🟡", "Confused"),
    "urgent": ("emotion-urgent", "🟠", "Urgent"),
    "distressed": ("emotion-distressed", "🔴", "Distressed"),
    "angry": ("emotion-angry", "🟥", "Angry"),
}

def _sentiment_pill(sentiment):
    cls, icon, label = SENTIMENT_MAP.get(sentiment, SENTIMENT_MAP["calm"])
    blink = " dot-blink" if sentiment in ("distressed", "angry") else ""
    return f"<span class='emotion-pill {cls}{blink}'>{icon} {label}</span>"

def _conf_pill(conf):
    return f"<span class='pill conf-pill'>{conf * 100:.0f}%</span>"

def _lang_pill(lang):
    return f"<span class='pill lang-pill'>{lang.upper()}</span>"

def _render_header():
    st.markdown('''<div class="nav-bar"><div class="nav-brand-group"><div style="display:flex;align-items:center;gap:12px;"><div class="nav-icon-bg">📞</div><h1 style="margin:0;font-family:'Syne',sans-serif;font-weight:800;font-size:18px;color:var(--text-primary);line-height:1.2;">Namma Vanni — <span style="color:var(--accent-red)">1092</span> AI Helpline</h1></div><p style="margin:6px 0 0 50px;font-size:12px;color:var(--text-secondary);letter-spacing:0.3px;">Voice-to-voice citizen assistant for Karnataka</p></div><div class="nav-badge">LIVE</div></div>''', unsafe_allow_html=True)

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
_render_header()
if MOCK_MODE:
    st.caption("⚙️ Running in **MOCK MODE** — no API calls.")

# ═══════════════════════════════════════════════════════════════════════════
# STAGE: input_record
# ═══════════════════════════════════════════════════════════════════════════
if st.session_state.stage == "input_record":
    # Pulsing mic hero
    st.markdown('''<div style="display:flex;flex-direction:column;align-items:center;margin:40px 0 20px 0;">
        <div style="position:relative;width:150px;height:150px;display:flex;justify-content:center;align-items:center;">
            <div style="position:absolute;width:120px;height:120px;border-radius:50%;background:rgba(232,68,90,0.08);border:2px solid rgba(232,68,90,0.2);animation:pulse-ring 2.2s infinite;"></div>
            <div style="position:absolute;width:150px;height:150px;border-radius:50%;border:1px solid rgba(232,68,90,0.08);animation:pulse-ring 2.2s infinite 0.4s;"></div>
            <div style="width:64px;height:64px;border-radius:50%;background:var(--accent-red);display:flex;align-items:center;justify-content:center;font-size:24px;z-index:2;box-shadow:0 0 30px rgba(232,68,90,0.3);">🎤</div>
        </div>
        <h2 style="margin-top:24px;font-weight:800;">Start Emergency Call</h2>
        <p style="color:var(--text-muted) !important;font-size:14px;margin-top:4px;">Speak clearly. The AI will triage your issue in real-time.</p>
    </div>''', unsafe_allow_html=True)

    audio_bytes = st.audio_input("Speak your concern or upload audio")

    # Stats grid
    st.markdown('''<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:24px;">
        <div class="card" style="margin:0;text-align:center;"><span class="section-label">Active Calls</span><br><span style="font-size:20px;font-weight:700;color:var(--accent-red) !important;">14</span></div>
        <div class="card" style="margin:0;text-align:center;"><span class="section-label">Resolved Today</span><br><span style="font-size:20px;font-weight:700;color:var(--accent-green) !important;">238</span></div>
        <div class="card" style="margin:0;text-align:center;"><span class="section-label">Avg Response</span><br><span style="font-size:20px;font-weight:700;">1m 12s</span></div>
    </div>''', unsafe_allow_html=True)

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
    data = st.session_state.ai_data or {}
    if not data:
        st.error("No analysis data found. Returning to start.")
        _reset()
        st.rerun()

    lang = data.get("language", "en")
    tts_path = data.get("verify_tts_path", "")
    sentiment = data.get("sentiment", "calm")
    conf = data.get("confidence", 0)

    # Initialize phase tracking once
    if "_ver_start" not in st.session_state:
        st.session_state._ver_start = time.time()
        st.session_state._ver_phase = 0
        st.session_state._mic_initialized = False

    elapsed_1 = time.time() - st.session_state._ver_start

    # --- Analysis pills ---
    st.markdown(f'''<div class="card"><span class="section-label">Caller Analysis</span>
        <div style="display:flex;gap:8px;margin-top:4px;flex-wrap:wrap;">
            {_lang_pill(lang)} {_conf_pill(conf)} {_sentiment_pill(sentiment)}
        </div>
    </div>''', unsafe_allow_html=True)

    # --- AI Summary ---
    st.markdown(f'''<div class="card"><span class="section-label">AI Summary</span>
        <h3 style="margin:4px 0 0 0;font-weight:600;">{data.get("normalized_issue", "—")}</h3>
    </div>''', unsafe_allow_html=True)

    # --- Voice Prompt ---
    base_prompt = data.get("verification_prompt", "")
    normalized_issue = data.get("normalized_issue", "")
    appended_prompt = f"{base_prompt} I heard you say '{normalized_issue}'. Is this correct? Say Yes or No."

    st.markdown(f'''<div class="card" style="display:flex;gap:12px;align-items:start;">
        <div style="width:32px;height:32px;min-width:32px;background:rgba(96,108,200,0.2);border-radius:8px;display:flex;align-items:center;justify-content:center;">🤖</div>
        <div><span class="section-label">AI Voice Prompt</span><p style="margin:0;font-style:italic;color:var(--text-secondary) !important;line-height:1.5;">"{appended_prompt}"</p></div>
    </div>''', unsafe_allow_html=True)

    # Create a single placeholder for the audio player (prevents duplicates)
    audio_placeholder = st.empty()

    def play_verification_audio():
        path = data.get("verify_tts_path", "")
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            with audio_placeholder.container():
                st.info("🔊 Tap play to listen to verification.")
                st.audio(path, key="ver_tts_player_unique")
        else:
            with audio_placeholder.container():
                st.caption("🔇 TTS audio unavailable.")

    # Play initial summary audio once
    if "_verified_initial_play" not in st.session_state:
        play_verification_audio()
        st.session_state._verified_initial_play = True
    elif st.session_state._ver_phase == 1:
        # Replay phase — overwrite the same container
        with audio_placeholder.container():
            st.info("🔊 Replaying: Do I understand your problem? Reply now.")
            if os.path.isfile(tts_path):
                st.audio(tts_path, key="replay_tts_player_unique")
            else:
                st.warning("⚠️ No further voice prompt available.")

    # PHASE 0: Single-click mic initialization (browser security requirement)
    if not st.session_state._mic_initialized:
        st.warning("⚠️ Click ONCE to activate live microphone. Keep device close.")
        if st.button("🎤 Activate Live Microphone", key="btn_init_mic", type="primary"):
            st.session_state._mic_initialized = True
            st.rerun()
    else:
        # PHASE 1: Initial 10-second auto-listening window
        if st.session_state._ver_phase == 0:
            st.markdown('<div class="card" style="border-left:3px solid var(--accent-red);"><span class="section-label" style="color:var(--accent-red) !important;">🔴 LISTENING</span><p style="color:var(--text-secondary) !important;margin:0;">Speak Yes / No / Han / Sari / Illa now.</p></div>', unsafe_allow_html=True)
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
                    st.markdown(f'''<div class="card" style="border-left:3px solid var(--accent-yellow);"><span class="section-label">AI heard</span><p style="color:var(--text-secondary) !important;margin:0;">"{parsed['summary']}"</p></div>''', unsafe_allow_html=True)
                    st.warning("❓ Did you mean Yes or No?")
                    st.session_state.stage = "input_record"
                    st.rerun()

            if elapsed_1 >= 10:
                st.session_state._ver_phase = 1
                st.rerun()

        # PHASE 2: Final Response Window
        elif st.session_state._ver_phase == 1:
            if "_rep_start" not in st.session_state:
                st.session_state._rep_start = time.time()

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
                        st.warning("⚠️ Please speak clearly again.")
                        st.session_state.stage = "input_record"
                    st.rerun()
                else:
                    st.markdown(f'''<div class="card" style="border-left:3px solid var(--accent-yellow);"><span class="section-label">AI heard</span><p style="color:var(--text-secondary) !important;margin:0;">"{parsed['summary']}"</p></div>''', unsafe_allow_html=True)
                    st.warning("❓ Did you mean 'Yes' or 'No'?")
                    st.session_state.stage = "input_record"
                    st.rerun()

            if elapsed_2 >= 3:
                st.warning("⏰ No further response detected. Transferring to human agent with available context...")
                st.session_state.stage = "handover"
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
    sentiment = data.get("sentiment", "calm")
    conf = data.get("confidence", 0)
    lang = data.get("language", "—")
    issue = data.get('normalized_issue', '—')

    # Success banner
    st.markdown('''<div style="background:rgba(61,214,140,0.08);border:1px solid rgba(61,214,140,0.2);border-radius:12px;padding:14px 18px;margin-bottom:20px;display:flex;align-items:center;gap:8px;">
        <span style="color:var(--accent-green);">✅</span>
        <span style="font-size:14px;font-weight:700;color:var(--accent-green) !important;">Issue Verified — Agent Dashboard Active</span>
    </div>''', unsafe_allow_html=True)

    # Metadata + Issue grid
    st.markdown(f'''<div style="display:grid;grid-template-columns:220px 1fr;gap:12px;margin-bottom:20px;">
        <div class="card"><span class="section-label">Metadata</span>
            <div style="margin-bottom:8px;">{_sentiment_pill(sentiment)}</div>
            <div style="margin-bottom:8px;">{_conf_pill(conf)}</div>
            <div>{_lang_pill(lang)}</div>
        </div>
        <div class="card" style="border-left:3px solid var(--accent-green);">
            <span class="section-label">Verified Issue</span>
            <h2 style="font-weight:800;margin:4px 0 0 0;font-size:20px;">{issue}</h2>
        </div>
    </div>''', unsafe_allow_html=True)

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

    # Feedback log
    st.markdown('<div style="margin-top:24px;"><span class="section-label" style="font-size:13px;">📊 Feedback Log</span></div>', unsafe_allow_html=True)
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
    sentiment = data.get("sentiment", "calm")
    conf = data.get("confidence", 0)
    issue = data.get('normalized_issue', '—')

    # Red alert banner
    st.markdown(f'''<div style="background:rgba(232,68,90,0.08);border:1px solid rgba(232,68,90,0.25);border-radius:12px;padding:18px 20px;margin-bottom:20px;display:flex;gap:16px;align-items:start;">
        <div style="width:40px;height:40px;min-width:40px;background:rgba(232,68,90,0.15);border-radius:10px;display:flex;align-items:center;justify-content:center;color:var(--accent-red);font-size:18px;">⚠️</div>
        <div>
            <span style="font-size:16px;font-weight:800;color:var(--accent-red) !important;display:block;margin-bottom:4px;">HUMAN AGENT TAKING OVER</span>
            <span style="font-size:13px;color:rgba(232,68,90,0.7) !important;">AI could not confidently verify the caller's issue. A human agent will now handle this call directly.</span>
        </div>
    </div>''', unsafe_allow_html=True)

    # Metadata + Summary grid
    st.markdown(f'''<div style="display:grid;grid-template-columns:220px 1fr;gap:12px;margin-bottom:16px;">
        <div class="card"><span class="section-label">Metadata</span>
            <div style="margin-bottom:8px;">{_sentiment_pill(sentiment)}</div>
            <div>{_conf_pill(conf)}</div>
        </div>
        <div class="card" style="border-left:3px solid var(--accent-red);">
            <span class="section-label">Last AI Summary</span>
            <h2 style="font-weight:700;margin:4px 0 0 0;font-size:20px;">{issue}</h2>
        </div>
    </div>''', unsafe_allow_html=True)

    raw_text = data.get("raw_text", "No transcript available.")
    st.markdown(f'''<div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 12px; padding: 20px; margin-bottom: 16px;">
        <span class="section-label">📄 Raw Transcript</span>
        <p style="font-family: 'Space Mono', monospace; font-size: 13px; line-height: 1.6; color: var(--text-secondary) !important; margin-top: 12px; white-space: pre-wrap; word-break: break-word; max-height: 150px; overflow-y: auto;">
            {raw_text}
        </p>
    </div>''', unsafe_allow_html=True)

    if st.button("🔄 End Call", use_container_width=True):
        feedback = dict(data)
        feedback["citizen_response"] = "Handover"
        feedback["agent_correction"] = ""
        log_feedback(feedback)
        st.toast("📋 Call logged as handover.", icon="🔄")
        _reset()
        st.rerun()
