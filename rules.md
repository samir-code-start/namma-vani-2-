
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