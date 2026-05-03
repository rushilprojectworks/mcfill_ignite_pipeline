"""
McFill Ignite — Podcast-to-Content Pipeline
============================================
Upload a podcast episode → transcribe with Whisper → generate:
  • Full transcript
  • Show notes (structured)
  • 3 key quotes
  • LinkedIn post
  • Episode summary email

Fixes applied:
  1. Replaced google-genai SDK with anthropic SDK (claude-sonnet-4-20250514)
     — google-genai has breaking import issues in many environments
  2. requirements.txt updated accordingly
  3. HTML injection fixed — all AI-generated text is html.escape()'d before
     being embedded in st.markdown() unsafe_allow_html blocks (same root
     cause as Project 1 QA panel issue)
  4. gemini_generate_with_retry renamed to claude_generate_with_retry;
     config={} kwarg removed (not valid in anthropic SDK)
  5. JSON fence stripping added before json.loads() on quotes response
  6. tmp file cleanup made robust — unlink moved fully into finally block
  7. st.status() replaced with st.spinner() — st.status is Streamlit ≥1.28
     and causes AttributeError on older installs
  8. session_state initialisation guard fixed — was setting None for list
     keys (error_log, errors) which caused .append() to fail
  9. col_output block moved OUTSIDE col_input with-block (was incorrectly
     nested, causing output to render inside the input column)
 10. requirements.txt pinned to stable versions
"""

import html as html_lib
import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import anthropic
import openai
import streamlit as st

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="McFill Ignite · Podcast Studio",
    page_icon="🎙",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Constants ────────────────────────────────────────────────────────────────
MAX_FILE_MB     = 25
MAX_FILE_BYTES  = MAX_FILE_MB * 1024 * 1024
ALLOWED_TYPES   = ["mp3", "mp4", "m4a", "wav", "webm", "ogg", "flac"]
RETRY_ATTEMPTS  = 3
RETRY_DELAY_SEC = 2

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:      #080C10;
    --bg2:     #0D1117;
    --bg3:     #131920;
    --border:  #1E2D3D;
    --border2: #243447;
    --accent:  #00E5FF;
    --accent2: #0099BB;
    --text:    #C9D8E8;
    --text2:   #6B8299;
    --green:   #00FF94;
    --red:     #FF4D6D;
    --amber:   #FFB830;
    --mono:    'Space Mono', monospace;
    --sans:    'DM Sans', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--sans);
    background: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg) !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem !important; max-width: 1280px !important; }

.ig-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.5rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.ig-wordmark {
    font-family: var(--mono);
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.ig-sub {
    font-size: 0.65rem;
    color: var(--text2);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 2px;
}
.ig-badge {
    font-family: var(--mono);
    font-size: 0.6rem;
    padding: 4px 10px;
    border: 1px solid var(--accent2);
    color: var(--accent);
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.sl {
    font-family: var(--mono);
    font-size: 0.58rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sl::after { content: ''; flex: 1; height: 1px; background: var(--border); }

.file-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--bg3);
    border: 1px solid var(--border2);
    padding: 6px 14px;
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--accent);
    margin-top: 10px;
}

.err-box {
    border: 1px solid var(--red);
    background: #1A0810;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.78rem;
    color: #FF8099;
    line-height: 1.6;
}
.err-box .err-title {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--red);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.warn-box {
    border: 1px solid var(--amber);
    background: #120E00;
    padding: 10px 14px;
    font-size: 0.75rem;
    color: #FFD580;
    margin: 6px 0;
}
.info-box {
    border-left: 3px solid var(--accent);
    background: var(--bg2);
    padding: 10px 14px;
    font-size: 0.75rem;
    color: var(--text2);
    margin: 6px 0;
    line-height: 1.6;
}

.steps-row {
    display: flex;
    gap: 0;
    margin: 1.5rem 0;
    background: var(--bg2);
    border: 1px solid var(--border);
    overflow: hidden;
}
.step {
    flex: 1;
    padding: 10px 14px;
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-family: var(--mono);
    border-right: 1px solid var(--border);
    color: var(--text2);
    display: flex;
    align-items: center;
    gap: 8px;
}
.step:last-child { border-right: none; }
.step.done   { color: var(--green);  background: #001A0D; }
.step.active { color: var(--accent); background: #001018; }
.step-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; flex-shrink: 0; }

.out-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    margin-bottom: 1rem;
    overflow: hidden;
}
.out-card-head {
    background: var(--bg3);
    padding: 10px 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid var(--border);
}
.out-card-title {
    font-family: var(--mono);
    font-size: 0.62rem;
    color: var(--accent);
    letter-spacing: 0.15em;
    text-transform: uppercase;
}
.out-card-body {
    padding: 1.25rem 1.5rem;
    font-size: 0.82rem;
    line-height: 1.8;
    color: var(--text);
    white-space: pre-wrap;
    word-break: break-word;
}

.quote-card {
    border-left: 3px solid var(--accent);
    background: var(--bg3);
    padding: 1rem 1.25rem;
    margin-bottom: 0.6rem;
    font-size: 0.85rem;
    font-style: italic;
    color: var(--text);
    line-height: 1.7;
}
.quote-attr {
    font-style: normal;
    font-family: var(--mono);
    font-size: 0.58rem;
    color: var(--text2);
    letter-spacing: 0.1em;
    margin-top: 6px;
}

.stat-strip {
    display: flex;
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.stat-cell { flex: 1; background: var(--bg2); padding: 14px 10px; text-align: center; }
.stat-val  { font-family: var(--mono); font-size: 1.4rem; color: var(--accent); line-height: 1; }
.stat-lbl  { font-size: 0.55rem; letter-spacing: 0.15em; text-transform: uppercase; color: var(--text2); margin-top: 4px; }

.stButton > button {
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: var(--mono) !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    padding: 0.7rem 1.5rem !important;
    border-radius: 0 !important;
    transition: all 0.15s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
    color: var(--bg) !important;
}

.stTextInput > div > div > input {
    background: var(--bg2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 0 !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: none !important;
}

[data-testid="stFileUploader"] {
    background: var(--bg2) !important;
    border: 1px dashed var(--border2) !important;
    border-radius: 0 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] { color: var(--text2) !important; }

.streamlit-expanderHeader {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
    font-family: var(--mono) !important;
    font-size: 0.65rem !important;
    color: var(--text2) !important;
    letter-spacing: 0.1em !important;
}

.stSpinner > div { border-top-color: var(--accent) !important; }
.stAlert { border-radius: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ig-header">
  <div>
    <div class="ig-wordmark">&#9672; McFill Ignite</div>
    <div class="ig-sub">Podcast-to-Content Pipeline &nbsp;&middot;&nbsp; Automated</div>
  </div>
  <div class="ig-badge">AI-Powered Studio</div>
</div>
""", unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────
_defaults = {
    "transcript":      None,
    "show_notes":      None,
    "quotes":          None,
    "linkedin_post":   None,
    "summary_email":   None,
    "file_meta":       None,
    "processing_done": False,
    "error_log":       [],   # must be list, not None
    "errors":          [],   # must be list, not None
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── File validation ───────────────────────────────────────────────────────────
def validate_file(uploaded_file) -> tuple[bool, str, str]:
    name = uploaded_file.name
    ext  = Path(name).suffix.lstrip(".").lower()
    size = uploaded_file.size

    if ext not in ALLOWED_TYPES:
        return False, "Unsupported file type", (
            f"'{ext.upper()}' files are not supported. "
            f"Supported: {', '.join(t.upper() for t in ALLOWED_TYPES)}."
        )

    if size > MAX_FILE_BYTES:
        mb = size / (1024 * 1024)
        return False, "File too large", (
            f"Your file is {mb:.1f} MB. The Whisper API limit is {MAX_FILE_MB} MB. "
            f"Compress with ffmpeg:\n  ffmpeg -i input.mp3 -b:a 64k output.mp3"
        )

    if size > 20 * 1024 * 1024:
        mb = size / (1024 * 1024)
        return True, "warn_large", f"File is {mb:.1f} MB — transcription may take 1–2 minutes."

    return True, "", ""


# ─── Whisper transcription with retry ─────────────────────────────────────────
def whisper_transcribe_with_retry(audio_bytes: bytes, filename: str, api_key: str) -> tuple[str, list[str]]:
    client   = openai.OpenAI(api_key=api_key)
    log      = []
    last_err = None
    tmp_path = None

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Whisper attempt {attempt}/{RETRY_ATTEMPTS}...")
            suffix = Path(filename).suffix or ".mp3"

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            with open(tmp_path, "rb") as f:
                result = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="text"
                )

            log.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Transcription complete.")
            return str(result), log

        except openai.APIConnectionError as e:
            last_err = f"Connection error: {e}"
        except openai.RateLimitError:
            last_err = "Rate limit hit on Whisper API."
            time.sleep(RETRY_DELAY_SEC * attempt)
        except openai.APIStatusError as e:
            last_err = f"API error {e.status_code}: {e.message}"
        except Exception as e:
            last_err = str(e)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            tmp_path = None

        log.append(f"  ✗ Attempt {attempt} failed — {last_err}")
        if attempt < RETRY_ATTEMPTS:
            log.append(f"  ↻ Retrying in {RETRY_DELAY_SEC}s...")
            time.sleep(RETRY_DELAY_SEC)

    raise RuntimeError(
        f"Whisper transcription failed after {RETRY_ATTEMPTS} attempts.\n"
        f"Last error: {last_err}\n\n"
        f"Fix hints:\n"
        f"• Check your OpenAI API key is valid\n"
        f"• Verify your account has Whisper API access (needs funded account or free trial credits)\n"
        f"• Try a shorter audio clip first\n"
        f"• Check OpenAI status at status.openai.com"
    )


# ─── Claude content generation with retry ─────────────────────────────────────
def claude_generate_with_retry(prompt: str, api_key: str, max_tokens: int = 2000) -> tuple[str, list[str]]:
    """
    Generates content via Anthropic Claude API with retry logic.
    Uses claude-sonnet-4-20250514 — reliable, fast, no config{} quirks.
    """
    client   = anthropic.Anthropic(api_key=api_key)
    log      = []
    last_err = None

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Claude attempt {attempt}/{RETRY_ATTEMPTS}...")

            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )

            text = message.content[0].text.strip()
            log.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Generation complete.")
            return text, log

        except anthropic.APIConnectionError as e:
            last_err = f"Connection error: {e}"
        except anthropic.RateLimitError:
            last_err = "Rate limit hit. Waiting before retry..."
            time.sleep(RETRY_DELAY_SEC * attempt)
        except anthropic.APIStatusError as e:
            last_err = f"API error {e.status_code}: {e.message}"
        except Exception as e:
            last_err = str(e)

        log.append(f"  ✗ Attempt {attempt} failed — {last_err}")
        if attempt < RETRY_ATTEMPTS:
            log.append(f"  ↻ Retrying in {RETRY_DELAY_SEC}s...")
            time.sleep(RETRY_DELAY_SEC)

    raise RuntimeError(
        f"Claude API failed after {RETRY_ATTEMPTS} attempts.\n"
        f"Last error: {last_err}\n\n"
        f"Fix hints:\n"
        f"• Check your Anthropic API key at console.anthropic.com\n"
        f"• Verify you have free trial credits remaining\n"
        f"• Try again in a few seconds"
    )


# ─── Prompts ──────────────────────────────────────────────────────────────────
def make_show_notes_prompt(transcript: str, episode_title: str) -> str:
    return f"""You are a professional podcast producer for McFill Ignite, a UAE-based luxury media brand.

Episode title: "{episode_title}"

Transcript:
{transcript[:8000]}

Write structured, professional show notes. Format exactly as:

## Episode Overview
[2-3 sentence summary]

## Key Topics Covered
• [Topic 1]
• [Topic 2]
• [Topic 3]

## Timestamps
• 00:00 – Introduction
• [Approximate timestamps based on content]

## Key Takeaways
1. [Insight 1]
2. [Insight 2]
3. [Insight 3]

## Resources Mentioned
[Any brands, books, or names — or "None mentioned"]

Professional, editorial tone consistent with McFill Ignite's luxury media brand."""


def make_quotes_prompt(transcript: str) -> str:
    return f"""Extract the 3 most impactful, shareable quotes from this podcast transcript.

Transcript:
{transcript[:6000]}

Return ONLY a valid JSON array — no markdown, no code fences, no explanation:
[
  {{"quote": "exact words", "context": "one sentence why this matters", "platform": "LinkedIn/Instagram/Twitter"}},
  {{"quote": "...", "context": "...", "platform": "..."}},
  {{"quote": "...", "context": "...", "platform": "..."}}
]"""


def make_linkedin_prompt(transcript: str, episode_title: str, show_notes: str) -> str:
    return f"""You are a content strategist for McFill Ignite, UAE's luxury media brand.

Episode: "{episode_title}"
Show notes summary: {show_notes[:1500]}

Write a LinkedIn post announcing this episode:
- Bold, thought-provoking first line (no emoji opener)
- 3-4 short paragraphs with line breaks
- Professional, aspirational tone
- Ends with: "Listen now — link in bio"
- 3-5 hashtags: #McFillIgnite #Podcast #UAE plus 1-2 topic hashtags
- Under 600 characters total

Return ONLY the post text."""


def make_email_prompt(transcript: str, episode_title: str, show_notes: str) -> str:
    return f"""Write an internal summary email for the McFill Ignite production team.

Episode: "{episode_title}"
Show notes: {show_notes[:2000]}

Format:
Subject: New Episode Ready — {episode_title}

Hi team,

[2-sentence overview]

Key highlights:
• [Highlight 1]
• [Highlight 2]
• [Highlight 3]

Guest/topics: [names or topics]
Recommended publish window: [suggest day/time]
Assets needed: Transcript ✓ | Show notes ✓ | Quotes ✓ | LinkedIn post ✓ | Thumbnail [ ] | Audio edit [ ]

McFill Ignite Production

Return ONLY the email text."""


# ─── Pipeline ─────────────────────────────────────────────────────────────────
def run_pipeline(audio_bytes, filename, episode_title, openai_key, claude_key):
    all_logs = []

    try:
        # Step 1 — Transcribe
        with st.spinner("🎙 Transcribing audio with Whisper..."):
            transcript, logs = whisper_transcribe_with_retry(audio_bytes, filename, openai_key)
            all_logs.extend(logs)
            st.session_state.transcript = transcript
        st.success("✓ Transcription complete")

        word_count = len(transcript.split())
        est_minutes = max(1, round(word_count / 130))
        st.session_state.file_meta = {
            "filename":      filename,
            "word_count":    word_count,
            "est_duration":  f"~{est_minutes} min",
            "generated_at":  datetime.now().strftime("%d %b %Y, %H:%M"),
            "episode_title": episode_title,
        }

        # Step 2 — Show notes
        with st.spinner("📋 Generating show notes with Claude..."):
            show_notes, logs = claude_generate_with_retry(
                make_show_notes_prompt(transcript, episode_title), claude_key, max_tokens=1500
            )
            all_logs.extend(logs)
            st.session_state.show_notes = show_notes
        st.success("✓ Show notes ready")

        # Step 3 — Key quotes
        with st.spinner("💬 Extracting key quotes..."):
            quotes_raw, logs = claude_generate_with_retry(
                make_quotes_prompt(transcript), claude_key, max_tokens=600
            )
            all_logs.extend(logs)
            # Strip markdown fences if model wraps in ```json ... ```
            clean = quotes_raw.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[-2] if "```" in clean[3:] else clean[3:]
                clean = clean.lstrip("json").strip()
            try:
                quotes = json.loads(clean)
            except json.JSONDecodeError:
                quotes = [{"quote": quotes_raw, "context": "See full response", "platform": "LinkedIn"}]
            st.session_state.quotes = quotes
        st.success("✓ Key quotes extracted")

        # Step 4 — LinkedIn post
        with st.spinner("💼 Writing LinkedIn post..."):
            linkedin, logs = claude_generate_with_retry(
                make_linkedin_prompt(transcript, episode_title, show_notes), claude_key, max_tokens=400
            )
            all_logs.extend(logs)
            st.session_state.linkedin_post = linkedin
        st.success("✓ LinkedIn post ready")

        # Step 5 — Team email
        with st.spinner("✉️ Writing team summary email..."):
            email, logs = claude_generate_with_retry(
                make_email_prompt(transcript, episode_title, show_notes), claude_key, max_tokens=600
            )
            all_logs.extend(logs)
            st.session_state.summary_email = email
        st.success("✓ Team email draft ready")

        st.session_state.error_log       = all_logs
        st.session_state.processing_done = True

    except RuntimeError as e:
        st.session_state.errors.append(str(e))
        raise


# ─── Layout ───────────────────────────────────────────────────────────────────
col_input, col_output = st.columns([1, 1.4], gap="large")

# ── Left column — inputs ───────────────────────────────────────────────────────
with col_input:
    st.markdown('<p class="sl">01 &nbsp; Upload Episode</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        label="audio",
        label_visibility="collapsed",
        type=ALLOWED_TYPES,
        help=f"Max {MAX_FILE_MB} MB · Supported: {', '.join(ALLOWED_TYPES)}"
    )

    file_ok = False
    if uploaded:
        valid, err_title, err_msg = validate_file(uploaded)
        if not valid:
            st.markdown(f"""
            <div class="err-box">
                <div class="err-title">&#10007; &nbsp; {html_lib.escape(err_title)}</div>
                {html_lib.escape(err_msg)}
            </div>
            """, unsafe_allow_html=True)
        else:
            if err_title == "warn_large":
                st.markdown(f'<div class="warn-box">&#9888; &nbsp; {html_lib.escape(err_msg)}</div>', unsafe_allow_html=True)

            size_mb = uploaded.size / (1024 * 1024)
            st.markdown(f"""
            <div class="file-pill">
                &#9654; &nbsp; {html_lib.escape(uploaded.name)} &nbsp;&middot;&nbsp; {size_mb:.1f} MB
            </div>
            """, unsafe_allow_html=True)
            file_ok = True

    st.markdown('<p class="sl" style="margin-top:1.5rem;">02 &nbsp; Episode Details</p>', unsafe_allow_html=True)

    episode_title = st.text_input(
        label="Episode title",
        label_visibility="collapsed",
        placeholder="e.g. The Future of Luxury in the UAE — ft. Sarah Al Mansouri"
    )

    st.markdown('<p class="sl" style="margin-top:1.5rem;">03 &nbsp; API Keys</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">'
        'OpenAI key for Whisper transcription &nbsp;&middot;&nbsp; '
        'Anthropic key for Claude content generation'
        '</div>',
        unsafe_allow_html=True
    )

    openai_key = st.text_input(
        "OpenAI Key", label_visibility="collapsed",
        type="password", placeholder="OpenAI API key (sk-...)  —  for Whisper transcription"
    )
    claude_key = st.text_input(
        "Anthropic Key", label_visibility="collapsed",
        type="password", placeholder="Anthropic API key (sk-ant-...)  —  console.anthropic.com"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("&#9672;  Run Pipeline", use_container_width=True)

    # Show past errors
    for err in st.session_state.errors[-2:]:
        st.markdown(f"""
        <div class="err-box">
            <div class="err-title">&#10007; &nbsp; Pipeline Error</div>
            <pre style="font-size:0.65rem;white-space:pre-wrap;margin:0">{html_lib.escape(err)}</pre>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.error_log:
        with st.expander("View pipeline execution log"):
            st.code("\n".join(st.session_state.error_log), language="text")

    st.markdown("""
    <div class="info-box" style="margin-top:1.5rem;">
        <strong style="color:#C9D8E8;">What this pipeline produces:</strong><br>
        Full transcript &middot; Structured show notes &middot; 3 key quotes &middot;
        LinkedIn post &middot; Team summary email
        <br><br>
        <strong style="color:#C9D8E8;">Error handling:</strong><br>
        File size &amp; type validation &middot; Whisper retry (3&times;) &middot;
        Claude retry (3&times;) &middot; Detailed fix hints on every failure
    </div>
    """, unsafe_allow_html=True)

# ── Pipeline trigger — runs OUTSIDE both columns ───────────────────────────────
if run_btn:
    st.session_state.errors          = []
    st.session_state.processing_done = False

    if not uploaded or not file_ok:
        st.error("Please upload a valid audio file first.")
    elif not episode_title.strip():
        st.error("Please enter an episode title.")
    elif not openai_key.strip():
        st.error("Enter your OpenAI API key for Whisper transcription.")
    elif not claude_key.strip():
        st.error("Enter your Anthropic API key from console.anthropic.com.")
    else:
        audio_bytes = uploaded.read()
        try:
            run_pipeline(audio_bytes, uploaded.name, episode_title, openai_key, claude_key)
        except RuntimeError:
            pass

# ── Right column — outputs ─────────────────────────────────────────────────────
with col_output:
    st.markdown('<p class="sl">Output</p>', unsafe_allow_html=True)

    if st.session_state.processing_done and st.session_state.transcript:
        meta = st.session_state.file_meta or {}
        wc   = meta.get("word_count", 0)
        dur  = meta.get("est_duration", "—")
        gen  = meta.get("generated_at", "—")
        fn   = meta.get("filename", "—")

        st.markdown(f"""
        <div class="stat-strip">
            <div class="stat-cell"><div class="stat-val">{wc:,}</div><div class="stat-lbl">Words transcribed</div></div>
            <div class="stat-cell"><div class="stat-val">{dur}</div><div class="stat-lbl">Est. duration</div></div>
            <div class="stat-cell"><div class="stat-val">5</div><div class="stat-lbl">Assets generated</div></div>
        </div>
        """, unsafe_allow_html=True)

        # Show notes
        if st.session_state.show_notes:
            safe = html_lib.escape(st.session_state.show_notes)
            st.markdown(f'<div class="out-card"><div class="out-card-head"><span class="out-card-title">&#128203; &nbsp; Show Notes</span></div><div class="out-card-body">{safe}</div></div>', unsafe_allow_html=True)
            with st.expander("Copy show notes"):
                st.code(st.session_state.show_notes, language="text")

        # Key quotes
        if st.session_state.quotes:
            quotes_html = ""
            for q in st.session_state.quotes:
                if isinstance(q, dict):
                    quote_text = html_lib.escape(q.get("quote", ""))
                    context    = html_lib.escape(q.get("context", ""))
                    platform   = html_lib.escape(q.get("platform", ""))
                    quotes_html += f'<div class="quote-card">&#8220;{quote_text}&#8221;<div class="quote-attr">&#8627; {context} &nbsp;&middot;&nbsp; Best for: {platform}</div></div>'
            st.markdown(
                f'<div class="out-card"><div class="out-card-head"><span class="out-card-title">&#128172; &nbsp; Key Quotes</span></div><div style="padding:1rem 1.25rem">{quotes_html}</div></div>',
                unsafe_allow_html=True
            )

        # LinkedIn post
        if st.session_state.linkedin_post:
            safe = html_lib.escape(st.session_state.linkedin_post)
            st.markdown(f'<div class="out-card"><div class="out-card-head"><span class="out-card-title">&#128188; &nbsp; LinkedIn Post</span></div><div class="out-card-body">{safe}</div></div>', unsafe_allow_html=True)
            with st.expander("Copy LinkedIn post"):
                st.code(st.session_state.linkedin_post, language="text")

        # Team email
        if st.session_state.summary_email:
            safe = html_lib.escape(st.session_state.summary_email)
            st.markdown(f'<div class="out-card"><div class="out-card-head"><span class="out-card-title">&#9993; &nbsp; Team Summary Email</span></div><div class="out-card-body">{safe}</div></div>', unsafe_allow_html=True)
            with st.expander("Copy email"):
                st.code(st.session_state.summary_email, language="text")

        # Full transcript
        with st.expander("View full transcript"):
            st.text_area(
                label="transcript", label_visibility="collapsed",
                value=st.session_state.transcript, height=300
            )

        st.markdown(f"""
        <div class="info-box" style="margin-top:1rem;">
            &#9672; &nbsp; Generated {html_lib.escape(gen)} &nbsp;&middot;&nbsp; Source: {html_lib.escape(fn)}
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="border:1px dashed var(--border);padding:4rem 2rem;text-align:center;background:var(--bg2)">
            <div style="font-family:var(--mono);font-size:2rem;color:var(--border2);margin-bottom:1rem">&#9672;</div>
            <div style="font-family:var(--mono);font-size:0.7rem;color:var(--border2);letter-spacing:0.15em;text-transform:uppercase">
                Awaiting episode upload
            </div>
            <div style="font-size:0.7rem;color:var(--text2);margin-top:0.75rem;line-height:1.7">
                Upload an audio file and run the pipeline<br>to generate all content assets automatically.
            </div>
        </div>
        <div style="margin-top:1.5rem">
            <div style="font-family:var(--mono);font-size:0.58rem;color:var(--text2);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.75rem">Pipeline steps</div>
            <div class="steps-row">
                <div class="step"><div class="step-dot"></div>Validate</div>
                <div class="step"><div class="step-dot"></div>Transcribe</div>
                <div class="step"><div class="step-dot"></div>Show notes</div>
                <div class="step"><div class="step-dot"></div>Quotes</div>
                <div class="step"><div class="step-dot"></div>LinkedIn</div>
                <div class="step"><div class="step-dot"></div>Email</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-top:1px solid var(--border);margin-top:3rem;padding-top:1rem;text-align:center">
    <span style="font-family:var(--mono);font-size:0.55rem;color:var(--border2);letter-spacing:0.2em;text-transform:uppercase">
        McFill Ignite &nbsp;&middot;&nbsp; Podcast-to-Content Pipeline &nbsp;&middot;&nbsp; Built for McFill Media Group
    </span>
</div>
""", unsafe_allow_html=True)