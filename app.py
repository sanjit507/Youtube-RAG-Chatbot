"""
YouTube RAG Chatbot – Streamlit UI
"""

import streamlit as st
from rag_engine import build_rag_chain, extract_video_id

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YT Chat – Ask Your Video",
    page_icon="🎬",
    layout="centered",          # ← centered keeps content from overflowing
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #141428 50%, #1a1040 100%);
        min-height: 100vh;
    }

    /* ── Hide chrome ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Main block padding ── */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 820px !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(15,12,41,0.97) !important;
        border-right: 1px solid rgba(167,139,250,0.12);
    }
    [data-testid="stSidebar"] .block-container { padding-top: 2rem; max-width: none !important; }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li {
        color: rgba(255,255,255,0.72) !important;
        font-size: 0.86rem !important;
    }

    /* ── Hero ── */
    .hero {
        text-align: center;
        padding: 1.5rem 1rem 2rem;
        animation: fadeInDown 0.6s ease;
    }
    .hero h1 {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa 0%, #60a5fa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.35rem;
        line-height: 1.15;
    }
    .hero p {
        color: rgba(255,255,255,0.48);
        font-size: 0.97rem;
        font-weight: 300;
        margin: 0;
    }

    /* ── Input fields ── */
    input, textarea,
    [data-testid="stTextInput"] input,
    [data-testid="stTextInput"] > div > div > input,
    .stTextInput input {
        background: #1e1a3a !important;
        background-color: #1e1a3a !important;
        border: 1.5px solid rgba(167,139,250,0.35) !important;
        border-radius: 10px !important;
        color: #e8e3ff !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        transition: border 0.2s ease, box-shadow 0.2s ease !important;
        -webkit-text-fill-color: #e8e3ff !important;
    }
    input:focus, textarea:focus,
    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextInput"] > div > div > input:focus,
    .stTextInput input:focus {
        background: #231f45 !important;
        background-color: #231f45 !important;
        border-color: #a78bfa !important;
        box-shadow: 0 0 0 3px rgba(167,139,250,0.18) !important;
        outline: none !important;
        color: #e8e3ff !important;
        -webkit-text-fill-color: #e8e3ff !important;
    }
    input::placeholder, textarea::placeholder,
    [data-testid="stTextInput"] input::placeholder {
        color: rgba(255,255,255,0.28) !important;
        -webkit-text-fill-color: rgba(255,255,255,0.28) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        height: 42px !important;
        width: 100% !important;
        letter-spacing: 0.2px !important;
        transition: all 0.22s ease !important;
        cursor: pointer !important;
        margin-top: 0 !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #8b5cf6, #6366f1) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 18px rgba(124,58,237,0.4) !important;
    }
    .stButton > button:active { transform: translateY(0) !important; }

    /* ── URL card ── */
    .url-section {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(167,139,250,0.2);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.5rem;
        animation: fadeIn 0.7s ease;
    }
    .url-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: rgba(167,139,250,0.8);
        letter-spacing: 0.6px;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }

    /* ── Badge ── */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .badge-ready { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
    .badge-info  { background: rgba(96,165,250,0.12);  color: #60a5fa;  border: 1px solid rgba(96,165,250,0.25); }

    /* ── Stats row ── */
    .stats-row {
        display: flex;
        gap: 0.9rem;
        margin: 1.2rem 0;
    }
    .stat-box {
        flex: 1;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        text-align: center;
    }
    .stat-val {
        font-size: 1.55rem;
        font-weight: 700;
        color: #a78bfa;
        line-height: 1.2;
    }
    .stat-lbl {
        font-size: 0.68rem;
        color: rgba(255,255,255,0.38);
        margin-top: 3px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* ── Divider ── */
    .divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.07);
        margin: 1.2rem 0;
    }

    /* ── Chat bubbles ── */
    .chat-area { animation: fadeIn 0.35s ease; }
    .chat-bubble {
        padding: 0.85rem 1.1rem;
        border-radius: 12px;
        margin-bottom: 0.7rem;
        line-height: 1.65;
        font-size: 0.93rem;
        word-wrap: break-word;
    }
    .bubble-user {
        background: rgba(167,139,250,0.12);
        border: 1px solid rgba(167,139,250,0.22);
        color: #e2d9fd;
        margin-left: 8%;
        border-bottom-right-radius: 4px;
    }
    .bubble-bot {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.09);
        color: rgba(255,255,255,0.88);
        margin-right: 8%;
        border-bottom-left-radius: 4px;
    }
    .bubble-label {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.7px;
        margin-bottom: 5px;
        opacity: 0.5;
        text-transform: uppercase;
    }

    /* ── Question area ── */
    .ask-section {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 1.1rem 1.3rem 0.8rem;
        margin-top: 0.8rem;
    }
    .hint-text {
        font-size: 0.74rem;
        color: rgba(255,255,255,0.3);
        margin-top: 0.4rem;
    }

    /* ── Empty state ── */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        opacity: 0.4;
        animation: fadeIn 0.6s ease;
    }
    .empty-state .icon { font-size: 3.5rem; margin-bottom: 1rem; }
    .empty-state p { color: rgba(255,255,255,0.65); font-size: 1rem; }

    /* ── Expander ── */
    details > summary {
        color: rgba(255,255,255,0.5) !important;
    }
    [data-testid="stExpander"] {
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 10px !important;
        background: rgba(255,255,255,0.02) !important;
    }

    /* ── Text area ── */
    [data-testid="stTextArea"] textarea {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.09) !important;
        border-radius: 8px !important;
        color: rgba(255,255,255,0.65) !important;
        font-size: 0.82rem !important;
        font-family: monospace !important;
    }

    /* ── Animations ── */
    @keyframes fadeIn    { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }
    @keyframes fadeInDown{ from{opacity:0;transform:translateY(-10px)} to{opacity:1;transform:translateY(0)} }
    @keyframes pulse     { 0%,100%{opacity:1} 50%{opacity:0.45} }
    .pulse { animation: pulse 1.5s infinite; }

    /* ── Vertical-align button with input ── */
    [data-testid="column"] > div { align-items: flex-end; display: flex; flex-direction: column; justify-content: flex-end; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state ────────────────────────────────────────────────────────────
defaults = {
    "chain": None,
    "transcript": "",
    "num_chunks": 0,
    "video_id": "",
    "history": [],
    "loaded_url": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    google_api_key = st.text_input(
        "Google API Key",
        type="password",
        placeholder="AIza...",
        help="Your Google Generative AI API key. Leave blank to use the .env file.",
    )
    if google_api_key:
        import os
        os.environ["GOOGLE_API_KEY"] = google_api_key

    st.markdown("---")
    st.markdown("### 📖 How it works")
    st.markdown(
        "1. **Paste** a YouTube URL  \n"
        "2. Click **Load Video** to index the transcript  \n"
        "3. **Ask** any question about the video  \n"
        "4. Powered by **Gemini + FAISS RAG**"
    )

    st.markdown("---")
    if st.session_state.chain:
        st.markdown('<span class="badge badge-ready">✦ Video Loaded</span>', unsafe_allow_html=True)
        st.markdown("")
        if st.button("🗑️ Clear & Reset", use_container_width=True):
            for k, v in defaults.items():
                st.session_state[k] = v
            st.rerun()
    else:
        st.markdown(
            '<span class="badge badge-info pulse">○ No video loaded</span>',
            unsafe_allow_html=True,
        )

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
      <h1>🎬 YT Chat</h1>
      <p>Chat with any YouTube video · Powered by LangChain &amp; Gemini</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── URL Input ─────────────────────────────────────────────────────────────────
st.markdown('<div class="url-section"><div class="url-label">YouTube URL</div>', unsafe_allow_html=True)
col_url, col_btn = st.columns([4, 1], gap="small")
with col_url:
    yt_url = st.text_input(
        "url",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed",
        key="yt_url_input",
    )
with col_btn:
    st.markdown('<div style="padding-top:0px">', unsafe_allow_html=True)
    load_clicked = st.button("🚀 Load", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── Load logic ────────────────────────────────────────────────────────────────
if load_clicked and yt_url:
    if yt_url == st.session_state.loaded_url:
        st.info("✅ This video is already loaded! Ask your questions below.")
    else:
        with st.spinner("⏳ Fetching transcript & building knowledge base…"):
            try:
                vid_id = extract_video_id(yt_url)
                chain, transcript, num_chunks = build_rag_chain(vid_id)

                st.session_state.chain       = chain
                st.session_state.transcript  = transcript
                st.session_state.num_chunks  = num_chunks
                st.session_state.video_id    = vid_id
                st.session_state.loaded_url  = yt_url
                st.session_state.history     = []

                st.success("✅ Video loaded! Start asking questions below.")
                st.rerun()
            except ValueError as e:
                st.error(f"❌ Invalid URL: {e}")
            except Exception as e:
                err = str(e)
                if "TranscriptsDisabled" in err or "NoTranscriptFound" in err:
                    st.error("❌ This video has no available transcript/subtitles.")
                else:
                    st.error(f"❌ Error: {err}")

elif load_clicked and not yt_url:
    st.warning("Please enter a YouTube URL first.")

# ── Loaded state ──────────────────────────────────────────────────────────────
if st.session_state.chain:
    word_count = len(st.session_state.transcript.split())
    q_count    = len(st.session_state.history) // 2

    # Stats
    st.markdown(
        f"""
        <div class="stats-row">
          <div class="stat-box">
            <div class="stat-val">{st.session_state.num_chunks}</div>
            <div class="stat-lbl">Chunks</div>
          </div>
          <div class="stat-box">
            <div class="stat-val">{word_count:,}</div>
            <div class="stat-lbl">Words</div>
          </div>
          <div class="stat-box">
            <div class="stat-val">{q_count}</div>
            <div class="stat-lbl">Asked</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Chat history
    if st.session_state.history:
        st.markdown('<div class="chat-area">', unsafe_allow_html=True)
        for role, msg in st.session_state.history:
            if role == "user":
                st.markdown(
                    f'<div class="chat-bubble bubble-user">'
                    f'<div class="bubble-label">You</div>{msg}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-bubble bubble-bot">'
                    f'<div class="bubble-label">AI</div>{msg}</div>',
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

    # Question input
    st.markdown('<div class="ask-section">', unsafe_allow_html=True)
    q_col, b_col = st.columns([5, 1], gap="small")
    with q_col:
        question = st.text_input(
            "question",
            placeholder="What is the main topic of this video?",
            label_visibility="collapsed",
            key="question_input",
        )
    with b_col:
        ask_clicked = st.button("💬 Ask", use_container_width=True)
    st.markdown(
        '<p class="hint-text">💡 Try: <em>Summarize this video</em> &nbsp;|&nbsp; '
        '<em>What are the key points?</em> &nbsp;|&nbsp; <em>Explain the main concept</em></p>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if ask_clicked and question:
        with st.spinner("🤔 Thinking…"):
            try:
                answer = st.session_state.chain.invoke(question)
                st.session_state.history.append(("user", question))
                st.session_state.history.append(("bot", answer))
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")
    elif ask_clicked and not question:
        st.warning("Please type a question first.")

    # Transcript viewer
    st.markdown("")
    with st.expander("📄 View Raw Transcript"):
        st.text_area(
            "",
            st.session_state.transcript,
            height=220,
            label_visibility="collapsed",
        )

else:
    # Empty state
    st.markdown(
        """
        <div class="empty-state">
          <div class="icon">📺</div>
          <p>Paste a YouTube URL above and click <strong>Load</strong> to get started.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
