"""
app/streamlit_app.py — Checkit Analytics Earnings Intelligence UI.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import time
from pathlib import Path

# Ensure project root on sys.path regardless of launch directory
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="Checkit Analytics",
    page_icon="📊",
    layout="wide",
)

# ── Brand CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>

    /* === GLOBAL === */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-size: 15px;
    }

    .stApp { background-color: #ffffff; }

    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1100px !important;
    }

    hr { border-color: #e9ecef !important; }

    [data-testid="stSpinner"] p { color: #003768 !important; }

    [data-testid="stAlert"] { border-radius: 8px !important; }


    /* === SIDEBAR === */
    [data-testid="stSidebar"] {
        background-color: #003768 !important;
        border-right: none !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        background-color: #003768 !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown {
        color: rgba(255, 255, 255, 0.90) !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.18) !important;
        margin: 0.75rem 0 !important;
    }

    [data-testid="stSidebar"] .stSelectbox label {
        color: rgba(255, 255, 255, 0.70) !important;
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.10) !important;
        border-color: rgba(255, 255, 255, 0.25) !important;
        border-radius: 6px !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] span {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] svg {
        fill: rgba(255, 255, 255, 0.70) !important;
    }

    [data-testid="stSidebar"] details summary {
        color: rgba(255, 255, 255, 0.90) !important;
        background-color: rgba(255, 255, 255, 0.07) !important;
        border-radius: 6px !important;
        padding: 0.5rem 0.75rem !important;
    }
    [data-testid="stSidebar"] details summary:hover {
        background-color: rgba(255, 255, 255, 0.12) !important;
    }
    [data-testid="stSidebar"] details summary svg {
        fill: rgba(255, 255, 255, 0.70) !important;
    }
    [data-testid="stSidebar"] details[open] summary {
        border-radius: 6px 6px 0 0 !important;
    }

    [data-testid="stSidebar"] code {
        background-color: rgba(255, 255, 255, 0.12) !important;
        color: #a8d4f5 !important;
        border: none !important;
        padding: 1px 5px !important;
        border-radius: 4px !important;
        font-size: 0.78rem !important;
    }


    /* === HEADER === */
    .stMarkdown h3 {
        color: #003768 !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.6rem !important;
    }


    /* === BUTTONS === */
    [data-testid="stBaseButton-primary"],
    .stButton > button[kind="primary"] {
        background-color: #003768 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.55rem 2.2rem !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.03em !important;
        min-width: 180px !important;
        transition: background-color 0.18s ease, box-shadow 0.18s ease !important;
        box-shadow: 0 2px 6px rgba(0, 55, 104, 0.30) !important;
    }
    [data-testid="stBaseButton-primary"]:hover,
    .stButton > button[kind="primary"]:hover {
        background-color: #004f99 !important;
        box-shadow: 0 4px 12px rgba(0, 55, 104, 0.40) !important;
    }

    [data-testid="stBaseButton-secondary"],
    .stButton > button:not([kind="primary"]) {
        background-color: #ffffff !important;
        color: #003768 !important;
        border: 1.5px solid #003768 !important;
        border-radius: 6px !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        transition: all 0.18s ease !important;
    }
    [data-testid="stBaseButton-secondary"]:hover,
    .stButton > button:not([kind="primary"]):hover {
        background-color: #003768 !important;
        color: #ffffff !important;
    }

    .stTextArea textarea {
        border: 1.5px solid #d1d5db !important;
        border-radius: 8px !important;
        font-size: 0.95rem !important;
        padding: 0.75rem 1rem !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    }
    .stTextArea textarea:focus {
        border-color: #0066CC !important;
        box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.12) !important;
        outline: none !important;
    }
    .stTextArea label {
        font-weight: 600 !important;
        color: #003768 !important;
        font-size: 0.88rem !important;
    }


    /* === RESULTS === */
    [data-testid="stExpander"] summary {
        font-weight: 500 !important;
        color: #374151 !important;
        font-size: 0.88rem !important;
    }


    /* === BADGES === */
    /* Sentiment, confidence, consistency, and risk-flag badges are rendered
       inline via Python helpers (_badge / _risk_flag_pills). No class rules
       needed here — all styling is applied directly in the HTML strings. */


    /* === FOOTER === */
    /* Footer is rendered inline via Python (st.markdown). No class rules needed. */

    </style>
    """,
    unsafe_allow_html=True,
)

# ── Company → ticker mapping ───────────────────────────────────────────────────
COMPANIES: dict[str, str] = {
    "NVIDIA Corporation":          "NVDA",
    "Apple Inc.":                  "AAPL",
    "Microsoft Corporation":       "MSFT",
    "Tesla Inc.":                  "TSLA",
    "Meta Platforms Inc.":         "META",
    "Palantir Technologies Inc.":  "PLTR",
    "Snowflake Inc.":              "SNOW",
    "Shopify Inc.":                "SHOP",
    "Datadog Inc.":                "DDOG",
    "CrowdStrike Holdings Inc.":   "CRWD",
    "Uber Technologies Inc.":      "UBER",
    "Airbnb Inc.":                 "ABNB",
    "ServiceNow Inc.":             "NOW",
    "Workday Inc.":                "WDAY",
    "Eli Lilly and Company":       "LLY",
    "Booking Holdings Inc.":       "BKNG",
    "Intuitive Surgical Inc.":     "ISRG",
    "Veeva Systems Inc.":          "VEEV",
    "Cloudflare Inc.":             "NET",
    "MongoDB Inc.":                "MDB",
    "JPMorgan Chase & Co.":              "JPM",
    "Goldman Sachs Group Inc.":          "GS",
    "Morgan Stanley":                    "MS",
    "Bank of America Corp.":             "BAC",
    "Wells Fargo & Company":             "WFC",
    "Salesforce Inc.":                   "CRM",
    "Zoom Video Communications Inc.":    "ZM",
    "HubSpot Inc.":                      "HUBS",
    "Amazon.com Inc.":                   "AMZN",
    "Nike Inc.":                         "NKE",
    "Starbucks Corporation":             "SBUX",
    "UnitedHealth Group Inc.":           "UNH",
    "Pfizer Inc.":                       "PFE",
    "Johnson & Johnson":                 "JNJ",
    "NextEra Energy Inc.":               "NEE",
    "Enphase Energy Inc.":               "ENPH",
    "First Solar Inc.":                  "FSLR",
    "Rivian Automotive Inc.":            "RIVN",
    "Lucid Group Inc.":                  "LCID",
    "Plug Power Inc.":                   "PLUG",
    "Bloom Energy Corp.":                "BE",
    "Sunrun Inc.":                       "RUN",
    "ChargePoint Holdings Inc.":         "CHPT",
    "Aptiv PLC":                         "APTV",
}

SECTION_OPTIONS: dict[str, str | None] = {
    "All Sections":      None,
    "Prepared Remarks":  "prepared_remarks",
    "Q&A":               "qa",
}

EXAMPLE_QUERIES = [
    "What did NVIDIA say about AI chip demand and data center growth?",
    "How is Microsoft's Azure cloud revenue trending compared to AWS?",
    "What risks did management highlight for the next quarter?",
    "Which companies mentioned macro headwinds or recession concerns?",
]

# ── Cached resource loaders ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model…")
def _load_searcher():
    """Import and warm-up the searcher singletons once per process."""
    from retrieval.searcher import _model, _collection  # noqa: PLC0415
    _model()
    _collection()
    from retrieval.searcher import search, format_context  # noqa: PLC0415
    return search, format_context


@st.cache_resource(show_spinner=False)
def _load_answer_fn():
    from reasoning.answerer import answer  # noqa: PLC0415
    return answer


# ── Badge helpers ─────────────────────────────────────────────────────────────

_SENTIMENT_COLORS: dict[str, tuple[str, str]] = {
    "positive": ("#d1fae5", "#065f46"),
    "negative": ("#fee2e2", "#991b1b"),
    "mixed":    ("#fef3c7", "#92400e"),
    "neutral":  ("#f1f5f9", "#475569"),
}
_CONFIDENCE_COLORS: dict[str, tuple[str, str]] = {
    "high":   ("#dbeafe", "#1e40af"),
    "medium": ("#fef3c7", "#92400e"),
    "low":    ("#fee2e2", "#991b1b"),
}
_CONSISTENCY_COLORS: dict[str, tuple[str, str]] = {
    "aligned":  ("#d1fae5", "#065f46"),  # green — all evidence agrees
    "mixed":    ("#fef3c7", "#92400e"),  # yellow — some disagreement
    "conflict": ("#fee2e2", "#991b1b"),  # red   — direct contradiction
}


def _badge(label: str, value: str, color_map: dict) -> str:
    bg, fg = color_map.get(value.lower(), ("#f1f5f9", "#475569"))
    return (
        f'<span style="'
        f'background:{bg};color:{fg};'
        f'padding:4px 14px;'
        f'border-radius:100px;'
        f'font-size:0.76rem;font-weight:700;'
        f'letter-spacing:0.05em;'
        f'text-transform:uppercase;'
        f'display:inline-block;margin-right:8px;'
        f'">'
        f'{label}: {value}'
        f'</span>'
    )


def _risk_flag_pills(flags: list[str]) -> str:
    """Render a list of risk flags as red-tinted pill badges."""
    pills = "".join(
        f'<span style="'
        f'background:#fee2e2;color:#991b1b;'
        f'padding:4px 12px;'
        f'border-radius:100px;'
        f'font-size:0.76rem;font-weight:600;'
        f'display:inline-block;margin:3px 6px 3px 0;'
        f'">{flag}</span>'
        for flag in flags
    )
    return f'<div style="margin:0.5rem 0 1rem 0;">{pills}</div>'


# ── HTML component helpers ────────────────────────────────────────────────────

def _answer_card(text: str) -> str:
    """White card with blue left border for the main answer."""
    safe = text.replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'<div style="'
        f'background:#ffffff;'
        f'border-left:4px solid #0066CC;'
        f'border-radius:0 8px 8px 0;'
        f'padding:1.1rem 1.4rem;'
        f'margin:0.5rem 0 1.2rem 0;'
        f'box-shadow:0 1px 4px rgba(0,0,0,0.07);'
        f'font-size:0.97rem;line-height:1.65;color:#1e293b;'
        f'">'
        f'{safe}'
        f'</div>'
    )


def _evidence_card(company: str, ticker: str, speaker: str, role: str,
                   quarter: str, quote: str, why: str) -> str:
    """White card with thin blue top border for each evidence item."""
    ticker_tag = (
        f'<span style="font-size:0.78rem;font-weight:500;color:#64748b;'
        f'background:#f1f5f9;border-radius:4px;padding:1px 6px;'
        f'margin-left:6px;">{ticker}</span>'
        if ticker else ""
    )
    meta_parts = []
    if speaker and role:
        meta_parts.append(f"{speaker}, <em>{role}</em>")
    elif speaker:
        meta_parts.append(speaker)
    if quarter:
        meta_parts.append(quarter)
    meta_html = (
        f'<div style="color:#64748b;font-size:0.82rem;margin-top:3px;">'
        f'{"&nbsp;·&nbsp;".join(meta_parts)}</div>'
        if meta_parts else ""
    )
    quote_html = (
        f'<blockquote style="'
        f'border-left:3px solid #0066CC;'
        f'margin:0.7rem 0 0.5rem 0;'
        f'padding:0.35rem 0.9rem;'
        f'color:#334155;font-style:italic;font-size:0.88rem;line-height:1.55;'
        f'">&ldquo;{quote}&rdquo;</blockquote>'
        if quote else ""
    )
    why_html = (
        f'<div style="color:#94a3b8;font-size:0.78rem;margin-top:4px;">{why}</div>'
        if why else ""
    )
    return (
        f'<div style="'
        f'background:#ffffff;'
        f'border:1px solid #e2e8f0;'
        f'border-top:3px solid #0066CC;'
        f'border-radius:8px;'
        f'padding:1rem 1.2rem;'
        f'margin-bottom:0.75rem;'
        f'box-shadow:0 1px 3px rgba(0,0,0,0.05);'
        f'">'
        f'<div style="font-weight:700;color:#003768;font-size:0.92rem;">'
        f'{company}{ticker_tag}'
        f'</div>'
        f'{meta_html}'
        f'{quote_html}'
        f'{why_html}'
        f'</div>'
    )


def _stats_bar(retrieval_ms: float, generation_ms: float,
               total_ms: float, n_chunks: int) -> str:
    return (
        f'<div style="'
        f'display:flex;gap:1.2rem;flex-wrap:wrap;'
        f'background:#f8fafc;border-radius:8px;'
        f'padding:0.55rem 1rem;margin-bottom:1.25rem;'
        f'font-size:0.78rem;color:#64748b;'
        f'border:1px solid #e2e8f0;'
        f'">'
        f'<span>⏱ Retrieval <strong>{retrieval_ms:.0f} ms</strong></span>'
        f'<span style="color:#cbd5e1;">|</span>'
        f'<span>Generation <strong>{generation_ms:.0f} ms</strong></span>'
        f'<span style="color:#cbd5e1;">|</span>'
        f'<span>Total <strong>{total_ms:.0f} ms</strong></span>'
        f'<span style="color:#cbd5e1;">|</span>'
        f'<span><strong>{n_chunks}</strong> source chunks</span>'
        f'</div>'
    )


# ── App layout ────────────────────────────────────────────────────────────────

# Header
st.markdown(
    '<h1 style="'
    'color:#003768;'
    'font-size:1.85rem;'
    'font-weight:700;'
    'margin-bottom:0.15rem;'
    'letter-spacing:-0.01em;'
    '">📊 Checkit Analytics — Earnings Intelligence</h1>',
    unsafe_allow_html=True,
)
# Thin brand-color rule under the header
st.markdown(
    '<hr style="border:none;border-top:2px solid #0066CC;margin:0 0 1.5rem 0;">',
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<h2 style="color:#ffffff;font-size:1.1rem;font-weight:700;'
        'margin-bottom:0.1rem;">📊 Checkit Analytics</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:rgba(255,255,255,0.65);font-size:0.80rem;'
        'margin-top:0;font-style:italic;">'
        'Earnings intelligence, grounded in primary sources.</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    company_choice = st.selectbox(
        "Filter by company",
        options=["All Companies"] + list(COMPANIES.keys()),
        index=0,
    )
    section_choice = st.selectbox(
        "Filter by section",
        options=list(SECTION_OPTIONS.keys()),
        index=0,
    )

    st.divider()

    with st.expander("How this works", expanded=False):
        st.markdown(
            """
1. Your question is embedded with **all-MiniLM-L6-v2**.
2. The top-8 most similar chunks are retrieved from **ChromaDB** (cosine similarity).
3. Those chunks are sent to **Groq (llama-3.1-8b-instant)** with a strict analyst prompt.
4. The model returns a structured JSON answer with evidence citations.
            """
        )

    with st.expander("Data coverage", expanded=False):
        st.markdown("**20 companies · 8 quarters (2023–2025)**")
        for name, ticker in COMPANIES.items():
            st.markdown(f"- {name} `{ticker}`")

# ── Session state for query prefill ───────────────────────────────────────────
if "query_text" not in st.session_state:
    st.session_state["query_text"] = ""

# ── Main query area ────────────────────────────────────────────────────────────
query_input = st.text_area(
    "Ask a question about earnings calls",
    value=st.session_state["query_text"],
    height=90,
    placeholder="e.g. What did NVIDIA say about AI chip demand?",
    key="query_area",
)

# Example query buttons
st.markdown(
    '<p style="font-size:0.82rem;font-weight:600;color:#64748b;'
    'text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.4rem;">'
    'Examples</p>',
    unsafe_allow_html=True,
)
cols = st.columns(len(EXAMPLE_QUERIES))
for col, example in zip(cols, EXAMPLE_QUERIES):
    short = example[:45] + "…" if len(example) > 45 else example
    if col.button(short, use_container_width=True):
        st.session_state["query_text"] = example
        st.rerun()

st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
run_btn = st.button("🔍 Ask Checkit", type="primary", use_container_width=False)

# ── Query execution ────────────────────────────────────────────────────────────
if run_btn:
    query = query_input.strip()
    if not query:
        st.warning("Please enter a question before clicking Ask Checkit.")
        st.stop()

    ticker_filter  = COMPANIES.get(company_choice)
    section_filter = SECTION_OPTIONS.get(section_choice)

    try:
        search_fn, format_ctx = _load_searcher()
        answer_fn = _load_answer_fn()
    except Exception as exc:
        st.error(f"Failed to load pipeline components: {exc}")
        st.stop()

    # Dynamic spinner message based on query type
    query_lower = query.lower()
    is_comparison = any(kw in query_lower for kw in ("compare", " vs ", "versus"))
    retrieval_msg = (
        "Retrieving data from multiple companies..."
        if is_comparison
        else "Searching transcripts..."
    )

    # Retrieve
    with st.spinner(retrieval_msg):
        t0 = time.perf_counter()
        chunks = search_fn(query, ticker=ticker_filter, section=section_filter, n_results=8)
        retrieval_ms = (time.perf_counter() - t0) * 1000

    if not chunks:
        st.warning(
            "No relevant chunks found for this query with the current filters. "
            "Try broadening the company or section filter."
        )
        st.stop()

    # Generate
    with st.spinner("Generating structured analyst answer via Groq…"):
        t1 = time.perf_counter()
        result = answer_fn(query, chunks)
        generation_ms = (time.perf_counter() - t1) * 1000

    total_ms = retrieval_ms + generation_ms

    # ── Results ────────────────────────────────────────────────────────────────
    st.markdown(
        '<hr style="border:none;border-top:1px solid #e2e8f0;margin:1rem 0 1rem 0;">',
        unsafe_allow_html=True,
    )
    st.markdown(_stats_bar(retrieval_ms, generation_ms, total_ms, len(chunks)),
                unsafe_allow_html=True)

    # Answer
    st.markdown("### Answer")
    st.markdown(_answer_card(result.get("answer", "No answer generated.")),
                unsafe_allow_html=True)

    # Key points
    key_points = result.get("key_points", [])
    if key_points:
        st.markdown("**Key Points**")
        for pt in key_points:
            st.markdown(f"- {pt}")

    # Risk flags — only shown when non-empty
    risk_flags = result.get("risk_flags", [])
    if risk_flags:
        st.markdown("**Risk Flags**")
        st.markdown(_risk_flag_pills(risk_flags), unsafe_allow_html=True)

    # Sentiment + confidence + consistency badges
    sentiment    = result.get("sentiment",    "neutral").lower()
    confidence   = result.get("confidence",   "low").lower()
    consistency  = result.get("consistency",  "aligned").lower()
    badge_html = (
        _badge("Sentiment",          sentiment,   _SENTIMENT_COLORS)
        + _badge("Confidence",       confidence,  _CONFIDENCE_COLORS)
        + _badge("Source Consistency", consistency, _CONSISTENCY_COLORS)
    )
    st.markdown(
        f'<div style="margin:0.8rem 0 1rem 0;">{badge_html}</div>',
        unsafe_allow_html=True,
    )

    # Evidence cards
    evidence = result.get("evidence", [])
    if evidence:
        st.markdown("### Evidence")
        for ev in evidence:
            st.markdown(
                _evidence_card(
                    company = ev.get("company", ""),
                    ticker  = ev.get("ticker",  ""),
                    speaker = ev.get("speaker", ""),
                    role    = ev.get("role",    ""),
                    quarter = ev.get("quarter", ""),
                    quote   = ev.get("quote",   ""),
                    why     = ev.get("why_relevant", ""),
                ),
                unsafe_allow_html=True,
            )

    # Limitations
    limitations = result.get("limitations", "")
    if limitations:
        st.markdown("### Limitations")
        st.markdown(
            f'<p style="color:#94a3b8;font-size:0.84rem;line-height:1.6;">'
            f'{limitations}</p>',
            unsafe_allow_html=True,
        )

    # Raw JSON toggle
    with st.expander("Raw JSON response", expanded=False):
        st.json(result)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<hr style="border:none;border-top:1px solid #e2e8f0;margin:2rem 0 1rem 0;">',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="'
    'text-align:center;'
    'color:#94a3b8;'
    'font-size:0.78rem;'
    'letter-spacing:0.02em;'
    'margin:0;'
    '">'
    'Checkit Analytics'
    '&nbsp;&nbsp;·&nbsp;&nbsp;'
    'Data sourced from SEC EDGAR official filings'
    '&nbsp;&nbsp;·&nbsp;&nbsp;'
    'Not financial advice'
    '</p>',
    unsafe_allow_html=True,
)
