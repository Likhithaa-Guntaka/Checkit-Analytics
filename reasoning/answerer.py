"""
reasoning/answerer.py — LLM reasoning layer: Groq-powered structured answers.

Architecture
------------
This module is the final stage of the RAG pipeline. It takes retrieved chunks
from retrieval/searcher.py, formats them into a context string, and sends them
to the Groq API alongside the analyst query. The model is prompted to produce
a deterministic, schema-constrained JSON response — never free-form prose.

Output schema
-------------
Every call returns a dict with these fields:
  answer       : 2-3 sentence direct answer grounded in the evidence
  key_points   : 3–5 bullet points drawn from transcript text
  sentiment    : "positive" | "negative" | "neutral" | "mixed"
  confidence   : "high" | "medium" | "low"
  evidence     : list of attributed quote objects (company/speaker/quote/…)
  limitations  : what the answer cannot tell you
  risk_flags   : specific risks extracted verbatim from the transcript evidence
  consistency  : "aligned" | "mixed" | "conflict" — agreement across evidence pieces

Retry logic
-----------
If the model returns malformed JSON, a second call is made with a stricter
"return only JSON" reminder. A static fallback dict is returned if both fail.

Comparison routing
------------------
full_pipeline() detects cross-company queries ("vs", "compare", "and", etc.)
and routes them through search_multi_company() so no single company dominates.

Public API
----------
build_prompt(query, context_string) → (system_prompt, user_prompt)
answer(query, chunks)               → dict
full_pipeline(query, ticker, section) → dict
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

# ── Load env ───────────────────────────────────────────────────────────────────

# Walk up from this file to find .env (works whether run as a module or script)
_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE, _HERE.parent, _HERE.parent.parent):
    if (_candidate / ".env").exists():
        load_dotenv(_candidate / ".env")
        break

# ── Configuration ──────────────────────────────────────────────────────────────

# llama-3.3-70b-versatile: chosen for sub-second latency and strong instruction-following
# on structured JSON tasks. The 8B model is sufficient for schema-constrained
# extraction from earnings transcripts; larger models add latency without quality gain.
GROQ_MODEL   = "llama-3.3-70b-versatile"
TEMPERATURE  = 0.1    # low temperature for factual, structured output
MAX_TOKENS   = 800

# ── Prompt builder ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert equity research analyst. \
You only answer based on the earnings call transcript evidence provided. \
You never make up information. \
You always cite your sources. \
When identifying risk_flags, extract only risks that are explicitly mentioned \
in the transcript evidence — never infer or fabricate risks not present in the text. \
When assessing consistency, compare evidence pieces: \
"aligned" means all evidence agrees, \
"mixed" means some disagreement or uncertainty exists, \
"conflict" means evidence pieces directly contradict each other. \
You respond ONLY in valid JSON with no additional text before or after.\
"""

_JSON_SCHEMA = """\
{
  "answer": "clear direct answer in 2-3 sentences",
  "key_points": ["3 to 5 bullet points drawn directly from the transcripts"],
  "sentiment": "positive or negative or neutral or mixed",
  "confidence": "high or medium or low",
  "evidence": [
    {
      "company": "full company name",
      "ticker": "TICKER",
      "speaker": "speaker name",
      "role": "role e.g. CEO",
      "quarter": "e.g. Q3 2024",
      "quote": "short direct quote under 30 words",
      "why_relevant": "one sentence explanation"
    }
  ],
  "limitations": "what this answer cannot tell you or what is uncertain",
  "risk_flags": ["specific risk or headwind mentioned verbatim in the evidence — omit if none"],
  "consistency": "aligned or mixed or conflict"
}\
"""

_USER_TEMPLATE = """\
### Context (earnings call excerpts)
{context}

### Question
{question}

### Instructions
Answer the question using ONLY the context above.
- If evidence is weak, set confidence to "low" and explain in the answer.
- Never fabricate quotes — use only text that appears verbatim in the context.
- Return ONLY the following JSON object, with no markdown fences, no preamble:

{schema}\
"""

_RETRY_SUFFIX = (
    "\n\nYour previous response was not valid JSON. "
    "Return ONLY the JSON object below — no markdown, no explanation, "
    "no ```json fences. Start your response with { and end with }."
    f"\n\n{_JSON_SCHEMA}"
)

_FALLBACK_RESPONSE = {
    "answer": "Unable to generate a structured answer due to a parsing error.",
    "key_points": [],
    "sentiment": "neutral",
    "confidence": "low",
    "evidence": [],
    "limitations": "The model returned a response that could not be parsed as JSON.",
    "risk_flags": [],
    "consistency": "aligned",
}


def build_prompt(query: str, context_string: str) -> tuple[str, str]:
    """
    Build (system_prompt, user_prompt) for the Groq chat completion.

    Parameters
    ----------
    query          : the analyst's question
    context_string : pre-formatted context from searcher.format_context()

    Returns
    -------
    (system_prompt, user_prompt) — both plain strings
    """
    user_prompt = _USER_TEMPLATE.format(
        context=context_string,
        question=query,
        schema=_JSON_SCHEMA,
    )
    return _SYSTEM_PROMPT, user_prompt


# ── JSON extraction ────────────────────────────────────────────────────────────

def _extract_json(text: str) -> Optional[dict]:
    """
    Try to extract a JSON object from a model response.
    Handles cases where the model wraps the JSON in markdown fences.
    Returns the parsed dict, or None if no valid JSON is found.
    """
    # 1. Try parsing the whole response directly
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 2. Strip ```json ... ``` fences and retry
    stripped = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    stripped = stripped.rstrip("`").strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 3. Find the first { ... } block by brace counting
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    break

    return None


# ── Groq caller ───────────────────────────────────────────────────────────────

def _call_groq(messages: list[dict]) -> str:
    """Send messages to Groq and return the content string."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()


# ── Main answer function ───────────────────────────────────────────────────────

def answer(query: str, chunks: list[dict]) -> dict:
    """
    Generate a structured analyst answer for *query* grounded in *chunks*.

    Attempts JSON parsing twice:
      1. Normal call with full prompt.
      2. If parsing fails, retry with a stricter "JSON only" reminder appended.
    Returns a fallback dict on double failure.

    Parameters
    ----------
    query  : the natural-language question
    chunks : list of chunk dicts as returned by searcher.search()

    Returns
    -------
    Parsed JSON dict matching the schema defined in _JSON_SCHEMA.
    """
    # Import here to avoid circular imports when this module is used standalone
    from retrieval.searcher import format_context

    context_string = format_context(chunks)
    system_prompt, user_prompt = build_prompt(query, context_string)

    messages = [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": user_prompt},
    ]

    # ── Attempt 1 ─────────────────────────────────────────────────────────────
    try:
        raw = _call_groq(messages)
        parsed = _extract_json(raw)
        if parsed is not None:
            return parsed
        # Parsing failed → fall through to retry
        print("[answerer] Attempt 1: JSON parse failed, retrying …")
    except Exception as exc:
        print(f"[answerer] Attempt 1 API error: {exc}")
        return {**_FALLBACK_RESPONSE, "limitations": str(exc)}

    # ── Attempt 2: stricter prompt ────────────────────────────────────────────
    retry_messages = messages + [
        {"role": "assistant", "content": raw},
        {"role": "user",      "content": _RETRY_SUFFIX},
    ]
    try:
        raw2   = _call_groq(retry_messages)
        parsed = _extract_json(raw2)
        if parsed is not None:
            return parsed
        print("[answerer] Attempt 2: JSON parse still failed, using fallback.")
    except Exception as exc:
        print(f"[answerer] Attempt 2 API error: {exc}")

    return {
        **_FALLBACK_RESPONSE,
        "limitations": (
            "Two Groq API calls both returned non-JSON responses. "
            "Raw response snippet: " + (raw or "")[:200]
        ),
    }


# ── Comparison query detection ─────────────────────────────────────────────────

# Trigger words that suggest a cross-company comparison query
_COMPARISON_TRIGGERS: frozenset[str] = frozenset(
    {"and", "vs", "versus", "compare", "compared", "comparison", "between"}
)

# Map of query keywords → ticker symbol.
# Checked as whole words (word-boundary split) to avoid false positives
# on common English words like "net" or "now".
_TICKER_KEYWORDS: dict[str, str] = {
    # NVDA
    "nvidia": "NVDA", "nvda": "NVDA",
    # AAPL
    "apple": "AAPL", "aapl": "AAPL",
    # MSFT
    "microsoft": "MSFT", "msft": "MSFT", "azure": "MSFT",
    # TSLA
    "tesla": "TSLA", "tsla": "TSLA",
    # META
    "meta": "META", "facebook": "META",
    # PLTR
    "palantir": "PLTR", "pltr": "PLTR",
    # SNOW
    "snowflake": "SNOW", "snow": "SNOW",
    # SHOP
    "shopify": "SHOP", "shop": "SHOP",
    # DDOG
    "datadog": "DDOG", "ddog": "DDOG",
    # CRWD
    "crowdstrike": "CRWD", "crwd": "CRWD",
    # UBER
    "uber": "UBER",
    # ABNB
    "airbnb": "ABNB", "abnb": "ABNB",
    # NOW  — "servicenow" only; skip bare "now" (too ambiguous)
    "servicenow": "NOW",
    # WDAY
    "workday": "WDAY", "wday": "WDAY",
    # LLY
    "lilly": "LLY", "lly": "LLY",
    # BKNG
    "booking": "BKNG", "bkng": "BKNG",
    # ISRG
    "intuitive": "ISRG", "isrg": "ISRG",
    # VEEV
    "veeva": "VEEV", "veev": "VEEV",
    # NET — "cloudflare" only; skip bare "net" (too ambiguous)
    "cloudflare": "NET",
    # MDB
    "mongodb": "MDB", "mdb": "MDB", "mongo": "MDB",
}


def _is_comparison_query(query: str) -> bool:
    """Return True if the query looks like a cross-company comparison."""
    words = set(re.split(r"\W+", query.lower()))
    return bool(words & _COMPARISON_TRIGGERS)


def _extract_tickers(query: str) -> list[str]:
    """
    Extract ticker symbols from *query* by matching against _TICKER_KEYWORDS.

    Splits the query into tokens (words + adjacent two-word pairs) and checks
    each against the keyword map. Returns a deduplicated list in mention order.
    """
    q_lower = query.lower()
    # Build token list: single words + adjacent bigrams
    words  = re.split(r"\W+", q_lower)
    tokens = words + [f"{a}{b}" for a, b in zip(words, words[1:])]  # e.g. "crowdstrike"

    seen:    dict[str, int] = {}   # ticker → first-seen position
    for pos, token in enumerate(tokens):
        ticker = _TICKER_KEYWORDS.get(token)
        if ticker and ticker not in seen:
            seen[ticker] = pos

    return [t for t, _ in sorted(seen.items(), key=lambda x: x[1])]


# ── Full pipeline ──────────────────────────────────────────────────────────────

def full_pipeline(
    query:   str,
    ticker:  Optional[str] = None,
    section: Optional[str] = None,
) -> dict:
    """
    End-to-end RAG pipeline: retrieve → format → answer.

    For single-company queries (or when *ticker* is explicitly supplied) this
    calls search() with an optional ticker filter.

    For cross-company comparison queries (detected by trigger words such as
    "vs", "compare", "and") it calls search_multi_company() to ensure each
    relevant company contributes chunks to the context.

    Parameters
    ----------
    query   : natural-language analyst question
    ticker  : optional ticker filter e.g. "NVDA" (skips comparison detection)
    section : optional section filter "prepared_remarks" | "qa" | "earnings_release"

    Returns
    -------
    Structured answer dict (see _JSON_SCHEMA).
    """
    from retrieval.searcher import search, search_multi_company, format_context  # lazy import

    # ── Routing logic ──────────────────────────────────────────────────────────
    if ticker is None and _is_comparison_query(query):
        tickers = _extract_tickers(query)
        if len(tickers) >= 2:
            print(
                f"[answerer] Comparison query detected — "
                f"fetching per-company chunks for: {tickers}"
            )
            chunks = search_multi_company(query, tickers=tickers, n_per_company=3,
                                          section=section)
        else:
            # Trigger words present but fewer than 2 companies identified —
            # fall back to a broad unfiltered search
            chunks = search(query, section=section, n_results=4)
    else:
        chunks = search(query, ticker=ticker, section=section, n_results=4)

    if not chunks:
        return {
            **_FALLBACK_RESPONSE,
            "answer": "No relevant context was found in the database for this query.",
            "limitations": "The ChromaDB collection may be empty or the query had no matches.",
        }

    return answer(query, chunks)


# ── Entry-point test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, warnings
    warnings.filterwarnings("ignore")
    # Ensure project root is on sys.path when run as a script
    _root = str(Path(__file__).resolve().parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)

    test_query = (
        "What did NVIDIA say about AI chip demand in their most recent earnings call?"
    )
    print(f'Running full pipeline for:\n  "{test_query}"\n')

    result = full_pipeline(test_query, ticker="NVDA")
    print(json.dumps(result, indent=2))
