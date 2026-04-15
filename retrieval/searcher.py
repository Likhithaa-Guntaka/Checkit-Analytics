"""
retrieval/searcher.py — Semantic retrieval from ChromaDB for Checkit Analytics.

Architecture
------------
Two module-level singletons are initialised lazily on first use and then reused
across every call in the process lifetime, avoiding repeated model loading:

  _model()      → SentenceTransformer (all-MiniLM-L6-v2), cached via lru_cache
  _collection() → chromadb.Collection (persistent cosine-space index), cached via lru_cache

Retrieval flow
--------------
  1. Encode the query with the embedding model.
  2. Build an optional ChromaDB where-clause from ticker / section / speaker_type filters.
  3. Query the collection; if the filtered subset is too small ChromaDB raises —
     the fallback retries without the filter rather than surfacing an error.
  4. Convert cosine distances to similarity scores (score = 1 − distance).

Public API
----------
search(query, ticker, section, speaker_type, n_results) → list[dict]
search_multi_company(query, tickers, n_per_company, ...) → list[dict]
format_context(chunks)                                   → str
test_search()                                            → None
"""

import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

# ── Configuration ─────────────────────────────────────────────────────────────

_REPO_ROOT      = Path(__file__).resolve().parent.parent
CHROMA_PATH     = str(_REPO_ROOT / "chroma_db")
COLLECTION_NAME = "earnings_calls"
EMBED_MODEL     = "all-MiniLM-L6-v2"

# Similarity score below which a retrieved chunk is considered off-topic.
# Score = 1 − cosine_distance; 0.35 was chosen empirically: it is low enough
# to admit loosely-worded questions while excluding clearly unrelated content.
RELEVANCE_THRESHOLD = 0.35

# ── Singleton initialisation ──────────────────────────────────────────────────

# Suppress noisy sentence-transformers warnings on import
warnings.filterwarnings("ignore", message=".*position_ids.*")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    """Return the embedding model, loading it exactly once per process."""
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _collection() -> chromadb.Collection:
    """Return the ChromaDB collection, connecting exactly once per process."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ── Filter builder ────────────────────────────────────────────────────────────

def _build_where(
    ticker:       Optional[str] = None,
    section:      Optional[str] = None,
    speaker_type: Optional[str] = None,
) -> Optional[dict]:
    """Build a ChromaDB where-clause from optional equality filters.

    Returns None when no filters are provided (an empty $and would raise in ChromaDB).
    Returns a single $eq clause when only one filter is active.
    Returns an $and clause when multiple filters are combined.
    """
    clauses: list[dict] = []

    if ticker:
        clauses.append({"ticker": {"$eq": ticker.upper()}})
    if section:
        clauses.append({"section": {"$eq": section}})
    if speaker_type:
        clauses.append({"speaker_type": {"$eq": speaker_type}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


# ── Core search ───────────────────────────────────────────────────────────────

def search(
    query:        str,
    ticker:       Optional[str] = None,
    section:      Optional[str] = None,
    speaker_type: Optional[str] = None,
    n_results:    int = 8,
) -> list[dict]:
    """Embed *query* and return the most similar chunks from ChromaDB.

    Parameters
    ----------
    query        : Natural-language analyst question to embed and retrieve against.
    ticker       : Optional ticker filter, e.g. "NVDA" or "AAPL". Case-insensitive.
    section      : Optional section filter — "prepared_remarks" | "qa" | "earnings_release".
    speaker_type : Optional speaker filter — "management" | "analyst" | "operator".
    n_results    : Maximum number of chunks to return (clamped to collection size).

    Returns
    -------
    List of chunk dicts, each containing:
      text, company, ticker, quarter, year,
      speaker, speaker_role, speaker_type, section, filing_date, score.
    score = 1 − cosine_distance (range 0–1; 1.0 = identical).
    Results are ordered by descending score (most relevant first).
    Returns an empty list when the collection is empty.
    """
    col   = _collection()
    model = _model()

    total = col.count()
    if total == 0:
        return []
    n = min(n_results, total)

    query_vec = model.encode([query]).tolist()
    where     = _build_where(ticker, section, speaker_type)

    kwargs: dict = dict(
        query_embeddings=query_vec,
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    if where is not None:
        kwargs["where"] = where

    try:
        raw = col.query(**kwargs)
    except Exception as exc:
        # ChromaDB raises when the filtered subset is smaller than n_results;
        # retry without the filter rather than surfacing the error to the caller.
        print(f"[searcher] filter query failed ({exc}); retrying without filter")
        kwargs.pop("where", None)
        raw = col.query(**kwargs)

    hits: list[dict] = []
    for doc, meta, dist in zip(
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        hits.append({
            "text":         doc,
            "company":      meta.get("company", ""),
            "ticker":       meta.get("ticker", ""),
            "quarter":      meta.get("quarter", ""),
            "year":         meta.get("year", ""),
            "speaker":      meta.get("speaker", ""),
            "speaker_role": meta.get("speaker_role", ""),
            "speaker_type": meta.get("speaker_type", ""),
            "section":      meta.get("section", ""),
            "filing_date":  meta.get("filing_date", ""),
            "score":        round(1.0 - float(dist), 4),
        })

    return hits


# ── Multi-company search ──────────────────────────────────────────────────────

def search_multi_company(
    query:         str,
    tickers:       list[str],
    n_per_company: int = 3,
    section:       Optional[str] = None,
    speaker_type:  Optional[str] = None,
) -> list[dict]:
    """Run a per-ticker search and merge results so no company dominates the context.

    Parameters
    ----------
    query         : Natural-language analyst question.
    tickers       : Ticker symbols to query individually, e.g. ["SNOW", "MDB"].
    n_per_company : Chunks to retrieve per ticker (default 3).
    section       : Optional section filter passed through to search().
    speaker_type  : Optional speaker_type filter passed through to search().

    Returns
    -------
    Combined list of chunk dicts sorted by score descending.
    Tickers not found in the collection are silently skipped.
    Duplicate chunks (same ticker + text prefix) are deduplicated.
    """
    seen_ids: set[str]  = set()
    combined: list[dict] = []

    for ticker in tickers:
        hits = search(
            query,
            ticker=ticker,
            section=section,
            speaker_type=speaker_type,
            n_results=n_per_company,
        )
        for hit in hits:
            dedup_key = f"{hit['ticker']}::{hit['text'][:80]}"
            if dedup_key not in seen_ids:
                seen_ids.add(dedup_key)
                combined.append(hit)

    combined.sort(key=lambda h: h["score"], reverse=True)
    return combined


# ── Context formatter ─────────────────────────────────────────────────────────

def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a single context string for LLM injection.

    Each chunk is rendered as a labelled block:

        ---
        [NVIDIA Corporation | Q3 2024 | Jensen Huang, CEO | Prepared Remarks]
        "actual text here"
        ---

    The header contains: company name | quarter + year | speaker name + role |
    section label (underscores replaced with spaces, title-cased).
    When speaker info is absent the section label is used as the speaker field.

    Returns "(no context retrieved)" when chunks is empty.
    """
    if not chunks:
        return "(no context retrieved)"

    parts: list[str] = []
    for chunk in chunks:
        company = chunk.get("company", "Unknown")
        quarter = chunk.get("quarter", "")
        year    = chunk.get("year", "")
        speaker = chunk.get("speaker", "")
        role    = chunk.get("speaker_role", "")
        section = chunk.get("section", "").replace("_", " ").title()
        text    = chunk.get("text", "")

        if speaker and role and speaker != "N/A":
            speaker_label = f"{speaker}, {role}"
        elif speaker and speaker != "N/A":
            speaker_label = speaker
        else:
            speaker_label = section or "Unknown"

        period = f"{quarter} {year}".strip()
        header = f"[{company} | {period} | {speaker_label} | {section}]"
        parts.append(f"---\n{header}\n\"{text}\"\n---")

    return "\n\n".join(parts)


# ── Self-test ─────────────────────────────────────────────────────────────────

def test_search() -> None:
    """Run 3 representative queries and print PASS / FAIL for each."""
    test_queries: list[tuple[str, dict]] = [
        ("What is NVIDIA guidance for next quarter?",          {"ticker": "NVDA"}),
        ("How is Airbnb performing in international markets?", {"ticker": "ABNB"}),
        ("What risks did Snowflake mention?",                  {"ticker": "SNOW"}),
    ]
    relevance_keywords: list[list[str]] = [
        ["guidance", "outlook", "revenue", "quarter", "billion", "expect"],
        ["international", "market", "nights", "booking", "growth", "travel"],
        ["risk", "competition", "challenge", "headwind", "macro", "uncertain"],
    ]

    col_size = _collection().count()
    print(f"\n{'=' * 60}")
    print(f"  Retrieval Self-Test  (collection: {col_size:,} docs)")
    print(f"{'=' * 60}")

    all_passed = True

    for (query, filters), kw_list in zip(test_queries, relevance_keywords):
        print(f"\nQuery : \"{query}\"")
        if filters:
            print(f"Filter: {filters}")
        print("-" * 60)

        results = search(query, n_results=3, **filters)

        passed = False
        for rank, r in enumerate(results, 1):
            snippet = r["text"][:120].replace("\n", " ")
            label   = (
                f"{r['company']} {r['quarter']} {r['year']} | "
                f"{r['speaker'] or 'N/A'} | score {r['score']:.3f}"
            )
            print(f"  [{rank}] {label}")
            print(f"       \"{snippet}…\"")

            if r["score"] >= RELEVANCE_THRESHOLD and any(
                kw in r["text"].lower() for kw in kw_list
            ):
                passed = True

        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  → {status}")
        if not passed:
            all_passed = False

    print(f"\n{'=' * 60}")
    overall = "All tests passed ✓" if all_passed else "Some tests failed — check collection"
    print(f"  {overall}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    test_search()
