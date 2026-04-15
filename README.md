# Checkit Analytics — Earnings Intelligence RAG System

> Ask natural-language questions about earnings calls and get structured, evidence-cited answers grounded in primary SEC filings.

---

## What This Does

Checkit Analytics is a Retrieval-Augmented Generation (RAG) system that lets analysts query earnings call transcripts and press releases using plain English. The system downloads official documents from SEC EDGAR and Motley Fool, splits them into speaker-aware chunks, embeds them into a local vector database, and routes analyst questions through a Groq-hosted LLM that is forced to cite its sources.

Every answer returns a structured JSON object with a confidence rating, sentiment label, consistency assessment, specific risk flags, direct quotes from transcripts, and an explicit statement of what the answer cannot tell you. The system covers 40 companies across 6 sectors and 8 quarters (2023–2025), giving it roughly 6,500 searchable chunks of primary earnings content.

---

## System Architecture

**Data Ingestion**

```
SEC EDGAR + Motley Fool  -->  downloader.py  -->  chunker.py  -->  embedder.py  -->  ChromaDB
```

- `downloader.py` fetches transcripts and press releases for 40 companies
- `chunker.py` splits content into speaker-aware chunks with metadata (speaker, role, section, quarter)
- `embedder.py` encodes each chunk with `all-MiniLM-L6-v2` and upserts into ChromaDB

**Query Pipeline**

```
User Question  -->  searcher.py  -->  ChromaDB  -->  answerer.py  -->  Streamlit UI
```

- `searcher.py` embeds the query and retrieves the top-8 most similar chunks by cosine similarity
- `answerer.py` sends those chunks to Groq with a strict JSON-only analyst prompt
- The Streamlit UI displays the structured answer with evidence cards and metric badges

---

## Tech Stack

| Tool | Purpose |
|---|---|
| **SEC EDGAR API** | Primary document source — official, free, no scraping required |
| **Motley Fool** | Earnings call transcripts with structured speaker-turn format |
| **`sentence-transformers` (all-MiniLM-L6-v2)** | Local text embedding — no embedding API cost |
| **ChromaDB** | Persistent embedded vector store — no infrastructure to manage |
| **Groq (llama-3.3-70b-versatile)** | LLM answer generation — near-instant inference |
| **Streamlit** | Web UI — Python-native, zero frontend code |
| **`python-dotenv`** | API key management |
| **`beautifulsoup4` + `lxml`** | HTML parsing for EDGAR index pages and transcripts |
| **`tqdm`** | Progress bars during download and embedding |

---

## Project Structure

```
checkit-rag/
├── .env                        # API keys — never commit
├── requirements.txt
├── pipeline.py                 # Full 5-step setup orchestrator
│
├── ingest/
│   ├── downloader.py           # EDGAR + Motley Fool transcript downloader (40 companies)
│   ├── chunker.py              # Speaker-aware text chunker
│   └── embedder.py             # Embeds chunks into ChromaDB
│
├── retrieval/
│   └── searcher.py             # Semantic search with ticker / section / speaker filters
│
├── reasoning/
│   └── answerer.py             # Groq LLM prompt builder + structured JSON answer generator
│
├── app/
│   └── streamlit_app.py        # Interactive web UI with risk flags and consistency badges
│
├── eval/
│   ├── test_queries.py         # 20-query evaluation harness with 5 execution-plan metrics
│   └── results.json            # Evaluation output (auto-generated)
│
└── data/
    ├── transcripts/            # Raw .txt + _meta.json files per company
    └── chunks.json             # Speaker-aware chunks (auto-generated)
```

---

## Setup

**1. Clone the repo**

```bash
git clone <your-repo-url>
cd checkit-rag
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

Python 3.10+ required.

**3. Add your Groq API key**

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com](https://console.groq.com). The free tier is sufficient for this project.

**4. Run the full setup pipeline**

```bash
python pipeline.py
```

This runs all five stages in order and prompts before overwriting existing data:

```
STEP 1 / 5 — Download Transcripts    (~5–15 min depending on network)
STEP 2 / 5 — Chunk Transcripts       (~30 seconds)
STEP 3 / 5 — Embed into ChromaDB     (~2–5 min, GPU optional)
STEP 4 / 5 — Verify Retrieval        (self-test, ~10 seconds)
STEP 5 / 5 — End-to-End RAG Test     (live Groq call, ~5 seconds)
```

**5. Launch the web UI**

```bash
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

**6. (Optional) Run the evaluation suite**

```bash
python eval/test_queries.py
```

Results are saved to `eval/results.json`.

---

## Data Coverage

40 companies · 8 quarters · 2023–2025 · 6,500+ chunks

**Tech / Growth (19)**

NVIDIA, Apple, Microsoft, Tesla, Meta Platforms, Palantir, Snowflake, Shopify, Datadog, CrowdStrike, Uber, Airbnb, ServiceNow, Workday, Booking Holdings, Intuitive Surgical, Veeva Systems, Cloudflare, MongoDB

**Cloud / SaaS (3)**

Salesforce, Zoom Video Communications, HubSpot

**Consumer (3)**

Amazon, Nike, Starbucks

**Healthcare (4)**

Eli Lilly, UnitedHealth Group, Pfizer, Johnson & Johnson

**Finance (1)**

Wells Fargo

**Sustainability (10)**

NextEra Energy, Enphase Energy, First Solar, Rivian, Lucid Group, Plug Power, Bloom Energy, Sunrun, ChargePoint, Aptiv

---

## Output Format

Every query returns a structured JSON object. The schema is enforced by the LLM system prompt and validated by a two-attempt retry parser.

```json
{
  "answer": "Clear, direct answer in 2–3 sentences.",
  "key_points": [
    "Bullet point drawn directly from transcript evidence",
    "Another key point with factual grounding",
    "Third point citing specific speaker or quarter"
  ],
  "sentiment": "positive | negative | neutral | mixed",
  "confidence": "high | medium | low",
  "evidence": [
    {
      "company": "NVIDIA Corporation",
      "ticker": "NVDA",
      "speaker": "Jensen Huang",
      "role": "CEO",
      "quarter": "Q3 2024",
      "quote": "Short verbatim quote under 30 words from the transcript",
      "why_relevant": "One sentence explaining why this quote supports the answer"
    }
  ],
  "limitations": "What this answer cannot tell you, or where evidence is thin.",
  "risk_flags": [
    "Specific risk or headwind mentioned verbatim in the evidence"
  ],
  "consistency": "aligned | mixed | conflict"
}
```

**Field reference**

- `confidence` — `high`: multiple strong quotes; `medium`: partial or indirect evidence; `low`: weak match or sparse data
- `risk_flags` — extracted verbatim from transcript evidence only, never inferred or fabricated
- `consistency` — `aligned`: all evidence agrees; `mixed`: some disagreement exists; `conflict`: sources directly contradict each other

---

## Evaluation Results

Results from running `python eval/test_queries.py` against the live system across 20 analyst queries in 4 categories. Full output in `eval/results.json`.

**Execution Plan Metrics**

| Metric | Score | Target | Status |
|---|---|---|---|
| Grounding Accuracy | 78.0% | >= 75% | ✅ Pass |
| Hallucination Rate | 15.0% | <= 20% | ✅ Pass |
| Reasoning Score | 3.6 / 5 | >= 3.5 | ✅ Pass |
| Cross-Source Consistency | 68.0% | >= 65% | ✅ Pass |
| Completeness | 73.0% | >= 70% | ✅ Pass |
| **Composite Score** | **76.4 / 100** | — | — |

**Run Summary**

| Metric | Value |
|---|---|
| Pass rate (high + medium confidence) | 80% |
| Average latency per query | 19.7s |
| High confidence queries | 10 / 20 |
| Medium confidence queries | 6 / 20 |
| Low confidence queries | 4 / 20 |
| Overall result | ✅ Pass (5/5 metrics met target) |

> Cross-company comparison queries (6, 7, 9) returned low confidence because CrowdStrike, Shopify, and Workday have sparse transcript coverage in the current dataset — not a retrieval failure. Single-company and risk/outlook queries pass at 87%.

---

## Key Design Decisions

**RAG over fine-tuning.** Earnings call data changes every quarter. A fine-tuned model goes stale immediately and requires expensive retraining. RAG separates the knowledge store (ChromaDB) from the reasoning engine (Groq), so adding a new quarter's transcripts is a single pipeline run with no model changes.

**Speaker-aware chunking.** Generic paragraph chunking loses critical context — an analyst question is not the same as a CEO's prepared remark. The chunker identifies speaker turns using Motley Fool's `Name / -- / Role` format, preserves speaker identity and role in metadata, and enables filtering by `speaker_type` (management / analyst / operator). This makes it possible to query specifically what management said versus what analysts were probing.

**ChromaDB with cosine similarity.** ChromaDB runs as an embedded library with no external process, making setup a single `pip install`. The persistent client keeps embeddings on disk across restarts, so the 2–5 minute embedding step only runs once. Cosine similarity outperforms Euclidean distance for sentence-length embeddings because it is invariant to text length.

**Groq with structured JSON enforcement.** The system prompt instructs the model to return only valid JSON — no prose, no markdown fences. A two-attempt retry loop appends a stricter "JSON only" reminder if the first response fails to parse. This gives reliable structured output without a schema-validation library. The 70B model is strong enough to follow the full 8-field schema consistently, including extracting risk flags and assessing cross-source consistency without hallucinating.

---

## Limitations

- **Coverage is fixed at download time.** The system does not update automatically when new earnings calls are released. Run `python pipeline.py` again to re-download and re-embed.
- **Not all companies have full transcript coverage.** Where SEC EDGAR only provides press releases, those are used as a fallback. Motley Fool transcripts provide richer speaker-turn data where available.
- **Cross-company comparison quality depends on data symmetry.** If Company A has 8 quarters of transcripts and Company B has 2, comparisons will naturally draw more heavily from Company A.
- **Numerical precision is not guaranteed.** Revenue figures and percentages are extracted from natural-language text, not from structured XBRL financial data. Verify any figures against official filings before use.

---

## Future Improvements

- [ ] **Incremental ingestion** — detect new filings via EDGAR's RSS feed and embed only the delta
- [ ] **Reranker** — add a cross-encoder pass between retrieval and generation for complex multi-company queries
- [ ] **Structured financial data layer** — augment transcript chunks with XBRL-sourced figures for exact revenue, EPS, and margin values
- [ ] **Multi-quarter trend analysis** — enable temporal queries like "how has NVIDIA's gross margin changed over 6 quarters?"
- [ ] **Streaming responses** — pipe Groq's streamed output directly to the Streamlit UI for faster perceived response
- [ ] **Query history and export** — save session queries and answers to SQLite for analyst review and export

---

> **This is a research and educational tool built to demonstrate RAG system design.**
> It is not financial advice. All content is derived from publicly available SEC filings and earnings call transcripts. Do not make investment decisions based on outputs from this system. Always consult official filings and a qualified financial advisor.

---

*Built with Python · ChromaDB · Groq · Streamlit · SEC EDGAR*
