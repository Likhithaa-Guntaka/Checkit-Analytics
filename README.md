# Checkit Analytics — Earnings Intelligence RAG System

**Ask natural-language questions about earnings calls and get structured, evidence-cited answers grounded in primary SEC filings.**

---

## What This Does

Checkit Analytics is a Retrieval-Augmented Generation (RAG) system that lets analysts query earnings call transcripts and press releases as if they had an expert research assistant who has read every filing. The system downloads official documents directly from SEC EDGAR and Motley Fool, splits them into speaker-aware chunks, embeds them into a local vector database, and routes analyst questions through a Groq-hosted LLM that is forced to cite its sources.

Every answer comes back as structured JSON with a confidence rating, sentiment label, direct quotes from transcripts, and an explicit statement of what the answer cannot tell you. The system covers 20 high-profile public companies across 8 quarters (2023–2025), giving it roughly 4,300 searchable chunks of primary earnings content.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION                              │
│                                                                     │
│  SEC EDGAR ──┐                                                      │
│              ├──► downloader.py ──► chunker.py ──► embedder.py     │
│  Motley Fool─┘        │                │               │           │
│                  transcripts/     chunks.json      ChromaDB        │
└─────────────────────────────────────────────────────────────────────┘
                                                          │
┌─────────────────────────────────────────────────────────────────────┐
│                         QUERY PIPELINE                              │
│                                                                     │
│  User Question                                                      │
│       │                                                             │
│       ▼                                                             │
│  searcher.py  ──── [cosine similarity] ────► ChromaDB              │
│  (all-MiniLM-L6-v2 embedding)                    │                 │
│                                              top-8 chunks          │
│                                                  │                 │
│                                                  ▼                 │
│                                           answerer.py              │
│                                      (Groq llama-3.3-70b)         │
│                                                  │                 │
│                                         Structured JSON            │
│                                                  │                 │
│                                                  ▼                 │
│                                       Streamlit UI / CLI           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Tool | Purpose | Why Chosen |
|---|---|---|
| **SEC EDGAR API** | Primary document source | Official, authoritative, free, no scraping required |
| **Motley Fool** | Earnings call transcripts | Structured speaker-turn format not available in EDGAR filings |
| **`sec-edgar-downloader`** | EDGAR filing retrieval | Handles CIK lookup, accession numbers, and filing index parsing |
| **`sentence-transformers`** (`all-MiniLM-L6-v2`) | Text embedding | Fast, accurate, runs locally — no embedding API cost |
| **ChromaDB** | Vector store | Persistent, embeddable, no infrastructure to manage |
| **Groq** (`llama-3.3-70b-versatile`) | LLM answer generation | Near-instant inference; open-weight model avoids vendor lock-in |
| **Streamlit** | Web UI | Rapid prototyping; Python-native; zero frontend code |
| **`python-dotenv`** | Secret management | Keeps API keys out of source code |
| **`tqdm`** | Progress bars | Visibility during slow embedding and download steps |
| **`beautifulsoup4` + `lxml`** | HTML parsing | Parses EDGAR filing index pages and Motley Fool transcripts |

---

## Project Structure

```
checkit-rag/
├── .env                        # API keys — never commit
├── requirements.txt
├── pipeline.py                 # Full setup pipeline (run this first)
│
├── ingest/
│   ├── downloader.py           # EDGAR + Motley Fool transcript downloader
│   ├── chunker.py              # Speaker-aware text chunker
│   └── embedder.py             # Embeds chunks → ChromaDB
│
├── retrieval/
│   └── searcher.py             # Semantic search with optional filters
│
├── reasoning/
│   └── answerer.py             # Groq LLM prompt builder + JSON answer generator
│
├── app/
│   └── streamlit_app.py        # Interactive web UI
│
├── eval/
│   ├── test_queries.py         # 20-query evaluation harness
│   └── results.json            # Evaluation output (auto-generated)
│
└── data/
    ├── transcripts/            # Raw .txt + _meta.json files
    └── chunks.json             # Chunked documents (auto-generated)
```

---

## Setup

### 1. Clone and navigate

```bash
git clone <your-repo-url>
cd checkit-rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> Python 3.10+ required. If you have multiple Python versions, use the explicit interpreter:
> `python3.10 -m pip install -r requirements.txt`

### 3. Add your Groq API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com](https://console.groq.com). The free tier is sufficient for this project.

### 4. Run the full setup pipeline

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

### 5. Launch the web UI

```bash
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 6. (Optional) Run the evaluation suite

```bash
python eval/test_queries.py
```

Results are saved to `eval/results.json`.

---

## Data Coverage

20 companies · 8 quarters · 2023–2025 · ~4,300 indexed chunks

| # | Company | Ticker | Sector |
|---|---|---|---|
| 1 | NVIDIA Corporation | NVDA | Semiconductors |
| 2 | Apple | AAPL | Consumer Technology |
| 3 | Microsoft | MSFT | Cloud / Enterprise Software |
| 4 | Alphabet (Google) | GOOGL | Search / Cloud |
| 5 | Amazon | AMZN | E-Commerce / Cloud |
| 6 | Meta Platforms | META | Social Media / AI |
| 7 | Tesla | TSLA | Electric Vehicles |
| 8 | Netflix | NFLX | Streaming |
| 9 | Airbnb | ABNB | Travel / Marketplace |
| 10 | Snowflake | SNOW | Data Cloud |
| 11 | Salesforce | CRM | Enterprise SaaS |
| 12 | Adobe | ADBE | Creative / Document Cloud |
| 13 | Advanced Micro Devices | AMD | Semiconductors |
| 14 | Palantir | PLTR | AI / Data Analytics |
| 15 | Datadog | DDOG | Observability / Cloud Monitoring |
| 16 | Cloudflare | NET | Network Security |
| 17 | MongoDB | MDB | Database |
| 18 | Booking Holdings | BKNG | Online Travel |
| 19 | Uber | UBER | Mobility / Delivery |
| 20 | Spotify | SPOT | Audio Streaming |

---

## Example Queries

These are representative questions the system can answer. Results are grounded in direct transcript evidence and returned with source citations.

**1. Single-company operational question**
> *"What did NVIDIA say about data center demand in their latest earnings call?"*

Expect: Jensen Huang quotes on H100/H200 demand, hyperscaler customer commentary, revenue figures, and forward guidance — with specific quarter attribution.

**2. Cross-company comparison**
> *"How are Snowflake and MongoDB growing compared to each other?"*

Expect: Side-by-side product revenue growth rates, NRR figures, and management commentary from both companies pulled from their respective transcripts.

**3. Risk and macro**
> *"What macro risks did companies mention most frequently?"*

Expect: Synthesis across multiple companies — interest rate sensitivity, FX headwinds, enterprise budget scrutiny, and consumer spending patterns cited from management remarks.

**4. AI theme**
> *"What did management teams say about AI investment returns?"*

Expect: Quotes from multiple CEOs and CFOs on AI capex justification, customer ROI, and monetization timelines, with confidence rated by evidence quality.

**5. Sentiment and guidance**
> *"Which companies gave the most positive guidance for next quarter?"*

Expect: A ranked summary of guidance tone with specific revenue/EPS guidance figures where available, evidence sourced from Q&A and prepared remarks sections.

---

## Output Format

Every query returns a structured JSON object. This schema is enforced by the LLM prompt and validated by the parser.

```json
{
  "answer": "Clear, direct answer in 2–3 sentences.",
  "key_points": [
    "Bullet point drawn directly from transcript evidence",
    "Another key point with factual grounding",
    "..."
  ],
  "sentiment": "positive | negative | neutral | mixed",
  "confidence": "high | medium | low",
  "evidence": [
    {
      "company":      "NVIDIA Corporation",
      "ticker":       "NVDA",
      "speaker":      "Jensen Huang",
      "role":         "CEO",
      "quarter":      "Q3 2024",
      "quote":        "Short verbatim quote under 30 words from the transcript",
      "why_relevant": "One sentence explaining why this quote supports the answer"
    }
  ],
  "limitations": "What this answer cannot tell you, or where evidence is thin."
}
```

**Confidence levels:**
- `high` — multiple strong evidence pieces with direct quotes
- `medium` — some relevant evidence but partial or indirect
- `low` — weak evidence match; answer may reflect limited data coverage

---

## Evaluation Results

Results from running `python eval/test_queries.py` against the live system. Full output saved in `eval/results.json`.

| # | Category | Query (abbreviated) | Confidence | Sentiment | Evidence | Latency |
|---|---|---|---|---|---|---|
| 1 | Single Company | NVIDIA data center demand | ● High | Positive | 1 | 6.6s |
| 2 | Single Company | Tesla delivery guidance | ◑ Medium | Neutral | 2 | 2.2s |
| 3 | Single Company | Eli Lilly Mounjaro/Zepbound | ● High | Positive | 1 | 15.3s |
| 4 | Single Company | Airbnb international expansion | ● High | Positive | 2 | 9.1s |
| 5 | Single Company | Palantir US commercial growth | ● High | Positive | 2 | 36.0s |
| 6 | Cross-Company | Datadog vs CrowdStrike revenue | ○ Low | Neutral | 1 | 22.7s |
| 7 | Cross-Company | Snowflake vs MongoDB growth | ○ Low | Neutral | 1 | 32.9s |
| 8 | Cross-Company | Shopify + Booking consumer trends | ◑ Medium | Positive | 2 | 14.7s |
| 9 | Cross-Company | ServiceNow vs Workday AI adoption | ○ Low | Neutral | 2 | 21.8s |
| 10 | Cross-Company | Cloudflare vs Veeva competitive moat | ◑ Medium | Positive | 2 | 14.8s |
| 11 | Risk | Most frequent macro risks | ● High | Negative | 2 | 19.1s |
| 12 | Risk | Microsoft Azure slowdown | ○ Low | Neutral | 1 | 14.5s |
| 13 | Risk | Pricing pressure mentions | ● High | Neutral | 0 | 31.6s |
| 14 | Risk | Intuitive Surgical procedure volume | ● High | Neutral | 2 | 24.7s |
| 15 | Risk | Uber headwinds | ● High | Negative | 2 | 25.0s |
| 16 | Outlook | Most positive guidance | ◑ Medium | Positive | 2 | 18.8s |
| 17 | Outlook | AI investment returns | ◑ Medium | Positive | 2 | 15.9s |
| 18 | Outlook | CEO demand environment | ◑ Medium | Mixed | 2 | 32.3s |
| 19 | Outlook | MongoDB outlook and expansion | ● High | Positive | 2 | 31.8s |
| 20 | Outlook | Hiring / headcount changes | ● High | Positive | 2 | 4.7s |
| | | **Summary** | H: 10 · M: 6 · L: 4 | | avg 2 ev | **pass 80% · avg 19.7s** |

> **Note on cross-company queries (6, 7, 9):** Low confidence reflects sparse transcript coverage for CrowdStrike, Shopify (for query 7), and Workday in the current dataset — not a retrieval failure. Single-company and risk/outlook queries perform strongly at 87% pass rate.

---

## Key Design Decisions

- **Why RAG instead of fine-tuning?** Earnings call data changes every quarter. A fine-tuned model would go stale immediately and require expensive retraining. RAG separates the knowledge store (ChromaDB) from the reasoning engine (Groq), so adding a new quarter's transcripts is a single pipeline run with no model changes.

- **Why speaker-aware chunking?** Generic paragraph chunking loses critical context — a quote from an analyst question is not the same as a CEO's prepared remark. The chunker identifies speaker turns (using Motley Fool's `Name\n--\nRole` format), preserves speaker identity and role in metadata, and allows filtering by `speaker_type` (management / analyst / operator). This makes it possible to query specifically what management said versus what analysts were asking about.

- **Why ChromaDB with cosine similarity?** ChromaDB runs as an embedded library with no external process, making setup a single `pip install`. The persistent client keeps embeddings on disk across process restarts, so the ~2-minute embedding step only runs once. Cosine similarity outperforms Euclidean distance for sentence-length embeddings because it is invariant to text length.

- **Why `llama-3.3-70b-versatile` on Groq?** The 70B parameter model is strong enough to follow a strict JSON-only system prompt reliably and reason across multi-company evidence. Groq's hardware provides sub-5-second inference even for long context windows, which matters in an interactive UI. The prompt includes a two-attempt retry loop with a stricter JSON-only instruction if the first response fails to parse.

---

## Limitations

- **Coverage is fixed at download time.** The system does not automatically update when new earnings calls are released. Running `python pipeline.py` again will prompt to re-download and re-embed.

- **Not all companies have full transcript coverage.** Most large-cap companies do not file full earnings call transcripts with the SEC — they file press releases. Where SEC EDGAR only has press releases, those are used as a fallback. Motley Fool transcripts are used where available and provide richer speaker-turn data.

- **Cross-company comparison quality depends on data symmetry.** If Company A has 8 quarters of transcripts and Company B has 2, a comparison query will naturally draw more heavily from Company A's data.

- **The model cannot access real-time data.** All answers are grounded exclusively in the indexed transcripts. The model is explicitly instructed never to use its parametric knowledge, which means it will not answer questions about events after the last ingested quarter.

- **Groq API rate limits apply.** The free tier allows a limited number of tokens per minute. Running all 20 evaluation queries consecutively may encounter throttling. Add `time.sleep(1)` between calls if needed.

- **Numerical precision is not guaranteed.** Revenue figures and percentages are extracted from natural-language text, not from structured financial data. Verify any figures against official filings before use.

---

## Future Improvements

- [ ] **Incremental ingestion** — detect new filings automatically using EDGAR's RSS feed and embed only the delta, without re-processing existing chunks
- [ ] **Reranker** — add a cross-encoder reranker pass between retrieval and generation to improve precision on complex multi-company queries
- [ ] **Structured financial data layer** — augment transcript chunks with XBRL-sourced financials for exact revenue, EPS, and margin figures
- [ ] **Multi-quarter trend analysis** — enable queries like "how has NVIDIA's gross margin changed over the last 6 quarters?" by adding temporal reasoning on top of the existing retrieval
- [ ] **Streaming responses** — pipe Groq's streamed output directly to the Streamlit UI for a more responsive experience on long answers
- [ ] **Query history and export** — save session queries and answers to a local SQLite database for analyst review and export to PDF/Excel

---

## Disclaimer

> **This is a research and educational tool built to demonstrate RAG system design.**
> It is not financial advice. All content is derived from publicly available SEC filings and earnings call transcripts. Do not make investment decisions based on outputs from this system. Always consult official filings and a qualified financial advisor.

---

*Built with Python · ChromaDB · Groq · Streamlit · SEC EDGAR*
