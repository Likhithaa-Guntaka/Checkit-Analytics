# Checkit Analytics — Earnings Intelligence RAG System

**Ask natural-language questions about earnings calls and get structured, evidence-cited answers grounded in primary SEC filings.**

---

## What This Does

Checkit Analytics is a Retrieval-Augmented Generation (RAG) system that lets analysts query earnings call transcripts and press releases as if they had an expert research assistant who has read every filing. The system downloads official documents directly from SEC EDGAR and Motley Fool, splits them into speaker-aware chunks, embeds them into a local vector database, and routes analyst questions through a Groq-hosted LLM that is forced to cite its sources.

Every answer comes back as structured JSON with a confidence rating, sentiment label, risk flags, consistency score, direct quotes from transcripts, and an explicit statement of what the answer cannot tell you. The system covers 40 companies across 6 sectors, spanning 8 quarters (2023–2025), with over 6,500 searchable chunks of primary earnings content.

---

## Architecture
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
│                                              top-4 chunks          │
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
---

## Setup

### 1. Clone and navigate

```bash
git clone https://github.com/Likhithaa-Guntaka/Checkit-Analytics.git
cd Checkit-Analytics
```

### 2. Install dependencies

```bash
pip3 install -r requirements.txt
```

### 3. Add your Groq API key

Create a `.env` file in the project root:
Get a free key at [console.groq.com](https://console.groq.com).

### 4. Run the full setup pipeline

```bash
python3 pipeline.py
```

This runs all five stages in order:

STEP 1 / 5 — Download Transcripts    (~15–30 min for all 40 companies)
STEP 2 / 5 — Chunk Transcripts       (~1 minute)
STEP 3 / 5 — Embed into ChromaDB     (~3–5 min)
STEP 4 / 5 — Verify Retrieval        (~10 seconds)
STEP 5 / 5 — End-to-End RAG Test     (~5 seconds)
### 5. Launch the web UI

```bash
streamlit run app/streamlit_app.py
```

### 6. Run the evaluation suite

```bash
python3 eval/test_queries.py
```

---

## Data Coverage

**40 companies · 8 quarters · 2023–2025 · 6,500+ indexed chunks**

### Tech / Growth
| Company | Ticker |
|---------|--------|
| NVIDIA Corporation | NVDA |
| Apple Inc. | AAPL |
| Microsoft Corporation | MSFT |
| Tesla Inc. | TSLA |
| Meta Platforms Inc. | META |
| Palantir Technologies Inc. | PLTR |
| Snowflake Inc. | SNOW |
| Shopify Inc. | SHOP |
| Datadog Inc. | DDOG |
| CrowdStrike Holdings Inc. | CRWD |
| Uber Technologies Inc. | UBER |
| Airbnb Inc. | ABNB |
| ServiceNow Inc. | NOW |
| Workday Inc. | WDAY |
| Booking Holdings Inc. | BKNG |
| Intuitive Surgical Inc. | ISRG |
| Veeva Systems Inc. | VEEV |
| Cloudflare Inc. | NET |
| MongoDB Inc. | MDB |

### Cloud / SaaS
| Company | Ticker |
|---------|--------|
| Salesforce Inc. | CRM |
| Zoom Video Communications Inc. | ZM |
| HubSpot Inc. | HUBS |

### Consumer
| Company | Ticker |
|---------|--------|
| Amazon.com Inc. | AMZN |
| Nike Inc. | NKE |
| Starbucks Corporation | SBUX |

### Healthcare
| Company | Ticker |
|---------|--------|
| Eli Lilly and Company | LLY |
| UnitedHealth Group Inc. | UNH |
| Pfizer Inc. | PFE |
| Johnson & Johnson | JNJ |

### Finance
| Company | Ticker |
|---------|--------|
| Wells Fargo & Company | WFC |

### Sustainability
| Company | Ticker |
|---------|--------|
| NextEra Energy Inc. | NEE |
| Enphase Energy Inc. | ENPH |
| First Solar Inc. | FSLR |
| Rivian Automotive Inc. | RIVN |
| Lucid Group Inc. | LCID |
| Plug Power Inc. | PLUG |
| Bloom Energy Corp. | BE |
| Sunrun Inc. | RUN |
| ChargePoint Holdings Inc. | CHPT |
| Aptiv PLC | APTV |

---

## Example Queries

**1. Single company**
> *"What did NVIDIA say about data center demand in their latest earnings call?"*

**2. Cross-company comparison**
> *"How are Snowflake and MongoDB growing compared to each other?"*

**3. Risk and macro**
> *"What macro risks did companies mention most frequently?"*

**4. AI theme**
> *"What did management teams say about AI investment returns?"*

**5. Sustainability**
> *"What did Rivian say about production challenges and delivery guidance?"*

---

## Output Format

Every query returns a structured JSON object:

```json
{
  "answer": "Clear, direct answer in 2-3 sentences.",
  "key_points": ["Point 1", "Point 2", "Point 3"],
  "sentiment": "positive | negative | neutral | mixed",
  "confidence": "high | medium | low",
  "consistency": "aligned | mixed | conflict",
  "risk_flags": ["Risk 1", "Risk 2"],
  "evidence": [
    {
      "company": "NVIDIA Corporation",
      "ticker": "NVDA",
      "speaker": "Jensen Huang",
      "role": "CEO",
      "quarter": "Q3 2024",
      "quote": "Short verbatim quote under 30 words",
      "why_relevant": "One sentence explanation"
    }
  ],
  "limitations": "What this answer cannot tell you."
}
```

---

## Evaluation Results

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Grounding Accuracy | 46% | ≥75% | FAIL |
| Hallucination Rate | 0% | ≤20% | PASS |
| Reasoning Score | 3.5/5 | ≥3.5 | PASS |
| Cross-Source Consistency | 92.5% | ≥65% | PASS |
| Completeness | 100% | ≥70% | PASS |
| **Composite Score** | **76.2/100** | | **PASS** |
| Pass Rate | 80% | | |
| Average Latency | 11.4s | <5s | — |

> Grounding accuracy uses a strict formula (evidence / key_points). All answers have real citations — 0% hallucination rate confirms this. Cross-company queries score lower due to data asymmetry between companies, not retrieval failure.

---

## Key Design Decisions

- **RAG over fine-tuning:** Earnings data changes every quarter. RAG separates the knowledge store from the reasoning engine so new quarters require only a pipeline re-run, no retraining.

- **Speaker-aware chunking:** Each chunk preserves speaker identity, role, and section (prepared remarks vs Q&A), enabling filtered queries like "what did management say" vs "what did analysts ask."

- **ChromaDB with cosine similarity:** Runs embedded with no external process. Cosine similarity is invariant to text length, making it more reliable than Euclidean distance for sentence-length embeddings.

- **Groq llama-3.3-70b:** Strong enough to follow strict JSON-only prompts reliably. Includes a two-attempt retry loop with a stricter correction prompt if the first response fails to parse.

---

## Limitations

- Coverage is fixed at download time. Re-run `pipeline.py` to add new quarters.
- Not all companies file full transcripts with the SEC. Press releases are used as fallback.
- Cross-company comparisons depend on data symmetry between companies.
- Groq free tier has a 100k token/day limit. Space out large evaluation runs.
- Numerical figures come from natural language, not structured XBRL data.

---

## Future Improvements

- [ ] Incremental ingestion via EDGAR RSS feed
- [ ] Cross-encoder reranker for better retrieval precision
- [ ] XBRL financial data layer for exact revenue and EPS figures
- [ ] Multi-quarter trend analysis
- [ ] Streaming responses in the UI
- [ ] Query history export to PDF/Excel

---

## Disclaimer

> This is a research tool built to demonstrate RAG system design. It is not financial advice. Always consult official filings and a qualified financial advisor before making investment decisions.

---

*Built with Python · ChromaDB · Groq · Streamlit · SEC EDGAR*
