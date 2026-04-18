#!/usr/bin/env python3
"""
Checkit Analytics RAG — Incremental 8-K Filing Scheduler

Checks SEC EDGAR for new 8-K filings from our 40 tracked companies,
downloads any new content, re-chunks new files, and adds new embeddings
to ChromaDB without touching existing data.

SCHEDULING INSTRUCTIONS
-----------------------
Mac/Linux — add to crontab to run every Monday at 9am:

  1. Open terminal and run:   crontab -e
  2. Add this line (update the path):
       0 9 * * 1 /Users/YOUR_USERNAME/Documents/checkit-rag/run_update.sh
  3. Save and exit (in vim: :wq).
  4. Verify the entry:        crontab -l

Or run manually at any time:
  python3 scheduler.py
"""

import json
import logging
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

TRANSCRIPTS_DIR = ROOT / "data" / "transcripts"
CHUNKS_FILE = ROOT / "data" / "chunks.json"
CHROMA_PATH = str(ROOT / "chroma_db")
LOGS_DIR = ROOT / "logs"

# ── Logging ────────────────────────────────────────────────────────────────
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "update.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
EDGAR_RSS_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&type=8-K&dateb=&owner=include"
    "&count=40&search_text=&output=atom"
)
COLLECTION_NAME = "earnings_calls"
EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64
META_FIELDS = [
    "company", "ticker", "quarter", "year",
    "section", "speaker", "speaker_role", "speaker_type", "filing_date",
]
HEADERS = {
    "User-Agent": "Checkit Analytics LLC contact@checkitanalytics.com",
    "Accept-Encoding": "gzip, deflate",
}


# ── Step 1: discover what we already have ─────────────────────────────────

def get_existing_filings() -> dict[str, str]:
    """Return {ticker: latest_filing_date} from sidecar _meta.json files."""
    latest: dict[str, str] = {}
    for meta_path in TRANSCRIPTS_DIR.glob("*_meta.json"):
        try:
            meta = json.loads(meta_path.read_text())
            ticker = meta.get("ticker", "")
            date = meta.get("filing_date", "")
            if ticker and date and date > latest.get(ticker, ""):
                latest[ticker] = date
        except Exception as exc:
            log.warning("Could not read %s: %s", meta_path.name, exc)
    return latest


# ── Step 2: fetch RSS feed ────────────────────────────────────────────────

def fetch_rss_ciks() -> dict[str, str]:
    """
    Parse the EDGAR Atom feed and return {cik: filing_date} for recent 8-K filers.
    CIK values are normalised (leading zeros stripped) so they match COMPANIES dicts.
    """
    log.info("Fetching EDGAR RSS feed: %s", EDGAR_RSS_URL)
    try:
        resp = requests.get(EDGAR_RSS_URL, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        log.error("Failed to fetch RSS feed: %s", exc)
        return {}

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as exc:
        log.error("Failed to parse RSS XML: %s", exc)
        return {}

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    filings: dict[str, str] = {}

    for entry in root.findall("atom:entry", ns):
        link_elem = entry.find("atom:link", ns)
        title_elem = entry.find("atom:title", ns)
        updated_elem = entry.find("atom:updated", ns)

        filing_date = ""
        if updated_elem is not None and updated_elem.text:
            filing_date = updated_elem.text[:10]  # "YYYY-MM-DD"

        cik: str | None = None

        # Primary: extract CIK from href  .../edgar/data/{CIK}/...
        if link_elem is not None:
            href = link_elem.get("href", "")
            parts = href.split("/edgar/data/")
            if len(parts) > 1:
                raw = parts[1].split("/")[0].strip()
                if raw.isdigit():
                    cik = str(int(raw))

        # Fallback: extract 10-digit CIK in parentheses from title
        if cik is None and title_elem is not None and title_elem.text:
            m = re.search(r"\((\d{10})\)", title_elem.text)
            if m:
                cik = str(int(m.group(1)))

        if cik and filing_date:
            filings[cik] = filing_date

    log.info("RSS feed contained %d recent 8-K entries", len(filings))
    return filings


# ── Step 3: find companies with new filings ───────────────────────────────

def find_companies_with_new_filings(
    rss_filings: dict[str, str],
    existing_filings: dict[str, str],
    companies: list[dict],
) -> list[dict]:
    """
    Return the subset of our 40 companies that appear in the RSS feed
    with a filing date newer than what we already have on disk.
    """
    to_update: list[dict] = []
    for company in companies:
        ticker = company["ticker"]
        cik_raw = company.get("cik", "")
        # Normalise CIK to match rss_filings keys (strip leading zeros)
        try:
            cik = str(int(cik_raw)) if cik_raw else ""
        except ValueError:
            cik = ""

        if not cik or cik not in rss_filings:
            continue

        rss_date = rss_filings[cik]
        our_date = existing_filings.get(ticker, "")

        if rss_date > our_date:
            log.info(
                "New 8-K for %s  |  RSS date: %s  |  our latest: %s",
                ticker, rss_date, our_date or "none",
            )
            to_update.append(company)

    return to_update


# ── Step 4: download ──────────────────────────────────────────────────────

def download_companies(tickers: list[str]) -> list[Path]:
    """Download new filings for given tickers; return newly created .txt paths."""
    before = set(TRANSCRIPTS_DIR.glob("*.txt"))

    from ingest.downloader import download_specific_companies  # noqa: PLC0415
    log.info("Downloading filings for: %s", tickers)
    download_specific_companies(tickers)

    after = set(TRANSCRIPTS_DIR.glob("*.txt"))
    new_files = sorted(after - before)
    log.info("Download complete — %d new .txt file(s) created", len(new_files))
    return new_files


# ── Step 5: chunk new files ───────────────────────────────────────────────

def chunk_new_files(new_txt_paths: list[Path]) -> list[dict]:
    """
    Chunk only the newly downloaded files and append the results to chunks.json.
    Returns the list of brand-new chunks.
    """
    from ingest.chunker import process_file  # noqa: PLC0415

    existing_chunks: list[dict] = []
    if CHUNKS_FILE.exists():
        try:
            existing_chunks = json.loads(CHUNKS_FILE.read_text())
        except Exception as exc:
            log.warning("Could not load existing chunks.json: %s", exc)

    existing_ids = {c["chunk_id"] for c in existing_chunks}
    new_chunks: list[dict] = []

    for txt_path in new_txt_paths:
        try:
            file_chunks = process_file(txt_path)
            for chunk in file_chunks:
                if chunk["chunk_id"] not in existing_ids:
                    new_chunks.append(chunk)
                    existing_ids.add(chunk["chunk_id"])
            log.info("Chunked %s → %d chunk(s)", txt_path.name, len(file_chunks))
        except Exception as exc:
            log.error("Chunking failed for %s: %s", txt_path.name, exc)

    if new_chunks:
        all_chunks = existing_chunks + new_chunks
        CHUNKS_FILE.write_text(json.dumps(all_chunks, indent=2))
        log.info(
            "chunks.json updated: +%d new chunks (total: %d)",
            len(new_chunks), len(all_chunks),
        )
    else:
        log.info("No new chunks produced from downloaded files")

    return new_chunks


# ── Step 6: embed new chunks ──────────────────────────────────────────────

def embed_new_chunks(new_chunks: list[dict]) -> int:
    """
    Embed new chunks and upsert them into ChromaDB.
    Existing embeddings are never touched.
    Returns the number of chunks actually embedded.
    """
    if not new_chunks:
        return 0

    import chromadb  # noqa: PLC0415
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    log.info("Connecting to ChromaDB at %s", CHROMA_PATH)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Skip any chunks already present (idempotent by chunk_id)
    existing_ids = set(collection.get(include=[])["ids"])
    to_embed = [c for c in new_chunks if c["chunk_id"] not in existing_ids]

    if not to_embed:
        log.info("All new chunks are already present in ChromaDB — nothing to embed")
        return 0

    log.info("Loading embedding model %s...", EMBED_MODEL)
    model = SentenceTransformer(EMBED_MODEL)

    log.info("Embedding %d chunk(s) in batches of %d...", len(to_embed), BATCH_SIZE)
    total_embedded = 0

    for i in range(0, len(to_embed), BATCH_SIZE):
        batch = to_embed[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{k: c.get(k, "") for k in META_FIELDS} for c in batch],
        )
        total_embedded += len(batch)
        log.info("  Batch %d/%d embedded", i // BATCH_SIZE + 1, -(-len(to_embed) // BATCH_SIZE))

    log.info("ChromaDB upsert complete — %d new chunk(s) embedded", total_embedded)
    return total_embedded


# ── Orchestrator ──────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 60)
    log.info("Checkit Analytics RAG — incremental update started")
    log.info("=" * 60)

    from ingest.downloader import COMPANIES  # noqa: PLC0415

    # Fetch RSS and determine which companies have new 8-Ks
    rss_filings = fetch_rss_ciks()
    if not rss_filings:
        log.warning("RSS fetch returned no data — aborting update")
        print("\nSummary: 0 new transcripts added, 0 new chunks embedded")
        return

    existing_filings = get_existing_filings()
    log.info("We currently have filings for %d ticker(s)", len(existing_filings))

    companies_to_update = find_companies_with_new_filings(
        rss_filings, existing_filings, COMPANIES
    )

    if not companies_to_update:
        log.info("No new 8-K filings detected for any of our 40 companies")
        print("\nSummary: 0 new transcripts added, 0 new chunks embedded")
        return

    tickers = [c["ticker"] for c in companies_to_update]
    log.info("%d company/companies with new filings: %s", len(tickers), tickers)

    # Download
    new_txt_files = download_companies(tickers)
    if not new_txt_files:
        log.info("Downloader produced no new files (may already be current)")
        print("\nSummary: 0 new transcripts added, 0 new chunks embedded")
        return

    # Chunk
    new_chunks = chunk_new_files(new_txt_files)

    # Embed
    new_chunk_count = embed_new_chunks(new_chunks)

    # Report
    log.info("=" * 60)
    log.info(
        "Update complete: %d new transcript(s), %d new chunk(s) embedded",
        len(new_txt_files), new_chunk_count,
    )
    log.info("=" * 60)
    print(
        f"\nSummary: {len(new_txt_files)} new transcript(s) added, "
        f"{new_chunk_count} new chunk(s) embedded"
    )


if __name__ == "__main__":
    main()
