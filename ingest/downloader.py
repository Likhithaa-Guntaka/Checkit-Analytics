"""
ingest/downloader.py — Earnings call transcript downloader for Checkit Analytics.

Retrieval strategy (three-tier fallback per company):
  1. sec-edgar-downloader  — fetches 8-K filings to a temp directory.
  2. EDGAR submissions API — walks Item-2.02 / 7.01 filings directly via requests.
  3. Motley Fool           — attempts to upgrade press releases to full transcripts.

Most large-cap companies do NOT file call transcripts with the SEC; they only
file an earnings press release (EX-99.1).  We save that as useful RAG content
and mark the metadata content_type accordingly.  If Motley Fool has the full
transcript for the same period we overwrite the press release with the transcript.

Usage:
    python ingest/downloader.py            # download all companies
    python ingest/downloader.py NVDA AAPL  # download specific tickers only
"""

import json
import re
import sys
import tempfile
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
load_dotenv()


# ── Constants ─────────────────────────────────────────────────────────────────

TRANSCRIPTS_DIR = Path(__file__).resolve().parent.parent / "data" / "transcripts"
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

START_DATE  = "2023-01-01"
END_DATE    = "2025-04-01"
MAX_FILINGS = 12   # per company — buffer above 8 quarters

EDGAR_BASE = "https://www.sec.gov"
EDGAR_DATA = "https://data.sec.gov"

EDGAR_HEADERS: dict[str, str] = {
    "User-Agent": "CheckitAnalytics research@checkit.ai",
    "Accept-Encoding": "gzip, deflate",
}
BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
}

# Any one of these in the text → almost certainly a full earnings call transcript
TRANSCRIPT_STRONG: list[str] = [
    "prepared remarks", "question-and-answer", "q&a session",
]
# These keywords appear in press releases but not in random 8-Ks
EARNINGS_KEYWORDS: list[str] = [
    "revenue", "net income", "earnings per share", "operating income",
    "gross margin", "guidance", "fiscal", "quarter",
]
EARNINGS_THRESHOLD = 4   # minimum keyword hits to qualify as earnings content

# 8-K item numbers that signal an earnings announcement
EARNINGS_ITEMS: set[str] = {"2.02", "7.01"}

# All quarters to attempt for Motley Fool transcript search
ALL_QUARTERS: list[tuple[str, str]] = [
    ("Q1", "2023"), ("Q2", "2023"), ("Q3", "2023"), ("Q4", "2023"),
    ("Q1", "2024"), ("Q2", "2024"), ("Q3", "2024"), ("Q4", "2024"),
    ("Q1", "2025"),
]

COMPANIES: list[dict[str, str]] = [
    {"name": "NVIDIA Corporation",                "ticker": "NVDA", "cik": "0001045810"},
    {"name": "Apple Inc.",                        "ticker": "AAPL", "cik": "0000320193"},
    {"name": "Microsoft Corporation",             "ticker": "MSFT", "cik": "0000789019"},
    {"name": "Tesla Inc.",                        "ticker": "TSLA", "cik": "0001318605"},
    {"name": "Meta Platforms Inc.",               "ticker": "META", "cik": "0001326801"},
    {"name": "Palantir Technologies Inc.",        "ticker": "PLTR", "cik": "0001321655"},
    {"name": "Snowflake Inc.",                    "ticker": "SNOW", "cik": "0001640147"},
    {"name": "Shopify Inc.",                      "ticker": "SHOP", "cik": "0001594805"},
    {"name": "Datadog Inc.",                      "ticker": "DDOG", "cik": "0001561680"},
    {"name": "CrowdStrike Holdings Inc.",         "ticker": "CRWD", "cik": "0001535527"},
    {"name": "Uber Technologies Inc.",            "ticker": "UBER", "cik": "0001543151"},
    {"name": "Airbnb Inc.",                       "ticker": "ABNB", "cik": "0001559720"},
    {"name": "ServiceNow Inc.",                   "ticker": "NOW",  "cik": "0001373715"},
    {"name": "Workday Inc.",                      "ticker": "WDAY", "cik": "0001327811"},
    {"name": "Eli Lilly and Company",             "ticker": "LLY",  "cik": "0000059478"},
    {"name": "Booking Holdings Inc.",             "ticker": "BKNG", "cik": "0001075531"},
    {"name": "Intuitive Surgical Inc.",           "ticker": "ISRG", "cik": "0001035267"},
    {"name": "Veeva Systems Inc.",                "ticker": "VEEV", "cik": "0001372514"},
    {"name": "Cloudflare Inc.",                   "ticker": "NET",  "cik": "0001477333"},
    {"name": "MongoDB Inc.",                      "ticker": "MDB",  "cik": "0001441816"},
    {"name": "JPMorgan Chase & Co.",              "ticker": "JPM",  "cik": "0000019617"},
    {"name": "Goldman Sachs Group Inc.",          "ticker": "GS",   "cik": "0000886982"},
    {"name": "Morgan Stanley",                    "ticker": "MS",   "cik": "0000895421"},
    {"name": "Bank of America Corp.",             "ticker": "BAC",  "cik": "0000070858"},
    {"name": "Wells Fargo & Company",             "ticker": "WFC",  "cik": "0000072971"},
    {"name": "Salesforce Inc.",                   "ticker": "CRM",  "cik": "0001108524"},
    {"name": "Zoom Video Communications Inc.",    "ticker": "ZM",   "cik": "0001585521"},
    {"name": "HubSpot Inc.",                      "ticker": "HUBS", "cik": "0001404655"},
    {"name": "Amazon.com Inc.",                   "ticker": "AMZN", "cik": "0001018724"},
    {"name": "Nike Inc.",                         "ticker": "NKE",  "cik": "0000320187"},
    {"name": "Starbucks Corporation",             "ticker": "SBUX", "cik": "0000829224"},
    {"name": "UnitedHealth Group Inc.",           "ticker": "UNH",  "cik": "0000731766"},
    {"name": "Pfizer Inc.",                       "ticker": "PFE",  "cik": "0000078003"},
    {"name": "Johnson & Johnson",                 "ticker": "JNJ",  "cik": "0000200406"},
    {"name": "NextEra Energy Inc.",               "ticker": "NEE",  "cik": "0000753308"},
    {"name": "Enphase Energy Inc.",               "ticker": "ENPH", "cik": "0001463101"},
    {"name": "First Solar Inc.",                  "ticker": "FSLR", "cik": "0001274494"},
    {"name": "Rivian Automotive Inc.",            "ticker": "RIVN", "cik": "0001874178"},
    {"name": "Lucid Group Inc.",                  "ticker": "LCID", "cik": "0001811210"},
    {"name": "Plug Power Inc.",                   "ticker": "PLUG", "cik": "0000887936"},
    {"name": "Bloom Energy Corp.",                "ticker": "BE",   "cik": "0001368514"},
    {"name": "Sunrun Inc.",                       "ticker": "RUN",  "cik": "0001469367"},
    {"name": "ChargePoint Holdings Inc.",         "ticker": "CHPT", "cik": "0001777393"},
    {"name": "Aptiv PLC",                         "ticker": "APTV", "cik": "0001521332"},
]

# Primary Motley Fool URL slug per ticker
_FOOL_SLUGS: dict[str, str] = {
    "NVDA": "nvidia",
    "AAPL": "apple",
    "MSFT": "microsoft",
    "TSLA": "tesla",
    "META": "meta-platforms",
    "PLTR": "palantir-technologies",
    "SNOW": "snowflake",
    "SHOP": "shopify",
    "DDOG": "datadog",
    "CRWD": "crowdstrike-holdings",
    "UBER": "uber-technologies",
    "ABNB": "airbnb",
    "NOW":  "servicenow",
    "WDAY": "workday",
    "LLY":  "eli-lilly",
    "BKNG": "booking-holdings",
    "ISRG": "intuitive-surgical",
    "VEEV": "veeva-systems",
    "NET":  "cloudflare",
    "MDB":  "mongodb",
    "JPM":  "jpmorgan-chase",
    "GS":   "goldman-sachs-group",
    "MS":   "morgan-stanley",
    "BAC":  "bank-of-america",
    "WFC":  "wells-fargo",
    "CRM":  "salesforce",
    "ZM":   "zoom-video-communications",
    "HUBS": "hubspot",
    "AMZN": "amazon",
    "NKE":  "nike",
    "SBUX": "starbucks",
    "UNH":  "unitedhealth-group",
    "PFE":  "pfizer",
    "JNJ":  "johnson-johnson",
    "NEE":  "nextera-energy",
    "ENPH": "enphase-energy",
    "FSLR": "first-solar",
    "RIVN": "rivian-automotive",
    "LCID": "lucid-group",
    "PLUG": "plug-power",
    "BE":   "bloom-energy",
    "RUN":  "sunrun",
    "CHPT": "chargepoint-holdings",
    "APTV": "aptiv",
}

# Additional slug variants per ticker — Motley Fool is inconsistent for financials
_EXTRA_FOOL_SLUGS: dict[str, list[str]] = {
    "JPM": ["jpmorgan", "jp-morgan-chase"],
    "GS":  ["goldman-sachs"],
    "MS":  ["morgan-stanley-financial"],
    "BAC": ["bank-america"],
}

# Tickers whose fiscal year is offset from the calendar year
_FISCAL_YEAR_OFFSET: dict[str, int] = {
    "NVDA": 1,   # FY2025 reported in calendar 2024
}


# ── Logging ───────────────────────────────────────────────────────────────────

def log(level: str, msg: str) -> None:
    """Print a structured log line prefixed with [INFO], [WARN], or [ERROR]."""
    print(f"[{level}] {msg}")


# ── Generic helpers ───────────────────────────────────────────────────────────

def fetch(
    url: str,
    headers: Optional[dict[str, str]] = None,
    retries: int = 3,
) -> Optional[str]:
    """GET a URL and return the response text, or None on failure."""
    h = headers or EDGAR_HEADERS
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=h, timeout=30)
            if r.status_code == 200:
                return r.text
            if r.status_code == 429:
                wait = 2 ** (attempt + 1)
                log("WARN", f"rate-limited — sleeping {wait}s")
                time.sleep(wait)
            else:
                break
        except requests.RequestException as exc:
            if attempt == retries - 1:
                log("ERROR", f"fetch failed for {url}: {exc}")
    return None


def html_to_text(html: str) -> str:
    """Strip HTML tags and return clean plain text."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def is_transcript(text: str) -> bool:
    """Return True if text contains strong signals of a full earnings call transcript."""
    lower = text.lower()
    return any(kw in lower for kw in TRANSCRIPT_STRONG)


def is_earnings_content(text: str) -> bool:
    """Return True if text has enough earnings keywords to qualify as RAG content."""
    lower = text.lower()
    hits = sum(1 for kw in EARNINGS_KEYWORDS if kw in lower)
    return hits >= EARNINGS_THRESHOLD


def filing_date_to_quarter(date_str: str) -> tuple[str, str]:
    """Map an EDGAR filing date string to a (quarter, year) tuple.

    Jan–Mar → Q4 of previous year  |  Apr–Jun → Q1
    Jul–Sep → Q2                   |  Oct–Dec → Q3
    """
    d = datetime.strptime(date_str, "%Y-%m-%d")
    if d.month <= 3:
        return "Q4", str(d.year - 1)
    if d.month <= 6:
        return "Q1", str(d.year)
    if d.month <= 9:
        return "Q2", str(d.year)
    return "Q3", str(d.year)


def save_content(
    text: str,
    company: dict[str, str],
    quarter: str,
    year: str,
    filing_date: str,
    source_url: str,
    content_type: str = "transcript",
) -> Path:
    """Write transcript text and a sidecar metadata JSON file to TRANSCRIPTS_DIR."""
    base      = f"{company['ticker']}_{quarter}_{year}"
    txt_path  = TRANSCRIPTS_DIR / f"{base}.txt"
    meta_path = TRANSCRIPTS_DIR / f"{base}_meta.json"

    txt_path.write_text(text, encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "company":      company["name"],
                "ticker":       company["ticker"],
                "cik":          company["cik"],
                "quarter":      quarter,
                "year":         year,
                "filing_date":  filing_date,
                "source_url":   source_url,
                "content_type": content_type,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return txt_path


# ── EDGAR filing helpers ──────────────────────────────────────────────────────

def get_8k_filings(cik: str) -> list[dict]:
    """Fetch Item-2.02/7.01 8-K filings from the EDGAR submissions API."""
    padded  = cik.lstrip("0").zfill(10)
    url     = f"{EDGAR_DATA}/submissions/CIK{padded}.json"
    content = fetch(url)
    if not content:
        return []

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []

    recent     = data.get("filings", {}).get("recent", {})
    forms      = recent.get("form", [])
    dates      = recent.get("filingDate", [])
    accs       = recent.get("accessionNumber", [])
    items_list = recent.get("items", [])

    result: list[dict] = []
    for form, date, acc, items_str in zip(forms, dates, accs, items_list):
        if form not in ("8-K", "8-K/A"):
            continue
        if not (START_DATE <= date <= END_DATE):
            continue
        items = {i.strip() for i in str(items_str).split(",")}
        if items & EARNINGS_ITEMS:
            result.append({"accession": acc, "date": date, "items": items})

    return result[:MAX_FILINGS]


def get_filing_exhibits(cik: str, accession: str) -> list[dict]:
    """Parse the EDGAR filing index HTML and return a list of exhibit documents."""
    cik_num    = str(int(cik.lstrip("0") or "0"))
    acc_nodash = accession.replace("-", "")
    idx_url    = (
        f"{EDGAR_BASE}/Archives/edgar/data/{cik_num}/{acc_nodash}/{accession}-index.htm"
    )
    content = fetch(idx_url)
    if not content:
        return []

    soup = BeautifulSoup(content, "lxml")
    docs: list[dict] = []

    for table in soup.select("table.tableFile"):
        for row in table.select("tr")[1:]:
            cells = row.find_all("td")
            if len(cells) < 4:
                continue
            link  = cells[2].find("a")
            dtype = cells[3].get_text(strip=True)
            if not link:
                continue

            href = link.get("href", "")
            # Unwrap iXBRL viewer links: /ix?doc=/Archives/...
            if href.startswith("/ix?doc="):
                href = href[len("/ix?doc="):]
            full_url = EDGAR_BASE + href if href.startswith("/") else href

            docs.append({
                "name": link.get_text(strip=True),
                "href": href,
                "type": dtype,
                "url":  full_url,
            })

    return docs


def fetch_best_exhibit(docs: list[dict]) -> tuple[Optional[str], str, str]:
    """Download EX-99 exhibits in priority order and return (text, url, content_type).

    Returns (None, "", "") when no qualifying earnings document is found.
    """
    def _priority(doc: dict) -> int:
        t = doc["type"].lower()
        n = doc["name"].lower()
        if "ex-99" in t or "ex99" in n:
            return 0
        if "8-k" in t:
            return 1
        return 2

    for doc in sorted(docs, key=_priority):
        name = doc["name"].lower()
        if not any(name.endswith(ext) for ext in (".htm", ".html", ".txt")):
            continue
        if any(name.endswith(x) for x in (".xsd", "_lab.xml", "_pre.xml", "_htm.xml")):
            continue

        raw = fetch(doc["url"])
        time.sleep(0.5)
        if not raw:
            continue

        text = html_to_text(raw) if name.endswith((".htm", ".html")) else raw

        if is_transcript(text):
            return text, doc["url"], "transcript"
        if is_earnings_content(text):
            return text, doc["url"], "earnings_release"

    return None, "", ""


# ── Method 1: sec-edgar-downloader ───────────────────────────────────────────

def try_sec_edgar_downloader(
    company: dict[str, str],
    seen: set[str],
) -> list[str]:
    """Download 8-Ks via sec-edgar-downloader and extract earnings content."""
    try:
        from sec_edgar_downloader import Downloader  # noqa: PLC0415
    except ImportError:
        return []

    saved: list[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        dl = Downloader("CheckitAnalytics", "research@checkit.ai", tmpdir)
        try:
            dl.get(
                "8-K",
                company["ticker"],
                limit=MAX_FILINGS,
                after=START_DATE,
                before=END_DATE,
            )
        except Exception as exc:
            log("WARN", f"sec-edgar-downloader failed for {company['ticker']}: {exc}")
            return []

        base = Path(tmpdir) / "sec_edgar_filings" / company["ticker"] / "8-K"
        if not base.exists():
            return []

        # Cross-reference with submissions API to get filing dates
        filings  = get_8k_filings(company["cik"])
        date_map: dict[str, str] = {}
        for f in filings:
            date_map[f["accession"].replace("-", "")] = f["date"]
            date_map[f["accession"]]                  = f["date"]

        for acc_dir in sorted(base.iterdir(), reverse=True):
            if not acc_dir.is_dir():
                continue

            filing_date = date_map.get(acc_dir.name) or date_map.get(
                acc_dir.name.replace("-", "")
            )
            if not filing_date:
                continue

            quarter, year = filing_date_to_quarter(filing_date)
            key = f"{quarter}_{year}"
            if key in seen:
                continue

            candidates = (
                sorted(acc_dir.glob("*.htm*"))
                + sorted(acc_dir.glob("*.html"))
                + sorted(acc_dir.glob("*.txt"))
            )
            for cand in candidates:
                if any(cand.name.lower().endswith(x) for x in (".xsd", ".xml")):
                    continue
                try:
                    raw  = cand.read_text(encoding="utf-8", errors="ignore")
                    text = (
                        html_to_text(raw) if cand.suffix in (".htm", ".html") else raw
                    )
                    ctype = (
                        "transcript"      if is_transcript(text) else
                        "earnings_release" if is_earnings_content(text) else
                        None
                    )
                    if ctype:
                        cik_num    = str(int(company["cik"].lstrip("0") or "0"))
                        acc_nodash = acc_dir.name.replace("-", "")
                        source_url = (
                            f"{EDGAR_BASE}/Archives/edgar/data/{cik_num}/"
                            f"{acc_nodash}/{cand.name}"
                        )
                        path = save_content(
                            text, company, quarter, year,
                            filing_date, source_url, ctype,
                        )
                        seen.add(key)
                        saved.append(path.name)
                        log("INFO", f"[edgar-dl] saved {path.name} ({ctype})")
                        break
                except (OSError, UnicodeDecodeError):
                    continue

    return saved


# ── Method 2: requests + EDGAR submissions API ────────────────────────────────

def try_requests_edgar(
    company: dict[str, str],
    seen: set[str],
) -> list[str]:
    """Walk EDGAR submissions API and save earnings content from Item-2.02 8-Ks."""
    saved: list[str] = []
    filings = get_8k_filings(company["cik"])
    time.sleep(0.5)

    for filing in filings:
        quarter, year = filing_date_to_quarter(filing["date"])
        key = f"{quarter}_{year}"
        if key in seen:
            continue

        log("INFO", f"[requests] {filing['accession']} ({filing['date']}) → {quarter} {year}")

        docs = get_filing_exhibits(company["cik"], filing["accession"])
        time.sleep(0.5)

        if not docs:
            log("WARN", "no documents found in filing index")
            continue

        text, source_url, ctype = fetch_best_exhibit(docs)

        if text:
            path = save_content(
                text, company, quarter, year,
                filing["date"], source_url, ctype,
            )
            seen.add(key)
            saved.append(path.name)
            log("INFO", f"[requests] saved {path.name} ({ctype})")
        else:
            log("WARN", "no earnings content detected in filing")

    return saved


# ── Method 3: Motley Fool transcript fetcher ──────────────────────────────────

def _build_fool_url_candidates(
    ticker: str,
    quarter: str,
    year: str,
    filing_date: str,
) -> list[str]:
    """Build a list of candidate Motley Fool transcript URLs to probe."""
    slug  = _FOOL_SLUGS.get(ticker, ticker.lower())
    t     = ticker.lower()
    qnum  = quarter[1]   # "1" from "Q1"
    base  = "https://www.fool.com/earnings/call-transcripts"

    offset = _FISCAL_YEAR_OFFSET.get(ticker, 0)
    fyear  = str(int(year) + offset)

    try:
        fd = datetime.strptime(filing_date, "%Y-%m-%d")
        date_variants = [f"{fd.year}/{fd.month:02d}/{fd.day:02d}"]
        for delta in (-1, 1, -2, 2):
            d2 = fd + timedelta(days=delta)
            date_variants.append(f"{d2.year}/{d2.month:02d}/{d2.day:02d}")
    except ValueError:
        date_variants = []

    all_slugs = [slug] + _EXTRA_FOOL_SLUGS.get(ticker, [])
    slug_variants: list[str] = []
    for s in all_slugs:
        slug_variants += [
            f"{s}-{t}-q{qnum}-{fyear}-earnings-call-transcript",
            f"{s}-{t}-q{qnum}-{year}-earnings-call-transcript",
            f"{t}-q{qnum}-{fyear}-earnings-call-transcript",
            f"{s}-{t}-q{qnum}-fy{fyear}-earnings-call-transcript",
        ]

    return [
        f"{base}/{date_part}/{sv}/"
        for date_part in date_variants
        for sv in slug_variants
    ]


def _fool_search_fallback(company: dict[str, str], quarter: str, year: str) -> Optional[str]:
    """Search Motley Fool for a transcript when direct URL probing fails."""
    ticker = company["ticker"]
    slug   = _FOOL_SLUGS.get(ticker, ticker.lower())
    t      = ticker.lower()

    search_url = (
        f"https://www.fool.com/search/#q={ticker}+{quarter}+{year}"
        "+earnings+call+transcript&facet=articles&filter=transcript"
    )
    content = fetch(search_url, headers=BROWSER_HEADERS)
    if not content:
        return None

    soup  = BeautifulSoup(content, "lxml")
    links = soup.find_all("a", href=re.compile(r"/earnings/call-transcripts/"))

    for link in links[:6]:
        href = link.get("href", "").lower()
        if t not in href and slug.split("-")[0] not in href:
            continue
        url  = (
            "https://www.fool.com" + link["href"]
            if link["href"].startswith("/")
            else link["href"]
        )
        page = fetch(url, headers=BROWSER_HEADERS)
        if page:
            time.sleep(0.5)
            txt = html_to_text(page)
            if is_transcript(txt) and len(txt) > 5000:
                return txt

    return None


def try_motley_fool(
    company: dict[str, str],
    quarter: str,
    year: str,
    filing_date: str,
) -> Optional[str]:
    """Probe Motley Fool transcript URLs and return plain text if a transcript is found."""
    ticker     = company["ticker"]
    candidates = _build_fool_url_candidates(ticker, quarter, year, filing_date)

    for url in candidates:
        page = fetch(url, headers=BROWSER_HEADERS)
        if not page:
            time.sleep(0.3)
            continue
        if "earnings-call-transcript" not in page.lower():
            time.sleep(0.3)
            continue
        text = html_to_text(page)
        if is_transcript(text) and len(text) > 5000:
            return text
        time.sleep(0.3)

    return _fool_search_fallback(company, quarter, year)


def upgrade_to_transcript(
    company: dict[str, str],
    seen: set[str],
) -> list[str]:
    """Replace saved press releases with full Motley Fool transcripts where available."""
    upgraded: list[str] = []
    ticker = company["ticker"]

    # Quarters saved as press releases are upgrade candidates
    quarters_to_try: list[tuple[str, str, str]] = []
    for meta_path in sorted(TRANSCRIPTS_DIR.glob(f"{ticker}_*_meta.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if meta.get("content_type", "transcript") != "transcript":
            quarters_to_try.append(
                (meta.get("quarter", ""), meta.get("year", ""), meta.get("filing_date", ""))
            )

    # Also try quarters not yet saved at all
    saved_keys = {f"{q}_{y}" for q, y, _ in quarters_to_try} | seen
    for q, y in ALL_QUARTERS:
        if f"{q}_{y}" not in saved_keys:
            quarters_to_try.append((q, y, ""))

    for quarter, year, filing_date in quarters_to_try:
        key = f"{quarter}_{year}"
        log("INFO", f"[motley-fool] searching {ticker} {quarter} {year}")
        text = try_motley_fool(company, quarter, year, filing_date)
        if text:
            path = save_content(
                text, company, quarter, year,
                filing_date=filing_date or f"{year}-XX-XX",
                source_url="https://www.fool.com/earnings/call-transcripts/",
                content_type="transcript",
            )
            seen.add(key)
            upgraded.append(path.name)
            log("INFO", f"[motley-fool] saved transcript {path.name}")
        time.sleep(0.5)

    return upgraded


# ── Per-company orchestration ─────────────────────────────────────────────────

def download_company(company: dict[str, str]) -> list[str]:
    """Run all three download methods for one company and return saved filenames."""
    ticker = company["ticker"]
    log("INFO", f"{'─' * 56}")
    log("INFO", f"{company['name']}  ({ticker} / CIK {company['cik']})")
    log("INFO", f"{'─' * 56}")

    seen:  set[str]  = set()
    saved: list[str] = []

    log("INFO", "→ Method 1: sec-edgar-downloader")
    saved.extend(try_sec_edgar_downloader(company, seen))

    log("INFO", "→ Method 2: EDGAR submissions API")
    saved.extend(try_requests_edgar(company, seen))

    log("INFO", "→ Method 3: Motley Fool transcript search")
    saved.extend(upgrade_to_transcript(company, seen))

    if saved:
        unique = sorted(set(saved))
        log("INFO", f"✓ {ticker}: {len(unique)} file(s) saved")
        return unique

    log("WARN", f"✗ {ticker}: nothing saved — check logs above")
    return []


# ── Shared helpers for entry points ──────────────────────────────────────────

def _print_run_header(companies: list[dict[str, str]], label: str) -> None:
    """Print the startup banner for a download run."""
    print("=" * 60)
    print(f"  Checkit RAG — {label}")
    print(f"  Companies  : {', '.join(c['ticker'] for c in companies)}"
          if len(companies) <= 10
          else f"  Companies  : {len(companies)}")
    print(f"  Date range : {START_DATE}  →  {END_DATE}")
    print("=" * 60)


def _run_companies(companies: list[dict[str, str]]) -> dict[str, int]:
    """Download all companies in the list and return a ticker → file-count summary."""
    summary: dict[str, int] = {}
    for company in companies:
        try:
            files = download_company(company)
            summary[company["ticker"]] = len(files)
        except Exception as exc:
            log("ERROR", f"skipping {company['name']}: {exc}")
            summary[company["ticker"]] = 0
    return summary


def _print_summary_table(companies: list[dict[str, str]], summary: dict[str, int]) -> None:
    """Print a formatted summary table of files saved per company."""
    total = sum(summary.values())
    print("\n" + "=" * 60)
    print(f"  {'Company':<35} {'Saved':>6}")
    print("  " + "─" * 42)
    for company in companies:
        count = summary.get(company["ticker"], 0)
        print(f"  {company['name']:<35} {count:>6}")
    print("  " + "─" * 42)
    print(f"  {'TOTAL':<35} {total:>6}")
    print("=" * 60)
    print(f"\nDownload complete: {total} transcripts saved")


# ── Public API (used by pipeline.py) ─────────────────────────────────────────

def list_transcripts() -> list[Path]:
    """Return all .txt files in data/transcripts/ (excludes _meta.json companions)."""
    return sorted(TRANSCRIPTS_DIR.glob("*.txt"))


def download_url(url: str, filename: str) -> Path:
    """Download a single URL, convert HTML to plain text, and save to TRANSCRIPTS_DIR."""
    content = fetch(url, headers=BROWSER_HEADERS)
    if not content:
        raise RuntimeError(f"Failed to fetch {url}")
    text = html_to_text(content)
    out  = TRANSCRIPTS_DIR / filename
    out.write_text(text, encoding="utf-8")
    log("INFO", f"saved: {out}")
    return out


# ── Entry points ──────────────────────────────────────────────────────────────

def main() -> None:
    """Download earnings content for all companies in COMPANIES."""
    _print_run_header(COMPANIES, "SEC EDGAR Transcript Downloader")
    summary = _run_companies(COMPANIES)
    _print_summary_table(COMPANIES, summary)


def download_specific_companies(tickers: list[str]) -> None:
    """Download earnings content for the subset of COMPANIES matching *tickers*."""
    ticker_set = {t.upper() for t in tickers}
    targets    = [c for c in COMPANIES if c["ticker"] in ticker_set]
    unknown    = ticker_set - {c["ticker"] for c in targets}

    if unknown:
        log("WARN", f"unknown tickers (not in COMPANIES): {', '.join(sorted(unknown))}")
    if not targets:
        log("ERROR", "no matching companies found — exiting")
        return

    _print_run_header(targets, "Targeted Download")
    summary = _run_companies(targets)
    _print_summary_table(targets, summary)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        download_specific_companies(sys.argv[1:])
    else:
        main()
