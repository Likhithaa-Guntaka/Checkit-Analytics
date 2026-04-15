"""
ingest/chunker.py — Speaker-aware chunking of earnings call transcripts and press releases.

Transcript files (content_type: "transcript"):
  Parsed from Motley Fool format.  Navigation boilerplate is skipped by anchoring
  on the "Prepared Remarks:" heading.  Each speaker turn is identified by a
  two-line header (Name / -- / Role) or a standalone "Operator" line.
  The Q&A section is split from prepared remarks at the "Questions & Answers:"
  boundary and tagged separately so downstream filters can target either section.

Press release files (content_type: "earnings_release"):
  No speaker turns.  Text is chunked via a sliding paragraph window after
  stripping EDGAR SGML header artifacts.

Chunking constraints (applied to both types):
  MIN_WORDS  = 40   — turns shorter than this are dropped.
  MAX_WORDS  = 350  — turns longer than this are split at sentence boundaries,
                      with OVERLAP_WORDS carried forward for continuity.
  A hard word-window fallback handles sentences that cannot be split (e.g.
  concatenated financial tables).

All chunks are saved to data/chunks.json.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

TRANSCRIPTS_DIR = Path(__file__).resolve().parent.parent / "data" / "transcripts"
CHUNKS_FILE     = Path(__file__).resolve().parent.parent / "data" / "chunks.json"

# ── Chunk size limits ─────────────────────────────────────────────────────────

MIN_WORDS      = 40
MAX_WORDS      = 350
OVERLAP_WORDS  = 40
MAX_FILE_BYTES = 300_000   # files above this size are likely bad downloads

# ── Compiled regex patterns ───────────────────────────────────────────────────

# Marks the start of the prepared-remarks section in Motley Fool transcripts
PREPARED_RE = re.compile(r"Prepared Remarks?:\s*\n", re.IGNORECASE)

# Marks the Q&A section boundary
QA_RE = re.compile(
    r"Questions?\s*(?:and|&)\s*Answers?[:\.]?\s*\n"
    r"|Question-and-Answer\s+Session",
    re.IGNORECASE,
)

# Marks the call-participants listing (sometimes leaks into the last speaker turn)
PARTICIPANTS_RE = re.compile(r"Call Participants?:\s*\n", re.IGNORECASE)

# Splits text at sentence boundaries (after . ! ?)
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

# Matches EDGAR SGML artifact lines at the top of press release text files
_SGML_SKIP_RE = re.compile(
    r"^(?:EX-99\.[12]|Document|Exhibit\s+99\.[12]|\d{1,2}|[a-z0-9\-]+\.htm[l]?)$",
    re.IGNORECASE,
)

# ── Known investment-bank names (used to classify analyst speakers) ───────────

_BANKS: set[str] = {
    "goldman sachs", "morgan stanley", "jp morgan", "jpmorgan", "j.p. morgan",
    "bank of america", "citigroup", "citi", "ubs", "credit suisse",
    "barclays", "deutsche bank", "jefferies", "cowen", "piper sandler",
    "cantor fitzgerald", "bernstein", "sanford bernstein", "wells fargo",
    "rbc", "rbc capital", "stifel", "raymond james", "needham", "oppenheimer",
    "baird", "william blair", "keybanc", "truist", "mizuho", "daiwa",
    "nomura", "hsbc", "evercore", "guggenheim", "rosenblatt", "wedbush",
    "da davidson", "d.a. davidson", "btig", "atlantic equities", "wolfe research",
    "loop capital", "melius research", "redburn", "atlantic", "moffettnathanson",
}


# ── Speaker classification ────────────────────────────────────────────────────

def classify_speaker(speaker: str, role: str) -> tuple[str, str]:
    """Return (speaker_type, speaker_role) for a given name and title string.

    speaker_type : "management" | "analyst" | "operator"
    speaker_role : normalised title — "CEO", "CFO", "COO", "CRO", "CTO", "CPO",
                   "President", "Analyst", "IR", "VP", "Operator", or the first
                   comma-segment of the raw title when no pattern matches.
    """
    if speaker.strip() == "Operator":
        return "operator", "Operator"

    rl = role.lower()

    if "chief executive" in rl or re.search(r"\bceo\b", rl):
        sp_role = "CEO"
    elif "chief financial" in rl or re.search(r"\bcfo\b", rl):
        sp_role = "CFO"
    elif "chief operating" in rl or re.search(r"\bcoo\b", rl):
        sp_role = "COO"
    elif "chief revenue" in rl or re.search(r"\bcro\b", rl):
        sp_role = "CRO"
    elif "chief technology" in rl or re.search(r"\bcto\b", rl):
        sp_role = "CTO"
    elif "chief product" in rl or re.search(r"\bcpo\b", rl):
        sp_role = "CPO"
    elif "president" in rl:
        sp_role = "President"
    elif "analyst" in rl:
        sp_role = "Analyst"
    elif "investor relations" in rl:
        sp_role = "IR"
    elif "general counsel" in rl:
        sp_role = "General Counsel"
    elif "vice president" in rl or re.search(r"\bvp\b", rl):
        sp_role = "VP"
    elif role.strip():
        sp_role = role.split(",")[0].strip()
    else:
        sp_role = "Unknown"

    rl_clean = re.sub(r"[^a-z\s]", " ", rl)
    if "analyst" in rl or any(bank in rl_clean for bank in _BANKS):
        sp_type = "analyst"
    else:
        sp_type = "management"

    return sp_type, sp_role


# ── Text splitting ────────────────────────────────────────────────────────────

def split_long_text(text: str) -> list[str]:
    """Split text exceeding MAX_WORDS at sentence boundaries with OVERLAP_WORDS carry-forward."""
    sentences = SENTENCE_END_RE.split(text.strip())
    if not sentences:
        return []

    blocks: list[str] = []
    current: list[str] = []
    current_wc = 0

    for sent in sentences:
        sw = len(sent.split())
        if current_wc + sw <= MAX_WORDS:
            current.append(sent)
            current_wc += sw
        else:
            if current:
                blocks.append(" ".join(current))
            overlap: list[str] = []
            ow = 0
            for s in reversed(current):
                sw2 = len(s.split())
                if ow + sw2 <= OVERLAP_WORDS:
                    overlap.insert(0, s)
                    ow += sw2
                else:
                    break
            current    = overlap + [sent]
            current_wc = sum(len(s.split()) for s in current)

    if current:
        blocks.append(" ".join(current))

    return [b for b in blocks if b.strip()]


# ── Transcript parsing ────────────────────────────────────────────────────────

def _parse_speaker_turns(section_text: str, section: str) -> list[dict]:
    """Extract individual speaker turns from one section of a transcript.

    Detects the Motley Fool two-line header (Name / -- / Role) and the
    standalone "Operator" marker.  Returns a list of turn dicts with keys:
    speaker, role, speaker_type, speaker_role, section, text.
    """
    lines = section_text.split("\n")
    turns: list[dict] = []

    current_speaker: str | None = None
    current_role:    str        = ""
    current_lines:   list[str]  = []

    def flush() -> None:
        """Emit the accumulated speaker turn and reset state."""
        nonlocal current_speaker, current_role, current_lines
        if not current_speaker:
            return
        content = "\n".join(current_lines).strip()
        cp = PARTICIPANTS_RE.search(content)
        if cp:
            content = content[: cp.start()].strip()
        if content:
            sp_type, sp_role = classify_speaker(current_speaker, current_role)
            turns.append({
                "speaker":      current_speaker,
                "role":         current_role,
                "speaker_type": sp_type,
                "speaker_role": sp_role,
                "section":      section,
                "text":         content,
            })
        current_speaker = None
        current_role    = ""
        current_lines   = []

    i = 0
    n = len(lines)
    while i < n:
        stripped = lines[i].strip()

        if (
            stripped
            and i + 2 < n
            and lines[i + 1].strip() == "--"
            and len(stripped) <= 80
            and stripped[0].isupper()
            and stripped[-1] not in ".?!"
            and not stripped.startswith(("Prepared", "Questions", "Call", "["))
        ):
            flush()
            current_speaker = stripped
            current_role    = lines[i + 2].strip()
            current_lines   = []
            i += 3
            continue

        if stripped == "Operator":
            flush()
            current_speaker = "Operator"
            current_role    = "Operator"
            current_lines   = []
            i += 1
            continue

        if current_speaker and stripped:
            current_lines.append(stripped)

        i += 1

    flush()
    return turns


def parse_transcript(text: str) -> list[dict]:
    """Parse a Motley Fool earnings call transcript into a list of speaker-turn dicts.

    Returns an empty list if the text does not contain a "Prepared Remarks:" anchor.
    """
    prep_m = PREPARED_RE.search(text)
    if not prep_m:
        return []

    content = text[prep_m.start():]
    qa_m    = QA_RE.search(content)

    if qa_m:
        prepared_text = content[: qa_m.start()]
        qa_text       = content[qa_m.start():]
    else:
        prepared_text = content
        qa_text       = ""

    turns: list[dict] = []
    turns.extend(_parse_speaker_turns(prepared_text, "prepared_remarks"))
    if qa_text:
        turns.extend(_parse_speaker_turns(qa_text, "qa"))
    return turns


# ── Press release parsing ─────────────────────────────────────────────────────

def _strip_edgar_header(text: str) -> str:
    """Remove leading EDGAR SGML metadata lines from press release text."""
    lines = text.split("\n")
    start = 0
    for i, line in enumerate(lines[:15]):
        if _SGML_SKIP_RE.match(line.strip()):
            start = i + 1
        elif i > 0:
            break
    return "\n".join(lines[start:])


def parse_press_release(text: str) -> list[dict]:
    """Parse an earnings press release into sliding-window paragraph blocks.

    No speaker turns are extracted; every block is tagged section="earnings_release".
    Short or numeric-only lines (table cells) are skipped.
    """
    text      = _strip_edgar_header(text)
    raw_paras = re.split(r"\n{2,}|\n•\s*", text)

    blocks:    list[dict] = []
    buffer:    list[str]  = []
    buffer_wc: int        = 0

    def flush_buffer() -> None:
        """Emit the current paragraph buffer as a block dict and reset."""
        nonlocal buffer, buffer_wc
        if buffer:
            blocks.append({
                "speaker":      "",
                "role":         "",
                "speaker_type": "N/A",
                "speaker_role": "N/A",
                "section":      "earnings_release",
                "text":         " ".join(buffer),
            })
        buffer    = []
        buffer_wc = 0

    for para in raw_paras:
        para = para.strip()
        if not para or len(para) < 40 or re.match(r"^[\d\s\$\%\.\,\(\)\-]+$", para):
            continue
        pw = len(para.split())
        if buffer_wc + pw <= MAX_WORDS:
            buffer.append(para)
            buffer_wc += pw
        else:
            flush_buffer()
            buffer    = [para]
            buffer_wc = pw

    flush_buffer()
    return blocks


# ── Chunk builder ─────────────────────────────────────────────────────────────

def make_chunks(turns: list[dict], meta: dict, ticker: str) -> list[dict]:
    """Convert parsed speaker turns into structured chunk dicts with metadata.

    Enforces MIN_WORDS and MAX_WORDS per chunk.  Long turns are split at sentence
    boundaries; a hard word-window fallback handles unsplittable run-on text
    (e.g. financial tables concatenated into one string).
    """
    chunks: list[dict] = []
    idx = 0

    for turn in turns:
        text = turn["text"].strip()
        wc   = len(text.split())

        if wc < MIN_WORDS:
            continue

        blocks = [text] if wc <= MAX_WORDS else split_long_text(text)

        final_blocks: list[str] = []
        for blk in blocks:
            if len(blk.split()) <= MAX_WORDS:
                final_blocks.append(blk)
            else:
                bwords = blk.split()
                start  = 0
                while start < len(bwords):
                    final_blocks.append(" ".join(bwords[start : start + MAX_WORDS]))
                    start += MAX_WORDS - OVERLAP_WORDS
        blocks = final_blocks

        for block in blocks:
            bw = len(block.split())
            if bw < MIN_WORDS:
                continue
            chunk_id = f"{ticker}_{meta.get('quarter', '')}_{meta.get('year', '')}_{idx:03d}"
            chunks.append({
                "chunk_id":     chunk_id,
                "company":      meta.get("company", ""),
                "ticker":       ticker,
                "quarter":      meta.get("quarter", ""),
                "year":         meta.get("year", ""),
                "filing_date":  meta.get("filing_date", ""),
                "section":      turn["section"],
                "speaker":      turn["speaker"],
                "speaker_role": turn["speaker_role"],
                "speaker_type": turn["speaker_type"],
                "text":         block,
                "word_count":   bw,
                "char_count":   len(block),
            })
            idx += 1

    return chunks


# ── Per-file driver ───────────────────────────────────────────────────────────

def process_file(txt_path: Path) -> list[dict]:
    """Load one transcript or press-release file and return its chunks.

    Skips the file and returns [] if the sidecar _meta.json is missing,
    the file exceeds MAX_FILE_BYTES, or the metadata cannot be parsed.
    Falls back to press-release chunking if transcript parsing yields no turns.
    """
    meta_path = txt_path.with_name(txt_path.stem + "_meta.json")
    if not meta_path.exists():
        print(f"  [skip] no meta for {txt_path.name}")
        return []

    file_bytes = txt_path.stat().st_size
    if file_bytes > MAX_FILE_BYTES:
        print(f"  [skip] {txt_path.name} is {file_bytes // 1024} KB — too large")
        return []

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        text = txt_path.read_text(encoding="utf-8")
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [skip] could not read {txt_path.name}: {exc}")
        return []

    ticker       = meta.get("ticker", txt_path.stem.split("_")[0])
    content_type = meta.get("content_type", "earnings_release")

    if content_type == "transcript":
        turns = parse_transcript(text)
        if not turns:
            print(f"  [warn] no speaker turns in {txt_path.name}, using fallback chunker")
            turns = parse_press_release(text)
    else:
        turns = parse_press_release(text)

    return make_chunks(turns, meta, ticker)


# ── process_all() helpers ─────────────────────────────────────────────────────

def _find_transcript_files() -> list[Path]:
    """Return sorted .txt files in TRANSCRIPTS_DIR, excluding _meta companion files."""
    return sorted(
        p for p in TRANSCRIPTS_DIR.glob("*.txt")
        if not p.stem.endswith("_meta")
    )


def _save_chunks(chunks: list[dict]) -> None:
    """Serialise chunks to CHUNKS_FILE as JSON."""
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_FILE.write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _print_chunk_stats(chunks: list[dict], total_files: int, processed: int) -> None:
    """Print a breakdown of chunk counts by company and section."""
    total      = len(chunks)
    by_company: dict[str, int] = defaultdict(int)
    by_section: dict[str, int] = defaultdict(int)
    total_words = 0

    for c in chunks:
        by_company[c["ticker"]] += 1
        by_section[c["section"]] += 1
        total_words += c["word_count"]

    avg_words = total_words / total if total else 0.0

    print(f"\n{'=' * 55}")
    print(f"  Chunks saved → {CHUNKS_FILE.relative_to(CHUNKS_FILE.parent.parent)}")
    print(f"{'=' * 55}")
    print(f"\n  Total chunks created : {total:,}")
    print(f"  Files processed      : {processed} / {total_files}")
    print(f"  Average word count   : {avg_words:.1f}")

    print(f"\n  Breakdown by company")
    print(f"  {'Ticker':<8}  {'Chunks':>7}")
    print("  " + "-" * 18)
    for ticker, count in sorted(by_company.items()):
        print(f"  {ticker:<8}  {count:>7,}")

    print(f"\n  Breakdown by section")
    print(f"  {'Section':<25}  {'Chunks':>7}")
    print("  " + "-" * 35)
    for section, count in sorted(by_section.items()):
        print(f"  {section:<25}  {count:>7,}")


# ── Main entry point ──────────────────────────────────────────────────────────

def process_all() -> list[dict]:
    """Process every transcript in TRANSCRIPTS_DIR, save chunks.json, and print stats."""
    txt_files = _find_transcript_files()
    if not txt_files:
        print(f"No .txt files found in {TRANSCRIPTS_DIR}. Run the downloader first.")
        return []

    print(f"Processing {len(txt_files)} files …")

    all_chunks: list[dict] = []
    processed = 0
    for path in txt_files:
        chunks = process_file(path)
        all_chunks.extend(chunks)
        if chunks:
            processed += 1

    _save_chunks(all_chunks)
    _print_chunk_stats(all_chunks, total_files=len(txt_files), processed=processed)
    return all_chunks


# ── Backward-compat shims (used by pipeline.py / embedder.py) ────────────────

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Simple word-window chunker used by the pipeline.py fallback path."""
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        chunks.append(" ".join(words[start : start + chunk_size]))
        start += chunk_size - overlap
    return chunks


def chunk_file(file_path: Path, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """Chunk a single file with the simple word-window method."""
    text = file_path.read_text(encoding="utf-8")
    return [
        {"text": c, "source": file_path.name, "chunk_index": i}
        for i, c in enumerate(chunk_text(text, chunk_size, overlap))
    ]


def chunk_all_files(
    files: list[Path], chunk_size: int = 500, overlap: int = 50
) -> list[dict]:
    """Chunk all files with the simple word-window method."""
    all_chunks: list[dict] = []
    for f in files:
        all_chunks.extend(chunk_file(f, chunk_size, overlap))
    return all_chunks


if __name__ == "__main__":
    process_all()
