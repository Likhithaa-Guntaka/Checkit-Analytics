"""
pipeline.py — Full Checkit Analytics setup pipeline.

Runs every stage in order:
  1. Download transcripts  (ingest/downloader.py)
  2. Chunk transcripts     (ingest/chunker.py)
  3. Embed into ChromaDB   (ingest/embedder.py)
  4. Verify retrieval      (retrieval/searcher.py)
  5. End-to-end RAG test   (reasoning/answerer.py)

Usage:
    python pipeline.py
"""

import json
import sys
import traceback
from pathlib import Path

# Ensure the project root is on sys.path regardless of launch directory
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_TRANSCRIPTS_DIR = _ROOT / "data" / "transcripts"
_CHUNKS_FILE     = _ROOT / "data" / "chunks.json"

# ── Console helpers ────────────────────────────────────────────────────────────

WIDTH = 62


def _banner(step: int, total: int, title: str) -> None:
    print()
    print("═" * WIDTH)
    print(f"  STEP {step} / {total} — {title}")
    print("═" * WIDTH)


def _ok(msg: str) -> None:
    print(f"  ✓  {msg}")


def _info(msg: str) -> None:
    print(f"  →  {msg}")


def _warn(msg: str) -> None:
    print(f"  ⚠  {msg}")


def _err(msg: str) -> None:
    print(f"  ✗  {msg}")


def _ask_yn(prompt: str, default: str = "n") -> bool:
    """Prompt for y/n. Returns True for yes."""
    hint = " (y/n) "
    try:
        answer = input(f"  {prompt}{hint}").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        answer = default
    return answer in ("y", "yes")


def _ask_continue(label: str) -> None:
    """After a step error, ask whether to continue. Exits on 'n'."""
    if not _ask_yn(f"Step failed: {label}. Continue to next step anyway?"):
        print("\n  Pipeline aborted by user.")
        sys.exit(1)


# ── Step implementations ───────────────────────────────────────────────────────

def step_download() -> int:
    """Download transcripts. Returns number of .txt files afterwards."""
    _banner(1, 5, "Download Transcripts")

    txt_files = list(_TRANSCRIPTS_DIR.glob("*.txt")) if _TRANSCRIPTS_DIR.exists() else []
    existing  = len(txt_files)

    if existing > 0:
        _info(f"data/transcripts/ already contains {existing} .txt file(s).")
        if not _ask_yn("Re-download all transcripts? This may overwrite existing files."):
            _ok("Skipping download — using existing transcripts.")
            return existing

    try:
        from ingest.downloader import main as downloader_main  # noqa: PLC0415
        downloader_main()
    except Exception:
        _err("Downloader raised an exception:")
        traceback.print_exc()
        _ask_continue("Download Transcripts")

    # Count after (re-)download
    txt_files = list(_TRANSCRIPTS_DIR.glob("*.txt")) if _TRANSCRIPTS_DIR.exists() else []
    count     = len(txt_files)
    print()
    _ok(f"{count} .txt file(s) in data/transcripts/")
    return count


def step_chunk() -> int:
    """Chunk transcripts. Returns total chunks."""
    _banner(2, 5, "Chunk Transcripts")

    if _CHUNKS_FILE.exists():
        try:
            existing_count = len(json.loads(_CHUNKS_FILE.read_text(encoding="utf-8")))
        except Exception:
            existing_count = 0

        _info(f"data/chunks.json already exists ({existing_count:,} chunks).")
        if not _ask_yn("Re-chunk all transcripts? (overwrites chunks.json)"):
            _ok(f"Skipping chunking — using existing {existing_count:,} chunks.")
            return existing_count

    try:
        from ingest.chunker import process_all  # noqa: PLC0415
        chunks = process_all()
        count  = len(chunks)
    except Exception:
        _err("Chunker raised an exception:")
        traceback.print_exc()
        _ask_continue("Chunk Transcripts")
        # Try to read whatever was saved
        if _CHUNKS_FILE.exists():
            try:
                count = len(json.loads(_CHUNKS_FILE.read_text(encoding="utf-8")))
            except Exception:
                count = 0
        else:
            count = 0

    print()
    _ok(f"{count:,} total chunks created in data/chunks.json")
    return count


def step_embed() -> "chromadb.Collection | None":  # type: ignore[name-defined]
    """Embed chunks into ChromaDB. Returns the collection."""
    _banner(3, 5, "Embed into ChromaDB")

    try:
        from ingest.embedder import embed_and_load  # noqa: PLC0415
        collection = embed_and_load()
    except Exception:
        _err("Embedder raised an exception:")
        traceback.print_exc()
        _ask_continue("Embed into ChromaDB")
        return None

    print()
    _ok(f"ChromaDB collection holds {collection.count():,} documents")
    return collection


def step_verify_retrieval() -> None:
    """Run the retrieval self-test."""
    _banner(4, 5, "Verify Retrieval (Self-Test)")

    try:
        from retrieval.searcher import test_search  # noqa: PLC0415
        test_search()
    except Exception:
        _err("Retrieval test raised an exception:")
        traceback.print_exc()
        _ask_continue("Verify Retrieval")


def step_end_to_end_test() -> dict:
    """Run a full RAG pipeline test query. Returns the result dict."""
    _banner(5, 5, "End-to-End RAG Test")

    query = "What did NVIDIA say about data center revenue growth?"
    _info(f'Query: "{query}"')
    _info("Ticker filter: NVDA\n")

    try:
        from reasoning.answerer import full_pipeline  # noqa: PLC0415
        result = full_pipeline(query, ticker="NVDA")
    except Exception:
        _err("Full-pipeline test raised an exception:")
        traceback.print_exc()
        _ask_continue("End-to-End RAG Test")
        return {}

    print()
    print("  ── Result ──────────────────────────────────────────────")
    print(json.dumps(result, indent=4))
    return result


# ── Summary ───────────────────────────────────────────────────────────────────

def _final_summary(n_transcripts: int, n_chunks: int, collection) -> None:
    print()
    print("═" * WIDTH)
    print("  PIPELINE COMPLETE — Summary")
    print("═" * WIDTH)

    # Companies with data = unique tickers in transcript filenames
    tickers: set[str] = set()
    if _TRANSCRIPTS_DIR.exists():
        for f in _TRANSCRIPTS_DIR.glob("*.txt"):
            parts = f.stem.split("_")
            if parts:
                tickers.add(parts[0])

    chroma_count = collection.count() if collection is not None else "unknown"

    rows = [
        ("Companies with data",    str(len(tickers))),
        ("Total transcripts",      str(n_transcripts)),
        ("Total chunks",           f"{n_chunks:,}" if n_chunks else "unknown"),
        ("ChromaDB documents",     f"{chroma_count:,}" if isinstance(chroma_count, int) else chroma_count),
    ]
    for label, value in rows:
        print(f"  {'·'} {label:<28} {value}")

    print()
    print("  Run the UI with:")
    print()
    print("      streamlit run app/streamlit_app.py")
    print()
    print("═" * WIDTH)
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print("═" * WIDTH)
    print("  Checkit Analytics — Full Setup Pipeline")
    print("═" * WIDTH)
    print("  Stages: Download → Chunk → Embed → Verify → Test")
    print("═" * WIDTH)

    n_transcripts = step_download()
    n_chunks      = step_chunk()
    collection    = step_embed()
    step_verify_retrieval()
    step_end_to_end_test()

    # Read ChromaDB count via searcher singleton if embed returned None
    if collection is None:
        try:
            from retrieval.searcher import _collection  # noqa: PLC0415
            collection = _collection()
        except Exception:
            pass

    _final_summary(n_transcripts, n_chunks, collection)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Pipeline interrupted by user (Ctrl-C).")
        sys.exit(0)
