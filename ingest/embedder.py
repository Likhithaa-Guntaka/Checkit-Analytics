"""
ingest/embedder.py — Embed speaker-aware chunks and load them into ChromaDB.

Reads data/chunks.json produced by chunker.py, encodes each chunk's text with
the sentence-transformers model all-MiniLM-L6-v2, and upserts the resulting
vectors into a persistent ChromaDB collection named "earnings_calls".

Re-runs are idempotent: if the collection already holds more than SKIP_THRESHOLD
documents the embedding step is skipped and the existing collection is returned.

Public entry points
-------------------
embed_and_load()              → chromadb.Collection   (primary pipeline call)
embed_and_store(chunks, path) → None                  (pipeline.py compat shim)
get_collection(path)          → chromadb.Collection   (pipeline.py compat shim)
"""

import json
from pathlib import Path
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────

CHUNKS_FILE = Path(__file__).resolve().parent.parent / "data" / "chunks.json"
CHROMA_PATH = str(Path(__file__).resolve().parent.parent / "chroma_db")

# ── Configuration ─────────────────────────────────────────────────────────────

COLLECTION_NAME    = "earnings_calls"
EMBED_MODEL        = "all-MiniLM-L6-v2"
BATCH_SIZE         = 64     # chunks per embedding batch
SKIP_THRESHOLD     = 100    # skip embedding if collection already exceeds this count

# Sanity-check parameters
SANITY_QUERY              = "data center revenue growth"
SANITY_N_RESULTS          = 3
SANITY_DISTANCE_THRESHOLD = 0.6    # cosine distance below this is considered relevant
SANITY_KEYWORDS: list[str] = ["revenue", "data center", "cloud", "growth"]

# Metadata fields stored alongside each vector in ChromaDB
META_FIELDS: list[str] = [
    "company", "ticker", "quarter", "year",
    "section", "speaker", "speaker_role", "speaker_type", "filing_date",
]

# Module-level model cache — ensures the SentenceTransformer is loaded at most once
_MODEL_CACHE: dict[str, SentenceTransformer] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_collection() -> chromadb.Collection:
    """Connect to the persistent ChromaDB store and return the earnings collection."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _build_metadata(chunk: dict) -> dict[str, str]:
    """Extract and stringify the metadata fields from a chunk dict."""
    return {field: str(chunk.get(field, "")) for field in META_FIELDS}


def _load_model() -> SentenceTransformer:
    """Return the cached embedding model, loading it on first call."""
    if "model" not in _MODEL_CACHE:
        print(f"Loading embedding model '{EMBED_MODEL}' …")
        _MODEL_CACHE["model"] = SentenceTransformer(EMBED_MODEL)
    return _MODEL_CACHE["model"]


def _embed_batches(
    chunks: list[dict],
    collection: chromadb.Collection,
    model: SentenceTransformer,
) -> None:
    """Encode chunks in batches and upsert embeddings into the collection."""
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_start in tqdm(
        range(0, len(chunks), BATCH_SIZE),
        total=total_batches,
        desc="Embedding & loading",
        unit="batch",
    ):
        batch     = chunks[batch_start : batch_start + BATCH_SIZE]
        texts     = [c["text"]     for c in batch]
        ids       = [c["chunk_id"] for c in batch]
        metadatas = [_build_metadata(c) for c in batch]

        embeddings = model.encode(
            texts,
            show_progress_bar=False,
            batch_size=BATCH_SIZE,
        ).tolist()

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )


def _sanity_check(collection: chromadb.Collection) -> None:
    """Query the collection with a known phrase and print pass/fail for the top hits."""
    print(f"\n── Sanity check: '{SANITY_QUERY}' ─────────────────────────────")

    model = _load_model()
    q_vec = model.encode([SANITY_QUERY]).tolist()

    results   = collection.query(
        query_embeddings=q_vec,
        n_results=SANITY_N_RESULTS,
        include=["documents", "metadatas", "distances"],
    )
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    relevant = False
    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
        snippet = doc[:100].replace("\n", " ")
        print(
            f"  [{rank}] {meta.get('company', '?')} {meta.get('quarter', '?')} "
            f"{meta.get('year', '?')} | speaker: {meta.get('speaker', '?') or 'N/A'} "
            f"| dist: {dist:.4f}"
        )
        print(f"       \"{snippet}…\"")
        if dist < SANITY_DISTANCE_THRESHOLD and any(kw in doc.lower() for kw in SANITY_KEYWORDS):
            relevant = True

    if relevant:
        print("\nEmbedder sanity check passed ✓")
    else:
        print("\nEmbedder sanity check: results may be weak — check collection size.")


# ── Main ──────────────────────────────────────────────────────────────────────

def embed_and_load() -> chromadb.Collection:
    """Load chunks.json, embed with all-MiniLM-L6-v2, and upsert into ChromaDB.

    Skips the embedding phase when the collection already holds more than
    SKIP_THRESHOLD documents, making re-runs safe and fast.
    Returns the ChromaDB collection.
    """
    # Step 1: Load chunks
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"chunks.json not found at {CHUNKS_FILE}. Run ingest/chunker.py first."
        )
    chunks: list[dict] = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    print(f"Loaded {len(chunks):,} chunks from {CHUNKS_FILE.name}")

    # Step 2: Connect to ChromaDB
    collection = _get_collection()
    existing   = collection.count()
    print(f"ChromaDB collection '{COLLECTION_NAME}': {existing:,} existing documents")

    # Step 3: Check if already loaded
    if existing > SKIP_THRESHOLD:
        print("ChromaDB already loaded — skipping embedding.")
    else:
        # Step 4: Embed and upsert
        model = _load_model()
        _embed_batches(chunks, collection, model)
        print(
            f"\nDone. ChromaDB collection '{COLLECTION_NAME}' "
            f"now holds {collection.count():,} documents."
        )

    # Step 5: Sanity check
    _sanity_check(collection)

    return collection


# ── Backward-compat shims (used by pipeline.py) ───────────────────────────────

def get_collection(persist_path: str = CHROMA_PATH) -> chromadb.Collection:
    """Return the earnings_calls collection (pipeline.py compatibility shim)."""
    client = chromadb.PersistentClient(path=persist_path)
    return client.get_or_create_collection(COLLECTION_NAME)


def embed_and_store(chunks: list[dict], persist_path: str = CHROMA_PATH) -> None:
    """Embed a list of chunk dicts and upsert into ChromaDB (pipeline.py compat shim)."""
    model      = SentenceTransformer(EMBED_MODEL)
    client     = chromadb.PersistentClient(path=persist_path)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding"):
        batch      = chunks[i : i + BATCH_SIZE]
        texts      = [c["text"]     for c in batch]
        ids        = [c["chunk_id"] for c in batch]
        metadatas  = [_build_metadata(c) for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    print(f"Stored {len(chunks):,} chunks in ChromaDB at '{persist_path}'.")


if __name__ == "__main__":
    embed_and_load()
