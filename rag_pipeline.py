"""
rag_pipeline.py — Step 3: Document chunking, embedding, retrieval.

Workflow:
  1. Parse all kb/*.md files, extracting YAML frontmatter + section chunks.
  2. Embed chunks using sentence-transformers (local, free).
  3. Store in ChromaDB (persistent, local).
  4. On query: embed question, retrieve top-K chunks, apply policy precedence.
"""
import re
import sys
import hashlib
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

from src.config import (
    KB_DIR, CHROMA_DIR, EMBED_MODEL, TOP_K_CHUNKS,
    CHUNK_COLLECTION, DOC_TYPE_PRIORITY
)

# ── Embedding function (sentence-transformers, local) ─────────────────────────
_ST_MODEL: Optional[SentenceTransformer] = None


def _get_st_model() -> SentenceTransformer:
    global _ST_MODEL
    if _ST_MODEL is None:
        print(f"[RAG] Loading embedding model '{EMBED_MODEL}'...", flush=True)
        _ST_MODEL = SentenceTransformer(EMBED_MODEL)
    return _ST_MODEL


class _LocalEmbedFn(embedding_functions.EmbeddingFunction):
    """ChromaDB-compatible embedding function wrapping SentenceTransformer."""
    def __call__(self, input: list[str]) -> list[list[float]]:
        model = _get_st_model()
        return model.encode(input, convert_to_numpy=True).tolist()


# ── ChromaDB client ───────────────────────────────────────────────────────────
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None


def _get_collection() -> chromadb.Collection:
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _chroma_client.get_or_create_collection(
            name=CHUNK_COLLECTION,
            embedding_function=_LocalEmbedFn(),
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ── Markdown parsing ──────────────────────────────────────────────────────────
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SECTION_RE     = re.compile(r"^##\s+(\[[\w\.\-]+\])\s+(.*)", re.MULTILINE)


def _parse_frontmatter(text: str) -> dict:
    """
    Robust frontmatter parser: split each line on the FIRST colon only,
    so values like 'Reverse pickup: how it works' are preserved intact.
    Handles multi-word values without requiring YAML quoting.
    """
    fm: dict = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key   = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            fm[key] = value
    return fm


def parse_kb_file(md_path: Path) -> list[dict]:
    """
    Parse one KB Markdown file into a list of chunk dicts.

    Each chunk has:
      doc_id, title, doc_type, status, effective_from,
      section_id, section_title, text, chunk_id (stable hash)
    """
    raw = md_path.read_text(encoding="utf-8")

    # Extract YAML frontmatter
    fm_match = _FRONTMATTER_RE.match(raw)
    if not fm_match:
        return []
    fm = _parse_frontmatter(fm_match.group(1))
    body = raw[fm_match.end():]

    doc_id   = fm.get("doc_id", md_path.stem)
    doc_type = fm.get("doc_type", "unknown")
    status   = fm.get("status", "current")
    eff_from = str(fm.get("effective_from", ""))
    title    = fm.get("title", "")

    chunks = []

    # Split body at ## section headings
    sections = _SECTION_RE.split(body)
    # sections[0] is preamble before first ##
    # then repeating: section_id, section_title, section_text

    # Handle preamble (intro text before first ##)
    preamble = sections[0].strip()
    if preamble:
        chunk_id = _make_chunk_id(doc_id, "preamble")
        chunks.append({
            "chunk_id":      chunk_id,
            "doc_id":        doc_id,
            "doc_type":      doc_type,
            "title":         title,
            "status":        status,
            "effective_from":eff_from,
            "section_id":    doc_id,
            "section_title": title,
            "text":          preamble,
            "file":          md_path.name,
        })

    # Handle each ## section
    i = 1
    while i + 2 <= len(sections):
        sec_id    = sections[i]
        sec_title = sections[i + 1]
        sec_text  = sections[i + 2].strip()
        i += 3

        if not sec_text:
            continue

        chunk_id = _make_chunk_id(doc_id, sec_id)
        full_text = f"## {sec_id} {sec_title}\n\n{sec_text}"
        chunks.append({
            "chunk_id":      chunk_id,
            "doc_id":        doc_id,
            "doc_type":      doc_type,
            "title":         title,
            "status":        status,
            "effective_from":eff_from,
            "section_id":    sec_id.strip("[]"),
            "section_title": sec_title.strip(),
            "text":          full_text,
            "file":          md_path.name,
        })

    return chunks


def _make_chunk_id(doc_id: str, section: str) -> str:
    raw = f"{doc_id}::{section}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def load_all_chunks() -> list[dict]:
    """Parse every *.md file in KB_DIR and return all chunks."""
    chunks = []
    for md_file in sorted(KB_DIR.glob("*.md")):
        chunks.extend(parse_kb_file(md_file))
    print(f"[RAG] Parsed {len(chunks)} chunks from {KB_DIR}.", flush=True)
    return chunks


# ── Indexing ──────────────────────────────────────────────────────────────────
def build_index(force: bool = False) -> None:
    """
    Build (or rebuild) the ChromaDB vector index from kb/*.md files.

    Args:
        force: If True, drop and recreate the collection.
    """
    global _collection

    if force and _chroma_client:
        try:
            _chroma_client.delete_collection(CHUNK_COLLECTION)
        except Exception:
            pass
        _collection = None

    collection = _get_collection()

    if collection.count() > 0 and not force:
        print(f"[RAG] Index already has {collection.count()} chunks — skipping rebuild.", flush=True)
        return

    all_chunks = load_all_chunks()
    if not all_chunks:
        raise RuntimeError(f"No chunks found in {KB_DIR}. Check KB_DIR path.")

    ids        = [c["chunk_id"]    for c in all_chunks]
    documents  = [c["text"]        for c in all_chunks]
    metadatas  = [
        {
            "doc_id":        c["doc_id"],
            "doc_type":      c["doc_type"],
            "status":        c["status"],
            "effective_from":c["effective_from"],
            "section_id":    c["section_id"],
            "section_title": c["section_title"],
            "file":          c["file"],
        }
        for c in all_chunks
    ]

    # Upsert in batches of 50
    batch = 50
    for start in range(0, len(ids), batch):
        collection.upsert(
            ids       =ids[start:start+batch],
            documents =documents[start:start+batch],
            metadatas =metadatas[start:start+batch],
        )

    print(f"[RAG] Indexed {collection.count()} chunks into ChromaDB.", flush=True)


# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(query: str, top_k: int = TOP_K_CHUNKS) -> list[dict]:
    """
    Retrieve the most relevant KB chunks for a query.

    Policy precedence applied after retrieval:
      current-policy > superseded-policy (only if policy says it applies)
      policy > agent_guide > help > faq > marketing

    Returns list of chunk dicts sorted by relevance + precedence.
    """
    collection = _get_collection()
    if collection.count() == 0:
        build_index()

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k * 2, collection.count()),   # over-fetch then rerank
        include=["documents", "metadatas", "distances"],
    )

    raw_chunks = []
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        raw_chunks.append({
            "text":          doc,
            "doc_id":        meta.get("doc_id", ""),
            "doc_type":      meta.get("doc_type", ""),
            "status":        meta.get("status", ""),
            "effective_from":meta.get("effective_from", ""),
            "section_id":    meta.get("section_id", ""),
            "section_title": meta.get("section_title", ""),
            "file":          meta.get("file", ""),
            "distance":      dist,    # lower = closer
        })

    # Rerank: apply policy precedence boost
    reranked = _rerank(raw_chunks)
    return reranked[:top_k]


def _rerank(chunks: list[dict]) -> list[dict]:
    """
    Sort chunks by (policy_precedence_score, distance).
    Superseded documents are heavily penalised.
    """
    type_rank = {t: i for i, t in enumerate(DOC_TYPE_PRIORITY)}

    def score(c: dict) -> tuple:
        superseded_penalty = 1000 if c["status"] == "superseded" else 0
        type_score = type_rank.get(c["doc_type"], len(DOC_TYPE_PRIORITY))
        return (superseded_penalty, type_score, c["distance"])

    return sorted(chunks, key=score)


def ensure_index() -> None:
    """Called at startup — builds the index if it doesn't exist yet."""
    collection = _get_collection()
    if collection.count() == 0:
        build_index()
    else:
        print(f"[RAG] Using existing index ({collection.count()} chunks).", flush=True)
