"""
tests/test_rag.py — Unit tests for RAG pipeline.
Run: python -m pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import KB_DIR
from src.rag_pipeline import parse_kb_file, load_all_chunks


def test_parse_returns_policy():
    """The v4 returns policy should parse into multiple section chunks."""
    path = KB_DIR / "pol_returns_v4.md"
    chunks = parse_kb_file(path)
    assert len(chunks) >= 5, f"Expected >= 5 chunks, got {len(chunks)}"
    section_ids = [c["section_id"] for c in chunks]
    assert "RET-2" in section_ids, "RET-2 section missing"
    assert "RET-4" in section_ids, "RET-4 section missing"


def test_superseded_policy_marked():
    """The superseded v3 policy should have status=superseded."""
    path = KB_DIR / "pol_returns_v3_superseded.md"
    chunks = parse_kb_file(path)
    assert len(chunks) >= 1
    for c in chunks:
        assert c["status"] == "superseded", f"Expected superseded, got {c['status']}"


def test_load_all_chunks():
    """All 21 KB files should produce chunks."""
    chunks = load_all_chunks()
    assert len(chunks) >= 40, f"Too few chunks: {len(chunks)}"
    doc_ids = {c["doc_id"] for c in chunks}
    assert "POL-RET" in doc_ids, "Returns policy not indexed"
    assert "KB-00" in doc_ids, "About-KB doc not indexed"


def test_no_fabricated_superseded_content():
    """Superseded doc should not appear as current."""
    chunks = load_all_chunks()
    current_chunks = [c for c in chunks if c["status"] == "current"]
    for c in current_chunks:
        assert "superseded" not in c["file"].lower() or c["status"] == "superseded", \
            f"Superseded file marked as current: {c['file']}"
