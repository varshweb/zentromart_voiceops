"""
config.py — Central configuration for VoiceOps ZentroMart agent.
Loads all settings from environment variables (.env file).
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent.parent          # voiceops/
DATASET_DIR = ROOT_DIR.parent / "dataset" / "Capstone_Project_Dataset_FTTT_2026"
KB_DIR      = DATASET_DIR / "kb"
DATA_DIR    = DATASET_DIR / "data"
TOOLS_DIR   = DATASET_DIR / "tools"
PRACTICE_DIR= DATASET_DIR / "practice"

CHROMA_DIR  = ROOT_DIR / "chroma_db"
OUTPUTS_DIR = ROOT_DIR / "sample_outputs"
TICKETS_DIR = TOOLS_DIR  # tickets_created.csv lands here (tool default)

for _d in [CHROMA_DIR, OUTPUTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Load .env ────────────────────────────────────────────────────────────────
load_dotenv(ROOT_DIR / ".env")

# ── API keys (never hardcoded) ────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GNANI_API_KEY  = os.getenv("GNANI_API_KEY", "")

# ── LLM settings ─────────────────────────────────────────────────────────────
LLM_MODEL      = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
LLM_TEMPERATURE= float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_TOKENS     = int(os.getenv("MAX_TOKENS", "1024"))

# ── RAG settings ─────────────────────────────────────────────────────────────
EMBED_MODEL    = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
TOP_K_CHUNKS   = int(os.getenv("TOP_K_CHUNKS", "5"))
CHUNK_COLLECTION = "zentromart_kb"

# ── Voice settings ────────────────────────────────────────────────────────────
GNANI_STT_LANG = os.getenv("GNANI_STT_LANG", "en-IN")
GNANI_TTS_MODEL= os.getenv("GNANI_TTS_MODEL", "timbre-v2.5")
GNANI_TTS_VOICE= os.getenv("GNANI_TTS_VOICE", "Pranav")
STT_MAX_SECONDS= 60

# ── Dataset reference date ────────────────────────────────────────────────────
REFERENCE_DATE = "2026-10-20"   # fixed date per dataset spec

# ── Agent identity ────────────────────────────────────────────────────────────
AGENT_NAME     = "Zara"         # ZentroMart AI assistant
STORE_NAME     = "ZentroMart"

# ── Policy precedence (lower index = higher priority) ────────────────────────
DOC_TYPE_PRIORITY = ["policy", "agent_guide", "help", "faq", "marketing", "meta"]
