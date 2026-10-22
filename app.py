"""
app.py — Step 6: Flask web UI for ZentroMart Voice Support Agent.

Endpoints:
  GET  /              → Main chat UI
  POST /api/chat      → Text chat turn
  POST /api/voice     → Voice turn (upload audio)
  POST /api/reset     → Reset conversation
  GET  /api/status    → System health check
"""
import os
import uuid
import json
from pathlib import Path

from flask import (
    Flask, request, jsonify, send_file,
    render_template, session
)
from flask_session import Session

from src.config import OUTPUTS_DIR, GNANI_API_KEY, GEMINI_API_KEY, AGENT_NAME
from src.conversation_manager import new_conversation, chat_turn, ConversationState
from src.rag_pipeline import ensure_index

app = Flask(__name__, template_folder="../templates", static_folder="../static")
app.secret_key = os.urandom(32)
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = str(OUTPUTS_DIR / "flask_sessions")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB upload limit
Session(app)

Path(app.config["SESSION_FILE_DIR"]).mkdir(parents=True, exist_ok=True)

# Conversation store (server-side, keyed by session ID)
_CONVERSATIONS: dict[str, ConversationState] = {}


def _get_state() -> ConversationState:
    sid = session.get("conv_id")
    if not sid or sid not in _CONVERSATIONS:
        sid = str(uuid.uuid4())
        session["conv_id"] = sid
        _CONVERSATIONS[sid] = new_conversation()
    return _CONVERSATIONS[sid]


def _set_state(state: ConversationState) -> None:
    _CONVERSATIONS[session["conv_id"]] = state


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", agent_name=AGENT_NAME)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data    = request.get_json(force=True)
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "Empty message"}), 400

    state = _get_state()
    try:
        reply, state = chat_turn(message, state, use_rag=True)
        _set_state(state)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({
        "reply":         reply,
        "tool_results":  state.last_tool_results,
        "evidence":      [
            {"section_id": c.get("section_id", ""), "title": c.get("section_title", "")}
            for c in state.last_evidence
        ],
        "verified":      state.verification.verified,
        "ticket_ids": [
            t.get("ticket_id") for t in state.last_tool_results
            if t.get("created")
        ],
    })


@app.route("/api/voice", methods=["POST"])
def api_voice():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    suffix = Path(audio_file.filename or "upload.wav").suffix or ".wav"
    tmp_path = OUTPUTS_DIR / f"upload_{uuid.uuid4().hex[:8]}{suffix}"
    audio_file.save(str(tmp_path))

    if not GNANI_API_KEY:
        return jsonify({
            "error": "GNANI_API_KEY not configured. Voice features require a Gnani AI account.",
            "hint": "Visit https://app.gnani.ai/voice to get your API key.",
        }), 503

    try:
        from src.voice_agent import voice_turn
        state = _get_state()
        transcript, reply, audio_path = voice_turn(
            audio_path=tmp_path,
            state=state,
            use_rag=True,
            output_dir=OUTPUTS_DIR,
        )
        _set_state(state)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    resp = {
        "transcript": transcript,
        "reply":      reply,
        "verified":   state.verification.verified,
        "tool_results": state.last_tool_results,
        "evidence": [
            {"section_id": c.get("section_id", ""), "title": c.get("section_title", "")}
            for c in state.last_evidence
        ],
        "ticket_ids": [
            t.get("ticket_id") for t in state.last_tool_results
            if t.get("created")
        ],
    }

    # Attach audio file path for streaming
    if audio_path.exists() and audio_path.suffix in (".wav", ".mp3"):
        resp["audio_url"] = f"/api/audio/{audio_path.name}"

    return jsonify(resp)


@app.route("/api/audio/<filename>")
def serve_audio(filename: str):
    """Serve a generated TTS audio file."""
    audio_path = OUTPUTS_DIR / filename
    if not audio_path.exists():
        return jsonify({"error": "File not found"}), 404
    return send_file(str(audio_path), mimetype="audio/wav")


@app.route("/api/reset", methods=["POST"])
def api_reset():
    sid = session.get("conv_id")
    if sid and sid in _CONVERSATIONS:
        del _CONVERSATIONS[sid]
    session.pop("conv_id", None)
    return jsonify({"status": "reset"})


@app.route("/api/status")
def api_status():
    from src.rag_pipeline import _get_collection
    try:
        col = _get_collection()
        chunk_count = col.count()
    except Exception:
        chunk_count = -1

    return jsonify({
        "gemini_configured": bool(GEMINI_API_KEY),
        "gnani_configured":  bool(GNANI_API_KEY),
        "rag_chunks":        chunk_count,
        "agent_name":        AGENT_NAME,
    })


# ── Startup ───────────────────────────────────────────────────────────────────
def create_app() -> Flask:
    """Factory — builds the index on first run."""
    print("[App] Building RAG index (one-time)...", flush=True)
    try:
        ensure_index()
    except Exception as exc:
        print(f"[App] Warning: RAG index failed to build: {exc}", flush=True)
    return app


if __name__ == "__main__":
    create_app()
    app.run(host="0.0.0.0", port=5000, debug=False)
