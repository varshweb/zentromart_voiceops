"""
run.py — VoiceOps 2026: Main entry point.

Usage:
  # Start the web server (default)
  python run.py

  # Batch-process a folder of audio files
  python run.py --mode batch --audio-dir ./my_audio/ --output-dir ./sample_outputs/

  # Run the web server on a specific port
  python run.py --mode web --port 5000

  # Build/rebuild the RAG index only
  python run.py --mode index --force
"""
import argparse
import json
import sys
import os
from pathlib import Path

# ── Ensure project root is on sys.path ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUTS_DIR, GNANI_API_KEY


# ── Batch mode ────────────────────────────────────────────────────────────────
def run_batch(audio_dir: Path, output_dir: Path) -> None:
    """Process every audio file in audio_dir and save results to output_dir."""
    from src.rag_pipeline import ensure_index
    from src.conversation_manager import new_conversation
    from src.voice_agent import voice_turn

    if not GNANI_API_KEY:
        print(
            "\n[ERROR] GNANI_API_KEY is not set. "
            "Batch voice mode requires a Gnani AI account.\n"
            "Visit https://app.gnani.ai/voice to register.\n"
        )
        sys.exit(1)

    ensure_index()
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(
        f for f in audio_dir.iterdir()
        if f.suffix.lower() in (".wav", ".mp3", ".webm", ".ogg", ".m4a")
    )

    if not audio_files:
        print(f"[Batch] No audio files found in {audio_dir}")
        return

    print(f"[Batch] Processing {len(audio_files)} audio file(s)...\n")

    for audio_path in audio_files:
        print(f"  ▶ {audio_path.name}")
        state = new_conversation()
        stem  = audio_path.stem

        try:
            transcript, reply, reply_audio = voice_turn(
                audio_path=audio_path,
                state=state,
                use_rag=True,
                output_dir=output_dir,
            )
        except Exception as exc:
            print(f"    [ERROR] {exc}")
            continue

        # Save outputs
        result = {
            "audio_input":   str(audio_path),
            "transcript":    transcript,
            "reply":         reply,
            "reply_audio":   str(reply_audio),
            "evidence": [
                {"section_id": c.get("section_id",""), "title": c.get("section_title","")}
                for c in state.last_evidence
            ],
            "tool_results":  state.last_tool_results,
            "verified":      state.verification.verified,
        }

        out_json = output_dir / f"{stem}_result.json"
        out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        transcript_file = output_dir / f"{stem}_transcript.txt"
        transcript_file.write_text(transcript, encoding="utf-8")

        print(f"    Transcript:  {transcript[:80]}{'…' if len(transcript)>80 else ''}")
        print(f"    Reply:       {reply[:80]}{'…' if len(reply)>80 else ''}")
        print(f"    Audio saved: {reply_audio.name}")
        print(f"    Result JSON: {out_json.name}\n")

    print(f"[Batch] Done. Outputs saved to: {output_dir}")


# ── Index mode ────────────────────────────────────────────────────────────────
def run_index(force: bool = False) -> None:
    from src.rag_pipeline import build_index
    print(f"[Index] {'Force-rebuilding' if force else 'Building'} RAG index...")
    build_index(force=force)
    print("[Index] Done.")


# ── Web mode ──────────────────────────────────────────────────────────────────
def run_web(port: int = 5000, debug: bool = False) -> None:
    from src.app import create_app
    app = create_app()
    print(f"\n{'='*50}")
    print(f"  ZentroMart Support Agent")
    print(f"  URL: http://localhost:{port}")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*50}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VoiceOps 2026 — ZentroMart AI Support Agent"
    )
    parser.add_argument(
        "--mode",
        choices=["web", "batch", "index"],
        default="web",
        help="Run mode: web (default), batch (process audio files), index (build RAG index)",
    )
    parser.add_argument("--audio-dir",  type=Path, default=None, help="[batch] Input audio folder")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR, help="[batch] Output folder")
    parser.add_argument("--port",       type=int,  default=5000, help="[web] Server port")
    parser.add_argument("--debug",      action="store_true", help="[web] Flask debug mode")
    parser.add_argument("--force",      action="store_true", help="[index] Force rebuild")
    args = parser.parse_args()

    if args.mode == "web":
        run_web(port=args.port, debug=args.debug)
    elif args.mode == "batch":
        if not args.audio_dir:
            parser.error("--audio-dir is required for batch mode")
        run_batch(args.audio_dir, args.output_dir)
    elif args.mode == "index":
        run_index(force=args.force)
