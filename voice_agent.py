"""
voice_agent.py — Step 5: Gnani AI STT + TTS integration.

Uses the official gnani-vachana SDK:
  pip install gnani-vachana

Pipeline:
  audio_file → GnaniSTTClient → transcript → conversation agent
             → reply_text → GnaniTTSClient → audio_reply_file

API key: GNANI_API_KEY environment variable.
"""
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from src.config import (
    GNANI_API_KEY, GNANI_STT_LANG, GNANI_TTS_MODEL, GNANI_TTS_VOICE,
    STT_MAX_SECONDS, OUTPUTS_DIR
)
from src.conversation_manager import ConversationState, chat_turn


def _check_gnani_key() -> None:
    if not GNANI_API_KEY:
        raise EnvironmentError(
            "GNANI_API_KEY is not set. "
            "Register at https://app.gnani.ai/voice and add it to your .env file."
        )


# ── STT: Speech → Text ───────────────────────────────────────────────────────
def transcribe_audio(audio_path: str | Path) -> str:
    """
    Transcribe an audio file using Gnani AI STT.

    Args:
        audio_path: Path to a WAV/MP3 audio file (max 60 seconds).

    Returns:
        Transcribed text, or "" if the audio was unclear.

    Raises:
        EnvironmentError: If GNANI_API_KEY is missing.
        FileNotFoundError: If the audio file does not exist.
    """
    _check_gnani_key()
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        from gnani.stt import GnaniSTTClient
    except ImportError:
        try:
            from gnani_vachana import GnaniSTTClient  # type: ignore
        except ImportError:
            raise ImportError(
                "gnani-vachana is not installed. Run: pip install gnani-vachana"
            )

    client = GnaniSTTClient(api_key=GNANI_API_KEY)

    # Submit transcription (accepts file path or bytes)
    result = client.transcribe(
        audio=str(audio_path),
        language_code=GNANI_STT_LANG,
    )

    if isinstance(result, dict):
        transcript = result.get("transcript") or result.get("text") or ""
        if not transcript and isinstance(result.get("data"), dict):
            transcript = result["data"].get("transcript", "")
        elif not transcript and isinstance(result.get("data"), str):
            transcript = result["data"]
    else:
        transcript = str(result)
    return str(transcript).strip()


# ── TTS: Text → Speech ───────────────────────────────────────────────────────
def synthesize_speech(text: str, output_path: Optional[Path] = None) -> Path:
    """
    Synthesize speech from text using Gnani AI TTS.

    Args:
        text:        The text to speak (keep <200 words for best quality).
        output_path: Where to save the audio file. Auto-generated if None.

    Returns:
        Path to the saved audio file.
    """
    _check_gnani_key()
    if not text.strip():
        raise ValueError("Cannot synthesize empty text.")

    try:
        from gnani.tts import GnaniTTSClient
    except ImportError:
        try:
            from gnani_vachana import GnaniTTSClient  # type: ignore
        except ImportError:
            raise ImportError(
                "gnani-vachana is not installed. Run: pip install gnani-vachana"
            )

    if output_path is None:
        output_path = OUTPUTS_DIR / f"reply_{uuid.uuid4().hex[:8]}.wav"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = GnaniTTSClient(api_key=GNANI_API_KEY)

    audio_bytes = client.synthesize(
        text=text,
        voice=GNANI_TTS_VOICE,
        model=GNANI_TTS_MODEL,
        output_file=str(output_path),
    )

    if audio_bytes and not output_path.exists():
        output_path.write_bytes(audio_bytes)
    return output_path


# ── Full voice turn ───────────────────────────────────────────────────────────
def voice_turn(
    audio_path: str | Path,
    state: ConversationState,
    use_rag: bool = True,
    output_dir: Optional[Path] = None,
) -> tuple[str, str, Path]:
    """
    Complete voice turn: STT → Agent → TTS.

    Args:
        audio_path:  Customer's audio input file.
        state:       Current conversation state (mutated in-place).
        use_rag:     Whether to use RAG retrieval.
        output_dir:  Directory for the synthesized audio reply.

    Returns:
        (transcript, reply_text, reply_audio_path)

    Gracefully handles:
      - Empty or unclear transcripts → asks customer to repeat.
      - Gnani API errors → returns error message as text.
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir) if output_dir else OUTPUTS_DIR

    # Step 1: STT
    try:
        transcript = transcribe_audio(audio_path)
    except Exception as exc:
        error_msg = f"Sorry, I couldn't process the audio: {exc}"
        return "", error_msg, _synthesize_safe(error_msg, output_dir)

    # Step 2: Handle unclear transcript
    if not transcript or len(transcript) < 3:
        fallback = (
            "I'm sorry, I didn't catch that. Could you please repeat your question?"
        )
        return transcript, fallback, _synthesize_safe(fallback, output_dir)

    # Step 3: Generate agent reply
    try:
        reply, state = chat_turn(transcript, state, use_rag=use_rag)
    except Exception as exc:
        reply = (
            "I'm sorry, something went wrong while processing your request. "
            "Please try again or contact us at support@zentromart.com."
        )

    # Step 4: TTS
    audio_file_path = output_dir / f"reply_{uuid.uuid4().hex[:8]}.wav"
    final_audio = _synthesize_safe(reply, output_dir, audio_file_path)

    return transcript, reply, final_audio


def _synthesize_safe(text: str, output_dir: Path, out_path: Optional[Path] = None) -> Path:
    """Synthesize speech; on failure return a placeholder path."""
    try:
        return synthesize_speech(text, output_path=out_path)
    except Exception as exc:
        # Write a note file instead of crashing
        placeholder = output_dir / f"tts_error_{uuid.uuid4().hex[:6]}.txt"
        placeholder.parent.mkdir(parents=True, exist_ok=True)
        placeholder.write_text(f"TTS failed: {exc}\nText: {text}", encoding="utf-8")
        return placeholder
