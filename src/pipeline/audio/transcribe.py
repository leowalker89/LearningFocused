import assemblyai as aai  # type: ignore
import os
import json
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for API key
api_key = os.getenv("ASSEMBLYAI_API_KEY")
if not api_key:
    print("Error: ASSEMBLYAI_API_KEY not found in environment variables.")
    print("Please create a .env file with your API key.")
else:
    aai.settings.api_key = api_key
    # The AssemblyAI SDK defaults to a relatively small HTTP timeout. For large MP3 uploads
    # this can raise "The read operation timed out" even though the service is healthy.
    # Make this configurable so long episodes / slow networks don't fail spuriously.
    http_timeout = float(os.getenv("LF_ASSEMBLYAI_HTTP_TIMEOUT_SECONDS", "120"))
    if hasattr(aai.settings, "http_timeout"):
        aai.settings.http_timeout = http_timeout  # type: ignore[attr-defined]
    else:
        print(
            "Warning: assemblyai SDK does not expose `aai.settings.http_timeout`; "
            "consider upgrading the `assemblyai` package if you see timeouts."
        )


def transcribe_audio(file_path: str, output_dir: str = "transcripts") -> str:
    """
    Transcribes an audio file using AssemblyAI with speaker diarization and saves as JSON.

    Args:
        file_path: Path to the audio file.
        output_dir: Directory to save the transcription JSON.

    Returns:
        Path to the saved JSON file.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Transcribing: {file_path}")

    config = aai.TranscriptionConfig(speaker_labels=True)
    transcriber = aai.Transcriber()
    max_attempts = int(os.getenv("LF_ASSEMBLYAI_TRANSCRIBE_MAX_ATTEMPTS", "3"))
    initial_backoff = float(os.getenv("LF_ASSEMBLYAI_TRANSCRIBE_INITIAL_BACKOFF_SECONDS", "2.0"))
    max_backoff = float(os.getenv("LF_ASSEMBLYAI_TRANSCRIBE_MAX_BACKOFF_SECONDS", "30.0"))

    last_exc: Exception | None = None
    transcript = None
    for attempt in range(1, max_attempts + 1):
        try:
            transcript = transcriber.transcribe(file_path, config=config)
            last_exc = None
            break
        except Exception as e:  # noqa: BLE001
            last_exc = e
            msg = str(e).lower()
            retryable = any(s in msg for s in ["timed out", "timeout", "read operation timed out"])
            if (attempt >= max_attempts) or (not retryable):
                raise
            backoff = min(max_backoff, initial_backoff * (2 ** (attempt - 1)))
            print(f"Transcribe attempt {attempt}/{max_attempts} failed (will retry in {backoff:.1f}s): {e}")
            time.sleep(backoff)

    if transcript is None:
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Transcription failed without exception (unexpected).")

    if transcript.status == aai.TranscriptStatus.error:
        raise Exception(f"Transcription failed: {transcript.error}")

    json_output: dict = {
        "meta_data": {
            "og_file_name": os.path.basename(file_path),
            "file_path": file_path,
            "date_transcribed": datetime.now().isoformat(),
            "duration_seconds": transcript.audio_duration,
        },
        "transcript": [],
    }

    if transcript.utterances:
        for u in transcript.utterances:
            json_output["transcript"].append(
                {
                    "speaker": u.speaker,
                    "text": u.text,
                    "start_time": u.start / 1000.0,  # Convert ms to seconds
                    "end_time": u.end / 1000.0,  # Convert ms to seconds
                }
            )
    else:
        json_output["transcript"].append(
            {
                "speaker": None,
                "text": transcript.text,
                "start_time": 0.0,
                "end_time": transcript.audio_duration or 0.0,
            }
        )

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)

    return output_path


def main() -> None:
    """
    Main entry point for testing.
    """
    import sys
    from src.config import TRANSCRIPTS_DIR

    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        try:
            output_file = transcribe_audio(audio_file, output_dir=str(TRANSCRIPTS_DIR))
            print(f"\n✅ Transcription complete. Saved to: {output_file}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python -m src.pipeline.audio.transcribe <path_to_audio_file>")


if __name__ == "__main__":
    main()


