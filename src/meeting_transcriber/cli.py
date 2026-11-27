from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from .transcript import SpeakerTimeline, build_utterances, write_outputs

if TYPE_CHECKING:
    from faster_whisper.transcribe import Segment


class TranscriptionError(Exception):
    """An error that can be reported directly to the user."""


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _threshold(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe an English meeting recording.")
    parser.add_argument("recording", nargs="?", type=Path, help="path to an audio or video file")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("transcripts"),
        help="parent folder for results (default: transcripts)",
    )
    parser.add_argument("--model", default="small.en", help="Whisper model (default: small.en)")
    parser.add_argument(
        "--list-models", action="store_true", help="list built-in Whisper model names and exit",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="hide the transcription progress bar",
    )
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu", help="Whisper device (default: cpu)",
    )
    parser.add_argument("--hotwords", help="comma-separated names or phrases to hint to Whisper")
    parser.add_argument(
        "--review-threshold", type=_threshold, default=0.5,
        help="flag text with any word scoring below this value (default: 0.5)",
    )
    parser.add_argument("--diarize", action="store_true", help="label speaker turns with pyannote")
    parser.add_argument("--speakers", type=_positive_int, help="known speaker count for --diarize")
    return parser


def _load_speaker_pipeline():
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise TranscriptionError(
            "Speaker labeling requires the optional dependencies in the README."
        ) from exc

    token = os.environ.get("HF_TOKEN") or None
    help_message = (
        "Could not load the speaker model. Check your Hugging Face access and token "
        "using the README instructions."
    )
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=token)
    except Exception as exc:
        raise TranscriptionError(f"{help_message} {exc}") from exc
    if pipeline is None:
        raise TranscriptionError(help_message)
    return pipeline


def _collect_with_progress(segments: Iterable[Segment], duration: float) -> list[Segment]:
    inline = sys.stderr.isatty() and os.environ.get("TERM") not in (None, "", "dumb")
    step = 1 if inline else 10
    width = 20
    prefix, line_end = ("\r", "") if inline else ("", "\n")

    def show(percent: int) -> None:
        filled = percent * width // 100
        bar = "#" * filled + "-" * (width - filled)
        print(
            f"{prefix}Transcribing [{bar}] {percent:3d}%",
            end=line_end,
            file=sys.stderr,
            flush=True,
        )

    collected = []
    previous = 0
    show(previous)
    try:
        for segment in segments:
            collected.append(segment)
            if duration > 0:
                percent = int(segment.end / duration * 100)
                if previous + step <= percent < 100:
                    show(percent)
                    previous = percent
        show(100)
    finally:
        if inline:
            print(file=sys.stderr, flush=True)
    return collected


def _transcribe(args: argparse.Namespace) -> Path:
    source = args.recording.expanduser()
    if not source.is_file():
        raise TranscriptionError(f"Recording file not found: {source}")
    destination = args.output_dir.expanduser() / (source.stem.lstrip(".") or "meeting")
    if destination.exists() or destination.is_symlink():
        raise TranscriptionError(
            f"Results folder already exists: {destination}. Choose a different --output-dir."
        )

    try:
        from faster_whisper import WhisperModel
        from faster_whisper.audio import decode_audio
    except ImportError as exc:
        raise TranscriptionError("Install the project with python -m pip install -e .") from exc

    sample_rate = 16_000
    try:
        audio = decode_audio(str(source), sampling_rate=sample_rate)
    except Exception as exc:
        raise TranscriptionError(f"Could not decode audio from {source}. {exc}") from exc
    if audio.size == 0:
        raise TranscriptionError(f"The recording has no audio samples: {source}")

    if args.diarize:
        print("Loading speaker model...", file=sys.stderr)
        speaker_pipeline = _load_speaker_pipeline()

    print(f"Loading Whisper model {args.model} on {args.device}...", file=sys.stderr)
    try:
        whisper = WhisperModel(
            args.model,
            device=args.device,
            compute_type="int8" if args.device == "cpu" else "float16",
        )
        print(f"Preparing {source.name} for transcription...", file=sys.stderr)
        segments, info = whisper.transcribe(
            audio,
            language="en",
            task="transcribe",
            beam_size=5,
            vad_filter=True,
            word_timestamps=True,
            condition_on_previous_text=False,
            hotwords=args.hotwords,
        )
        duration = float(info.duration)
        segments = list(segments) if args.no_progress else _collect_with_progress(segments, duration)
    except Exception as exc:
        raise TranscriptionError(f"Whisper transcription failed. {exc}") from exc

    timeline = None
    if args.diarize and segments:
        print("Identifying speaker turns...", file=sys.stderr)
        import torch

        options = {"num_speakers": args.speakers} if args.speakers is not None else {}
        try:
            result = speaker_pipeline(
                {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": sample_rate},
                **options,
            )
        except Exception as exc:
            raise TranscriptionError(f"Speaker labeling failed. {exc}") from exc
        timeline = SpeakerTimeline(
            (turn.start, turn.end, speaker)
            for turn, _, speaker in result.exclusive_speaker_diarization.itertracks(
                yield_label=True
            )
        )

    utterances = build_utterances(segments, args.review_threshold, timeline)
    write_outputs(
        destination,
        utterances,
        source=source,
        model=args.model,
        duration=duration,
        threshold=args.review_threshold,
        diarized=args.diarize,
    )
    if not utterances:
        print("No English speech detected.", file=sys.stderr)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.list_models:
        try:
            from faster_whisper import available_models
        except ImportError:
            print("Error: Install the project with python -m pip install -e .", file=sys.stderr)
            return 1
        print("\n".join(available_models()))
        return 0
    if args.recording is None:
        parser.error("provide a recording or use --list-models")
    if args.speakers is not None and not args.diarize:
        parser.error("--speakers requires --diarize")
    try:
        destination = _transcribe(args)
    except (TranscriptionError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nTranscription interrupted.", file=sys.stderr)
        return 130
    print(f"Results saved to {destination.resolve()}")
    return 0
