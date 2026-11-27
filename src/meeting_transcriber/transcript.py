from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import asdict, dataclass
from itertools import groupby
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from faster_whisper.transcribe import Segment


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    score: float


@dataclass(frozen=True)
class Utterance:
    text: str
    start: float
    end: float
    speaker: str | None
    needs_review: bool
    words: list[Word]


class SpeakerTimeline:
    """Map time ranges to the exclusive speaker turn with the most overlap."""

    def __init__(self, turns: Iterable[tuple[float, float, str]]) -> None:
        self.turns = sorted(turns)
        self.ends = [end for _, end, _ in self.turns]
        self.labels: dict[str, str] = {}
        for _, _, label in self.turns:
            if label not in self.labels:
                self.labels[label] = f"Speaker {len(self.labels) + 1}"

    def speaker_at(self, start: float, end: float) -> str:
        best_label = "Unknown speaker"
        best_overlap = 0.0
        for index in range(bisect_right(self.ends, start), len(self.turns)):
            turn_start, turn_end, label = self.turns[index]
            if start == end and turn_start <= start < turn_end:
                return self.labels[label]
            if turn_start >= end:
                break
            overlap = min(end, turn_end) - max(start, turn_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = self.labels[label]
        return best_label


def build_utterances(
    segments: Iterable[Segment],
    threshold: float,
    timeline: SpeakerTimeline | None,
) -> list[Utterance]:
    utterances = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        words = [
            Word(word.word, float(word.start), float(word.end), float(word.probability))
            for word in segment.words or []
        ]
        aligned_text = "".join(word.text for word in words)
        complete = aligned_text.split() == text.split()
        if timeline is None or not complete:
            speaker = (
                timeline.speaker_at(segment.start, segment.end) if timeline else None
            )
            utterances.append(
                Utterance(
                    text, float(segment.start), float(segment.end), speaker,
                    not complete or any(word.score < threshold for word in words),
                    words,
                )
            )
            continue

        for speaker, group in groupby(
            words, key=lambda word: timeline.speaker_at(word.start, word.end)
        ):
            spoken = list(group)
            spoken_text = "".join(word.text for word in spoken).strip()
            if spoken_text:
                utterances.append(
                    Utterance(
                        spoken_text, spoken[0].start, spoken[-1].end, speaker,
                        any(word.score < threshold for word in spoken), spoken,
                    )
                )
    return utterances


def _timestamp(milliseconds: int, decimal: str) -> str:
    seconds, fraction = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}{decimal}{fraction:03}"


def _speaker_text(utterance: Utterance) -> str:
    text = " ".join(utterance.text.split())
    if utterance.speaker:
        return f"{utterance.speaker}: {text}"
    return text


def _text(utterances: list[Utterance]) -> str:
    lines = []
    for item in utterances:
        stamp = _timestamp(round(item.start * 1000), ".")
        review = " [review]" if item.needs_review else ""
        lines.append(f"[{stamp}]{review} {_speaker_text(item)}")
    return "\n".join(lines) + "\n" if lines else "No English speech detected.\n"


def _captions(utterances: list[Utterance]) -> str:
    cues = []
    previous_end = 0
    for number, item in enumerate(utterances, start=1):
        start = max(previous_end, round(item.start * 1000))
        end = max(start + 1, round(item.end * 1000))
        times = f"{_timestamp(start, ',')} --> {_timestamp(end, ',')}"
        cues.append(f"{number}\n{times}\n{_speaker_text(item)}\n\n")
        previous_end = end
    return "".join(cues)


def write_outputs(
    destination: Path,
    utterances: list[Utterance],
    *,
    source: Path,
    model: str,
    duration: float,
    threshold: float,
    diarized: bool,
) -> None:
    details = {
        "source": source.name,
        "language": "en",
        "model": model,
        "duration_seconds": duration,
        "review_threshold": threshold,
        "speaker_diarization": diarized,
        "utterances": [asdict(utterance) for utterance in utterances],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".transcribing-", dir=destination.parent) as temp:
        staged = Path(temp) / "result"
        staged.mkdir()
        (staged / "transcript.txt").write_text(_text(utterances), encoding="utf-8")
        (staged / "transcript.srt").write_text(_captions(utterances), encoding="utf-8")
        (staged / "transcript.json").write_text(
            json.dumps(details, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Results folder already exists: {destination}")
        # Publish the complete folder in one rename on the same filesystem.
        staged.rename(destination)
