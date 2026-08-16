"""Session metrics: LiveKit pipeline stats plus custom barge-in counters.

LiveKit already emits STT / LLM / TTS / end-of-utterance metrics. This module
aggregates those events and adds interruption-filter outcomes so a session can
be summarized as JSON (useful for debugging and for resume-ready numbers).
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .interrupt_filter import UtteranceKind


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(statistics.mean(values), 4)


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return round(ordered[index], 4)


@dataclass
class InterruptionCounters:
    fillers_ignored_while_speaking: int = 0
    fillers_accepted_while_silent: int = 0
    command_interrupts: int = 0
    content_interrupts: int = 0
    user_turns_while_silent: int = 0
    empty_transcripts: int = 0

    def record(self, kind: UtteranceKind, *, agent_speaking: bool) -> str:
        """Record one final transcript and return the decision label."""
        if kind is UtteranceKind.EMPTY:
            self.empty_transcripts += 1
            return "empty"

        if kind is UtteranceKind.COMMAND:
            if agent_speaking:
                self.command_interrupts += 1
                return "interrupt_command"
            self.user_turns_while_silent += 1
            return "command_while_silent"

        if kind is UtteranceKind.FILLER:
            if agent_speaking:
                self.fillers_ignored_while_speaking += 1
                return "ignore_filler"
            self.fillers_accepted_while_silent += 1
            return "filler_while_silent"

        if agent_speaking:
            self.content_interrupts += 1
            return "interrupt_speech"
        self.user_turns_while_silent += 1
        return "user_speech"

    @property
    def barge_in_attempts_while_speaking(self) -> int:
        return (
            self.fillers_ignored_while_speaking
            + self.command_interrupts
            + self.content_interrupts
        )

    @property
    def false_barge_in_suppression_rate(self) -> float | None:
        total = self.barge_in_attempts_while_speaking
        if total == 0:
            return None
        return round(self.fillers_ignored_while_speaking / total, 4)

    @property
    def valid_barge_in_rate(self) -> float | None:
        total = self.barge_in_attempts_while_speaking
        if total == 0:
            return None
        valid = self.command_interrupts + self.content_interrupts
        return round(valid / total, 4)


@dataclass
class PipelineStats:
    llm_ttft_s: list[float] = field(default_factory=list)
    llm_duration_s: list[float] = field(default_factory=list)
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_cancelled: int = 0
    tts_ttfb_s: list[float] = field(default_factory=list)
    tts_audio_duration_s: float = 0.0
    tts_cancelled: int = 0
    stt_audio_duration_s: float = 0.0
    eou_delay_s: list[float] = field(default_factory=list)
    transcription_delay_s: list[float] = field(default_factory=list)

    def collect(self, metrics: Any) -> None:
        metric_type = getattr(metrics, "type", None)
        if metric_type == "llm_metrics":
            ttft = getattr(metrics, "ttft", None)
            if isinstance(ttft, (int, float)) and ttft >= 0:
                self.llm_ttft_s.append(float(ttft))
            duration = getattr(metrics, "duration", None)
            if isinstance(duration, (int, float)) and duration >= 0:
                self.llm_duration_s.append(float(duration))
            self.llm_prompt_tokens += int(getattr(metrics, "prompt_tokens", 0) or 0)
            self.llm_completion_tokens += int(getattr(metrics, "completion_tokens", 0) or 0)
            if getattr(metrics, "cancelled", False):
                self.llm_cancelled += 1
        elif metric_type == "tts_metrics":
            ttfb = getattr(metrics, "ttfb", None)
            if isinstance(ttfb, (int, float)) and ttfb >= 0:
                self.tts_ttfb_s.append(float(ttfb))
            self.tts_audio_duration_s += float(getattr(metrics, "audio_duration", 0.0) or 0.0)
            if getattr(metrics, "cancelled", False):
                self.tts_cancelled += 1
        elif metric_type == "stt_metrics":
            self.stt_audio_duration_s += float(getattr(metrics, "audio_duration", 0.0) or 0.0)
        elif metric_type == "eou_metrics":
            eou = getattr(metrics, "end_of_utterance_delay", None)
            if isinstance(eou, (int, float)) and eou >= 0:
                self.eou_delay_s.append(float(eou))
            tx = getattr(metrics, "transcription_delay", None)
            if isinstance(tx, (int, float)) and tx >= 0:
                self.transcription_delay_s.append(float(tx))


@dataclass
class SessionMetrics:
    started_at: float = field(default_factory=time.time)
    interruptions: InterruptionCounters = field(default_factory=InterruptionCounters)
    pipeline: PipelineStats = field(default_factory=PipelineStats)

    def snapshot(self) -> dict[str, Any]:
        elapsed = max(0.0, time.time() - self.started_at)
        counters = self.interruptions
        pipe = self.pipeline
        return {
            "session_duration_s": round(elapsed, 3),
            "interruptions": {
                "fillers_ignored_while_speaking": counters.fillers_ignored_while_speaking,
                "fillers_accepted_while_silent": counters.fillers_accepted_while_silent,
                "command_interrupts": counters.command_interrupts,
                "content_interrupts": counters.content_interrupts,
                "user_turns_while_silent": counters.user_turns_while_silent,
                "empty_transcripts": counters.empty_transcripts,
                "barge_in_attempts_while_speaking": counters.barge_in_attempts_while_speaking,
                "false_barge_in_suppression_rate": counters.false_barge_in_suppression_rate,
                "valid_barge_in_rate": counters.valid_barge_in_rate,
            },
            "pipeline": {
                "llm_ttft_avg_s": _mean(pipe.llm_ttft_s),
                "llm_ttft_p95_s": _p95(pipe.llm_ttft_s),
                "llm_duration_avg_s": _mean(pipe.llm_duration_s),
                "llm_prompt_tokens": pipe.llm_prompt_tokens,
                "llm_completion_tokens": pipe.llm_completion_tokens,
                "llm_cancelled": pipe.llm_cancelled,
                "tts_ttfb_avg_s": _mean(pipe.tts_ttfb_s),
                "tts_ttfb_p95_s": _p95(pipe.tts_ttfb_s),
                "tts_audio_duration_s": round(pipe.tts_audio_duration_s, 4),
                "tts_cancelled": pipe.tts_cancelled,
                "stt_audio_duration_s": round(pipe.stt_audio_duration_s, 4),
                "eou_delay_avg_s": _mean(pipe.eou_delay_s),
                "transcription_delay_avg_s": _mean(pipe.transcription_delay_s),
            },
        }

    def write_json(self, path: str | Path) -> Path:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(self.snapshot(), indent=2) + "\n")
        return dest
