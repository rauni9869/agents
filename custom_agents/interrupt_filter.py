"""Classify transcribed utterances for barge-in handling.

Kept free of LiveKit imports so classification and metrics can be unit-tested
without provider credentials or a worker process.
"""

from __future__ import annotations

import os
import re
from enum import Enum
from typing import Iterable

DEFAULT_FILLERS = ("uh", "umm", "um", "hmm", "haan", "huh", "ah", "er")
DEFAULT_COMMANDS = ("wait", "stop", "hold", "pause")


class UtteranceKind(str, Enum):
    FILLER = "filler"
    COMMAND = "command"
    SPEECH = "speech"
    EMPTY = "empty"


def csv_env(name: str, default: tuple[str, ...] | list[str]) -> list[str]:
    raw = os.getenv(name, "")
    if not raw.strip():
        return [w.lower() for w in default]
    return [w.strip().lower() for w in raw.split(",") if w.strip()]


def normalize_words(text: str) -> list[str]:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    return [part for part in text.split() if part]


def classify_utterance(
    text: str,
    *,
    fillers: Iterable[str],
    commands: Iterable[str],
) -> tuple[UtteranceKind, list[str]]:
    words = normalize_words(text)
    if not words:
        return UtteranceKind.EMPTY, words

    filler_set = {w.lower() for w in fillers}
    command_set = {w.lower() for w in commands}

    if any(word in command_set for word in words):
        return UtteranceKind.COMMAND, words
    if all(word in filler_set for word in words):
        return UtteranceKind.FILLER, words
    return UtteranceKind.SPEECH, words
