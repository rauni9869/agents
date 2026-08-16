import logging
from typing import Iterable, Optional

from livekit.agents import Agent, AgentSession

from .interrupt_filter import (
    DEFAULT_COMMANDS,
    DEFAULT_FILLERS,
    UtteranceKind,
    classify_utterance,
    csv_env,
)
from .session_metrics import SessionMetrics

logger = logging.getLogger("filler_filter")
logger.setLevel(logging.INFO)


class InterruptHandlerMixin:
    """Event-driven barge-in filter with session metrics.

    While the agent is speaking:
      - filler-only transcripts are ignored
      - command keywords (wait/stop/hold/pause) interrupt immediately
      - other speech is treated as a real interruption
    While the agent is silent, all non-empty speech is accepted.
    """

    def __init__(
        self,
        ignored_words: Optional[Iterable[str]] = None,
        command_keywords: Optional[Iterable[str]] = None,
        metrics: Optional[SessionMetrics] = None,
        confidence_threshold: float = 0.55,
    ):
        self.ignored_words = set(ignored_words or csv_env("FILLER_WORDS", DEFAULT_FILLERS))
        self.command_keywords = set(
            command_keywords or csv_env("COMMAND_KEYWORDS", DEFAULT_COMMANDS)
        )
        self.confidence_threshold = confidence_threshold
        self.metrics = metrics or SessionMetrics()
        self._agent_speaking = False

        logger.info("Ignored fillers: %s", sorted(self.ignored_words))
        logger.info("Command keywords: %s", sorted(self.command_keywords))

    def _is_agent_speaking(self, session: AgentSession) -> bool:
        if self._agent_speaking:
            return True
        return bool(getattr(session, "current_speech", None))

    def attach(self, session: AgentSession) -> None:
        @session.on("agent_speech_started")
        def _on_speech_started(_ev):
            self._agent_speaking = True

        @session.on("agent_speech_ended")
        def _on_speech_ended(_ev):
            self._agent_speaking = False

        @session.on("user_input_transcribed")
        def _on_transcribed(ev):
            is_final = getattr(ev, "is_final", True)
            if not is_final:
                return

            transcript = (getattr(ev, "transcript", None) or "").strip()
            kind, _words = classify_utterance(
                transcript,
                fillers=self.ignored_words,
                commands=self.command_keywords,
            )
            speaking = self._is_agent_speaking(session)
            decision = self.metrics.interruptions.record(kind, agent_speaking=speaking)
            confidence = getattr(ev, "confidence", None)
            extra = f" conf={confidence:.2f}" if isinstance(confidence, (int, float)) else ""
            logger.info("decision=%s speaking=%s text=%r%s", decision, speaking, transcript, extra)

            if kind is UtteranceKind.EMPTY:
                return

            if kind is UtteranceKind.COMMAND:
                if speaking:
                    self._interrupt(session)
                return

            if speaking and kind is UtteranceKind.FILLER:
                try:
                    session.clear_user_turn()
                except Exception:
                    pass
                return

            if speaking and kind is UtteranceKind.SPEECH:
                self._interrupt(session)

    def _interrupt(self, session: AgentSession) -> None:
        try:
            session.interrupt()
        except Exception as exc:
            logger.warning("interrupt() failed: %s", exc)


class InterruptHandlerAgent(Agent, InterruptHandlerMixin):
    def __init__(self, metrics: Optional[SessionMetrics] = None, **kwargs):
        Agent.__init__(self, **kwargs)
        InterruptHandlerMixin.__init__(self, metrics=metrics)

    async def on_enter(self) -> None:
        self.attach(self.session)
        logger.info("Filler-word interruption filter attached.")
