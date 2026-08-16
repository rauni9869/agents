"""Run a full AgentSession (fake STT/LLM/TTS/VAD) plus a barge-in replay.

Cloud APIs are not required. Pipeline numbers come from LiveKit's
metrics_collected events. Interruption counts come from the same
InterruptHandlerMixin used in production, driven with a scripted call.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# Repo root + tests/ (fake plugins live there)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.fake_io import FakeAudioInput, FakeAudioOutput, FakeTextOutput  # noqa: E402
from tests.fake_llm import FakeLLM, FakeLLMResponse  # noqa: E402
from tests.fake_stt import FakeSTT, FakeUserSpeech  # noqa: E402
from tests.fake_tts import FakeTTS, FakeTTSResponse  # noqa: E402
from tests.fake_vad import FakeVAD  # noqa: E402

from livekit.agents import AgentSession, MetricsCollectedEvent, metrics  # noqa: E402
from livekit.agents.llm import ChatChunk, ChoiceDelta, CompletionUsage  # noqa: E402
from livekit.agents.voice.transcription.synchronizer import (  # noqa: E402
    TranscriptSynchronizer,
    _SyncedAudioOutput,
)

from custom_agents.interrupt_handler_agent import InterruptHandlerAgent  # noqa: E402
from custom_agents.session_metrics import SessionMetrics  # noqa: E402

logger = logging.getLogger("eval-session")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

OUT_PATH = Path("custom_agents/session_metrics.results.json")


class _UsageLLM(FakeLLM):
    def chat(self, **kwargs):  # type: ignore[no-untyped-def]
        stream = super().chat(**kwargs)
        orig_run = stream._run

        async def _run_with_usage() -> None:
            await orig_run()
            index = stream._get_index_text()
            resp = self.fake_response_map.get(index)
            if resp is None:
                return
            prompt = max(8, len(index.split()))
            completion = max(8, len(resp.content.split()))
            stream._event_ch.send_nowait(
                ChatChunk(
                    id=str(id(stream)),
                    delta=ChoiceDelta(role="assistant", content=""),
                    usage=CompletionUsage(
                        prompt_tokens=prompt * 4,
                        completion_tokens=completion * 4,
                        total_tokens=prompt * 4 + completion * 4,
                        prompt_cached_tokens=0,
                    ),
                )
            )

        stream._run = _run_with_usage  # type: ignore[method-assign]
        return stream


def _session_from_speeches(
    speeches: list[FakeUserSpeech],
    llm_map: list[FakeLLMResponse],
    tts_map: list[FakeTTSResponse],
    *,
    speed: float,
) -> AgentSession:
    stt = FakeSTT(fake_user_speeches=speeches)
    session = AgentSession(
        vad=FakeVAD(
            fake_user_speeches=speeches,
            min_silence_duration=0.5 / speed,
            min_speech_duration=0.05 / speed,
        ),
        stt=stt,
        llm=_UsageLLM(fake_responses=llm_map),
        tts=FakeTTS(fake_responses=tts_map),
        min_interruption_duration=0.5 / speed,
        min_endpointing_delay=0.5 / speed,
        max_endpointing_delay=6.0 / speed,
        allow_interruptions=True,
        resume_false_interruption=True,
        false_interruption_timeout=0.2,
    )
    audio_output = FakeAudioOutput()
    transcription_output = FakeTextOutput()
    transcript_sync = TranscriptSynchronizer(
        next_in_chain_audio=audio_output,
        next_in_chain_text=transcription_output,
        speed=speed,
    )
    session.input.audio = FakeAudioInput()
    session.output.audio = transcript_sync.audio_output
    session.output.transcription = transcript_sync.text_output
    return session


async def _run_pipeline(session_metrics: SessionMetrics) -> None:
    speed = 1.0
    # Times below are "wall" seconds before speed-up (same pattern as tests).
    speeches = [
        FakeUserSpeech(start_time=0.5 / speed, end_time=2.4 / speed, transcript="hello how are you", stt_delay=0.2 / speed),
        FakeUserSpeech(start_time=5.0 / speed, end_time=6.8 / speed, transcript="tell me the weather in Delhi", stt_delay=0.2 / speed),
        FakeUserSpeech(start_time=10.5 / speed, end_time=12.0 / speed, transcript="and also for Mumbai", stt_delay=0.2 / speed),
        FakeUserSpeech(start_time=16.0 / speed, end_time=17.2 / speed, transcript="thanks that is all", stt_delay=0.2 / speed),
    ]
    llm = [
        FakeLLMResponse(input="hello how are you", content="Hi, I am your voice assistant. How can I help today?", ttft=0.18 / speed, duration=0.42 / speed),
        FakeLLMResponse(input="tell me the weather in Delhi", content="In Delhi it is sunny and around thirty two degrees.", ttft=0.22 / speed, duration=0.50 / speed),
        FakeLLMResponse(input="and also for Mumbai", content="Mumbai is humid with a chance of light rain this evening.", ttft=0.20 / speed, duration=0.45 / speed),
        FakeLLMResponse(input="thanks that is all", content="You are welcome. Goodbye.", ttft=0.12 / speed, duration=0.28 / speed),
    ]
    tts = [
        FakeTTSResponse(input=llm[0].content, audio_duration=2.4 / speed, ttfb=0.16 / speed, duration=0.32 / speed),
        FakeTTSResponse(input=llm[1].content, audio_duration=2.8 / speed, ttfb=0.19 / speed, duration=0.35 / speed),
        FakeTTSResponse(input=llm[2].content, audio_duration=2.6 / speed, ttfb=0.17 / speed, duration=0.30 / speed),
        FakeTTSResponse(input=llm[3].content, audio_duration=1.4 / speed, ttfb=0.11 / speed, duration=0.22 / speed),
    ]

    session = _session_from_speeches(speeches, llm, tts, speed=speed)
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)
        session_metrics.pipeline.collect(ev.metrics)

    agent = InterruptHandlerAgent(
        metrics=session_metrics,
        instructions="You are a helpful voice assistant.",
    )
    await session.start(agent)
    assert isinstance(session.input.audio, FakeAudioInput)
    session.input.audio.push(0.1)
    assert isinstance(session.stt, FakeSTT)
    await session.stt.fake_user_speeches_done
    await asyncio.sleep(1.0)
    with contextlib.suppress(RuntimeError):
        await session.drain()
    await session.aclose()
    if isinstance(session.output.audio, _SyncedAudioOutput):
        await session.output.audio._synchronizer.aclose()

    logger.info("UsageCollector: %s", usage_collector.get_summary())


class _MockSession:
    def __init__(self) -> None:
        self._handlers: dict[str, list] = {}
        self.current_speech = None
        self.interrupt_count = 0

    def on(self, name: str):
        def _wrap(fn):
            self._handlers.setdefault(name, []).append(fn)
            return fn

        return _wrap

    async def emit(self, name: str, ev=None) -> None:
        for fn in self._handlers.get(name, []):
            result = fn(ev)
            if asyncio.iscoroutine(result):
                await result

    def interrupt(self) -> None:
        self.interrupt_count += 1
        self.current_speech = None

    def clear_user_turn(self) -> None:
        return


async def _run_barge_in_replay(session_metrics: SessionMetrics) -> None:
    """Drive the production mixin with a scripted call (fillers, commands, speech)."""
    mock = _MockSession()
    agent = InterruptHandlerAgent(metrics=session_metrics, instructions="eval")
    agent.attach(mock)

    turns = [
        ("agent_speech_started", None),
        ("user_input_transcribed", "um"),
        ("user_input_transcribed", "uh hmm"),
        ("user_input_transcribed", "haan"),
        ("user_input_transcribed", "what is the weather in Delhi"),
        ("agent_speech_ended", None),
        ("user_input_transcribed", "um"),  # silent: filler counts as speech
        ("agent_speech_started", None),
        ("user_input_transcribed", "uh"),
        ("user_input_transcribed", "umm"),
        ("user_input_transcribed", "stop"),
        ("agent_speech_ended", None),
        ("agent_speech_started", None),
        ("user_input_transcribed", "wait"),
        ("user_input_transcribed", "actually tell me about Mumbai"),
        ("agent_speech_ended", None),
        ("user_input_transcribed", "thanks that is all"),
    ]

    for event_name, text in turns:
        if event_name == "agent_speech_started":
            mock.current_speech = object()
            await mock.emit("agent_speech_started")
        elif event_name == "agent_speech_ended":
            mock.current_speech = None
            await mock.emit("agent_speech_ended")
        else:
            await mock.emit(
                "user_input_transcribed",
                SimpleNamespace(transcript=text, is_final=True, confidence=0.91),
            )

    logger.info("mock interrupt() calls: %s", mock.interrupt_count)


async def main() -> None:
    session_metrics = SessionMetrics()
    await _run_pipeline(session_metrics)
    # Pipeline run also attached the mixin; reset interruption counters so the
    # scripted barge-in call is the source of those rates (pipeline STT turns
    # do not include overlapping fillers).
    from custom_agents.session_metrics import InterruptionCounters

    session_metrics.interruptions = InterruptionCounters()
    await _run_barge_in_replay(session_metrics)

    snapshot = session_metrics.snapshot()
    snapshot["run_notes"] = {
        "pipeline": "LiveKit AgentSession with fake STT/LLM/TTS/VAD (no cloud API keys in this environment)",
        "interruptions": "InterruptHandlerMixin replay of a 16-turn call with fillers, stop/wait, and real barge-in",
    }
    OUT_PATH.write_text(json.dumps(snapshot, indent=2) + "\n")
    print(json.dumps(snapshot, indent=2))
    print(f"\nWrote {OUT_PATH.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
