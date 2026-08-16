from types import SimpleNamespace

from custom_agents.interrupt_filter import UtteranceKind, classify_utterance, normalize_words
from custom_agents.session_metrics import InterruptionCounters, PipelineStats, SessionMetrics


def test_normalize_and_classify_filler_command_speech():
    assert normalize_words("Um, uh!") == ["um", "uh"]
    fillers = {"um", "uh", "hmm"}
    commands = {"stop", "wait"}

    kind, words = classify_utterance("uh um", fillers=fillers, commands=commands)
    assert kind is UtteranceKind.FILLER
    assert words == ["uh", "um"]

    kind, _ = classify_utterance("please stop talking", fillers=fillers, commands=commands)
    assert kind is UtteranceKind.COMMAND

    kind, _ = classify_utterance("what is the weather", fillers=fillers, commands=commands)
    assert kind is UtteranceKind.SPEECH

    kind, _ = classify_utterance("   ", fillers=fillers, commands=commands)
    assert kind is UtteranceKind.EMPTY


def test_interruption_counters_and_rates():
    counters = InterruptionCounters()
    assert counters.record(UtteranceKind.FILLER, agent_speaking=True) == "ignore_filler"
    assert counters.record(UtteranceKind.FILLER, agent_speaking=True) == "ignore_filler"
    assert counters.record(UtteranceKind.SPEECH, agent_speaking=True) == "interrupt_speech"
    assert counters.record(UtteranceKind.COMMAND, agent_speaking=True) == "interrupt_command"
    assert counters.record(UtteranceKind.FILLER, agent_speaking=False) == "filler_while_silent"

    assert counters.fillers_ignored_while_speaking == 2
    assert counters.barge_in_attempts_while_speaking == 4
    assert counters.false_barge_in_suppression_rate == 0.5
    assert counters.valid_barge_in_rate == 0.5


def test_pipeline_collect_and_snapshot(tmp_path):
    stats = PipelineStats()
    stats.collect(
        SimpleNamespace(
            type="llm_metrics",
            ttft=0.2,
            duration=0.8,
            prompt_tokens=100,
            completion_tokens=40,
            cancelled=False,
        )
    )
    stats.collect(
        SimpleNamespace(
            type="llm_metrics",
            ttft=0.4,
            duration=1.0,
            prompt_tokens=50,
            completion_tokens=20,
            cancelled=True,
        )
    )
    stats.collect(
        SimpleNamespace(type="tts_metrics", ttfb=0.15, audio_duration=2.5, cancelled=False)
    )
    stats.collect(
        SimpleNamespace(type="eou_metrics", end_of_utterance_delay=0.3, transcription_delay=0.1)
    )

    session = SessionMetrics()
    session.pipeline = stats
    session.interruptions.record(UtteranceKind.FILLER, agent_speaking=True)
    snapshot = session.snapshot()

    assert snapshot["pipeline"]["llm_ttft_avg_s"] == 0.3
    assert snapshot["pipeline"]["llm_prompt_tokens"] == 150
    assert snapshot["pipeline"]["llm_cancelled"] == 1
    assert snapshot["pipeline"]["tts_ttfb_avg_s"] == 0.15
    assert snapshot["interruptions"]["fillers_ignored_while_speaking"] == 1

    path = session.write_json(tmp_path / "session_metrics.json")
    assert path.exists()
    assert "llm_ttft_avg_s" in path.read_text()
