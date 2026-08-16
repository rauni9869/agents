import logging
import os
from pathlib import Path

from livekit.agents import (
    AgentSession,
    JobContext,
    MetricsCollectedEvent,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.plugins import cartesia, deepgram, openai, silero

from custom_agents.interrupt_handler_agent import InterruptHandlerAgent
from custom_agents.session_metrics import SessionMetrics

logger = logging.getLogger("custom-agent")
logging.basicConfig(level=logging.INFO)

METRICS_PATH = Path(os.getenv("SESSION_METRICS_PATH", "custom_agents/session_metrics.json"))


async def entrypoint(ctx: JobContext):
    session_metrics = SessionMetrics()
    usage_collector = metrics.UsageCollector()

    agent = InterruptHandlerAgent(
        metrics=session_metrics,
        instructions=(
            "You are a helpful voice assistant. Ignore filler-only interruptions "
            "while speaking; stop immediately on commands like wait or stop."
        ),
    )

    session = AgentSession(
        turn_detection="vad",
        vad=silero.VAD.load(),
        stt=deepgram.STTv2(
            model="flux-general-en",
            eager_eot_threshold=0.4,
        ),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(model="sonic-3"),
        allow_interruptions=True,
        resume_false_interruption=True,
        false_interruption_timeout=0.2,
    )

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)
        session_metrics.pipeline.collect(ev.metrics)

    async def flush_metrics():
        usage = usage_collector.get_summary()
        logger.info("Usage: %s", usage)
        path = session_metrics.write_json(METRICS_PATH)
        logger.info("Wrote session metrics to %s", path.resolve())

    ctx.add_shutdown_callback(flush_metrics)

    await session.start(agent=agent, room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
