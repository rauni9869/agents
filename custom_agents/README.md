# Custom voice agent (interrupt filter + session metrics)

Run a LiveKit voice agent that:

1. Uses LiveKit's built-in STT / LLM / TTS / end-of-utterance metrics (`metrics_collected`)
2. Adds **custom barge-in metrics**: how often "um/uh" is ignored vs a real interrupt

## Run

```bash
export DEEPGRAM_API_KEY=...
export OPENAI_API_KEY=...
export CARTESIA_API_KEY=...
python custom_agents/run_custom_agent.py dev
```

When the session ends, metrics are written to `custom_agents/session_metrics.json` (override with `SESSION_METRICS_PATH`).

Optional:

```bash
export FILLER_WORDS=uh,um,umm,hmm
export COMMAND_KEYWORDS=wait,stop,hold,pause
```

## What the numbers mean

| Field | Meaning |
| --- | --- |
| `fillers_ignored_while_speaking` | Agent kept talking through "um/uh" (false barge-in suppressed) |
| `content_interrupts` / `command_interrupts` | Real barge-in while TTS was playing |
| `false_barge_in_suppression_rate` | Ignored fillers / all speaking-time barge-in attempts |
| `llm_ttft_avg_s` | Mean time-to-first-token from the LLM |
| `tts_ttfb_avg_s` | Mean time-to-first-byte from TTS |
| `eou_delay_avg_s` | Mean delay from end of speech to turn decision |

Copy values from a real session JSON into resume bullets. Do not invent latency or rates.
