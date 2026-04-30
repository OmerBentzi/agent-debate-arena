# Agent Debate Arena

A live multi-agent debate dashboard. Ten distinct AI personas argue any question you give them in real time, then a synthesizer model reads the full transcript and delivers a single verdict with confidence, reasoning, and a full statistical breakdown.

Built on top of [OpenRouter](https://openrouter.ai/) — point it at any OpenRouter-compatible model.

---

## Why

LLM debates usually devolve into both sides "seeing valid points" and refusing to commit. This tool forces a decision: ten agents with sharply different worldviews argue, then an analyst model synthesizes one answer, scoring confidence and showing exactly how it got there.

## Features

- **10 distinct personas** — Rust Belt worker, epidemiologist, retired Marine, libertarian engineer, Gen-Z student, and six more. Each has a fixed worldview and writing voice.
- **Live token-by-token streaming** — Server-Sent Events push tokens straight from OpenRouter to the browser. Active speaker is highlighted, response types in as it's generated.
- **Editable question** — debate any binary or multi-position question, not just the default. The synthesizer auto-detects the two opposing positions from the transcript.
- **Single-answer verdict** — at the end, an analyst model picks ONE answer based on argument strength (not a vote count), with confidence (0–100) and a 2–4 sentence rationale.
- **Full statistics dashboard**:
  - Stance distribution per agent (bar chart)
  - Words spoken per agent, color-coded by stance (horizontal bar)
  - Topic breakdown — the analyst extracts 6–10 dominant themes with weights (doughnut)
  - Mention share by round — how often each position name appeared per round (line chart)
  - Cross-agent mention heatmap — who responded to whom (10×10 grid)
  - Per-agent stance table with key verbatim quote
- **CLI mode** — same debate, terminal output, no server.

## How it works

```
        ┌──────────────────────────────────────────────────┐
        │  Browser (EventSource)                           │
        │  - 10 agent cards, live token streams            │
        │  - Charts.js dashboard rendered on done          │
        └──────────────▲───────────────────────────────────┘
                       │ SSE
        ┌──────────────┴───────────────────────────────────┐
        │  Flask /stream                                   │
        │   for each round:                                │
        │     for each agent:                              │
        │       stream(persona prompt + transcript so far) │
        │   synthesize_verdict(full transcript)            │
        │   compute_stats(transcript, position_a, b)       │
        │   emit "results" event                           │
        └──────────────▲───────────────────────────────────┘
                       │ OpenAI SDK
                       ▼
                  OpenRouter API
```

Each agent sees the full running transcript, so later turns can argue *against* specific earlier ones. The synthesizer is a separate, low-temperature call with a strict JSON schema (response_format=json_object).

## Quickstart

**Prerequisites:** Python 3.9+, an [OpenRouter API key](https://openrouter.ai/keys).

```bash
git clone https://github.com/OmerBentzi/agent-debate-arena.git
cd agent-debate-arena
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

export OPENROUTER_API_KEY=sk-or-...
./venv/bin/python server.py
```

Open <http://localhost:5050>, edit the question if you like, hit **Start debate**.

### CLI mode

```bash
export OPENROUTER_API_KEY=sk-or-...
./venv/bin/python debate.py --rounds 2 --model anthropic/claude-haiku-4.5
```

## Configuration

| Variable / flag        | Default                          | Description                                      |
|------------------------|----------------------------------|--------------------------------------------------|
| `OPENROUTER_API_KEY`   | *(required)*                     | Your OpenRouter key                              |
| `PORT`                 | `5050`                           | Server port                                      |
| `?rounds=N`            | `2`                              | Full rounds; total turns = `N × 10`              |
| `?model=ID`            | `anthropic/claude-haiku-4.5`     | Any OpenRouter model id                          |
| `?question=…`          | Trump-vs-Biden default           | Debate prompt                                    |

Personas live in [`debate.py`](debate.py) — edit the `AGENTS` list to swap voices, add domain experts, or change the persona count.

## Project structure

```
.
├── debate.py          # AGENTS list, build_messages, CLI entry point
├── server.py          # Flask + SSE + synthesizer + stats + embedded HTML/JS
├── requirements.txt   # openai, flask
├── .gitignore
└── README.md
```

## Sample questions to try

- *Trump or Biden — who should be President of the United States?* (default)
- *Should we adopt Kubernetes for our small startup?*
- *React or Vue for a new SaaS frontend?*
- *Is remote-first work better than hybrid for early-stage companies?*
- *Should public companies be required to disclose model training data?*

## Security note

`OPENROUTER_API_KEY` is read from the environment only. It's never written to disk, never sent to the browser, and never committed. `.gitignore` excludes `.env`, `venv/`, and other local artifacts.

## License

MIT
