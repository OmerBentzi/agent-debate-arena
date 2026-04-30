# Agent Debate Arena

> Ten AI agents argue any question you give them. Watch them debate live in 3D, see who tries to convince whom, who holds firm, who shifts — and get one synthesized answer at the end.

A real-time multi-agent debate platform with a live 3D network visualization, full statistical dashboard, persuasion-event tracking, and per-agent conviction analysis. Powered by [OpenRouter](https://openrouter.ai/) — works with any OpenRouter-compatible model.

---

## What it does

Most LLM "debates" are mush — both sides see valid points and refuse to commit. This tool forces commitment in two directions:

1. **Forces conflict during the debate.** Ten personas with sharply different worldviews (Rust Belt worker, epidemiologist, retired Marine, libertarian engineer, Gen-Z student, etc.) read each other's turns and argue *against* specific points. They name names.
2. **Forces a single answer at the end.** A synthesizer model reads the full transcript and picks ONE side based on argument strength — not vote-counting. With confidence score and rationale.

In between, you get a complete picture of *how* the debate went: who tried to flip whom, who refused to budge and why, what topics dominated, and how the rhetorical balance shifted round-by-round.

## Features

### 🌐 Live 3D debate network (Three.js)

Ten agents arranged on a 3D ring. As the debate streams in:

- **Active speaker** pulses orange and grows with their word count.
- **Live mention beams** — when an agent names another agent in their reply, an arc fires between them in real time.
- **OrbitControls** — drag to rotate, scroll to zoom, double-click to reset.

After the debate completes:

- Nodes recolor by their final stance (red / blue / gray).
- The transient mention arcs are replaced with **persistent persuasion edges**: green = successful persuasion, red = rebuffed, gray = ignored.

### 💬 Token-by-token streaming

Server-Sent Events push tokens straight from OpenRouter to the browser. Each agent's card updates live.

### ✅ Single-answer verdict

Strict-JSON synthesizer call at end-of-debate produces:

- `verdict` — one of two auto-detected positions (e.g. "Trump", "Biden", "React", "Yes")
- `confidence` — 0–100
- `reasoning` — 2–4 sentence rationale

### 🧠 Per-agent conviction analysis

For every agent, the analyst extracts:

- **Final stance** + **key verbatim quote**
- **Conviction score (0–100)** — how firmly they held their position across rounds
- **Held firm?** boolean (≥70 conviction + no real position change)
- **Stance journey** — one-sentence narrative of how their view evolved
- **Conviction reasoning** — *why* they stuck or shifted

### 🎯 Persuasion event tracking

Every direct rebuttal or persuasion attempt is extracted into a chronological table:

| Round | From | → | To | Argument | Outcome |
|-------|------|---|----|----------|---------|
| 1 | Patel | → | Marge | Tariff costs hit the same families they're meant to help | rebuffed |
| 2 | Hayes | → | Carlos | NATO fracture is a worse economic shock than inflation | unclear |

Outcomes: `successful` · `rebuffed` · `ignored` · `unclear`.

### 📊 Full statistics dashboard

- Stance distribution per agent (bar)
- Words per agent, color-coded by stance (horizontal bar)
- Conviction-score leaderboard (horizontal bar — orange = held firm, gray = shifted)
- Persuasion-outcome breakdown (doughnut)
- Topic breakdown — analyst extracts 6–10 dominant themes with weights (doughnut)
- Position mention share by round (line chart)
- Cross-agent mention matrix (10×10 heatmap)
- Per-agent stance + key-quote table
- KPI strip: total words, position mentions, held-firm count

### 🛠 Editable question

Debate any binary or two-position question. The synthesizer auto-detects the two opposing positions from the transcript (Trump vs Biden, React vs Vue, Adopt vs Don't), so the entire dashboard relabels itself.

---

## How it works

```
┌────────────────────────────────────────────────────────────────┐
│  Browser                                                       │
│  ┌────────────────────┐  ┌────────────────────────────────┐   │
│  │ EventSource        │  │ Three.js scene                 │   │
│  │ - cards stream     │  │ - 10 nodes on a ring           │   │
│  │ - live mentions    │──│ - mention beams                │   │
│  └────────────────────┘  │ - post-debate persuasion graph │   │
│  ┌────────────────────────────────────────────────────────┐   │
│  │ Chart.js dashboard (rendered on done)                  │   │
│  └────────────────────────────────────────────────────────┘   │
└────────────────────────────▲───────────────────────────────────┘
                             │ SSE
┌────────────────────────────┴───────────────────────────────────┐
│  Flask /stream                                                 │
│   for each round, for each agent:                              │
│     stream(persona prompt + full transcript so far) → tokens   │
│   on done:                                                     │
│     synthesize_verdict(transcript) → JSON                      │
│       · positions, verdict, confidence, reasoning              │
│       · per-agent stance + conviction analysis                 │
│       · persuasion events                                      │
│       · topic breakdown                                        │
│     compute_stats(transcript, position_a, position_b)          │
│     emit "results" event                                       │
└────────────────────────────▲───────────────────────────────────┘
                             │ OpenAI SDK (stream=True)
                             ▼
                       OpenRouter API
```

### Key design choices

- **Each agent sees the full running transcript**, so later turns can argue *against* specific earlier ones — that's what makes the debate feel real.
- **Two-pass analysis**: agents debate at high temperature (0.9, creative, in-character); the synthesizer runs at low temperature (0.2) with `response_format=json_object` for deterministic structured output.
- **Position-agnostic stats**: the synthesizer detects the two opposing labels from the transcript itself, then `compute_stats` counts those exact strings per round. Same code works for any two-position question.

## Quickstart

**Prerequisites:** Python 3.9+, an [OpenRouter API key](https://openrouter.ai/keys), a modern browser (Three.js requires WebGL).

```bash
git clone https://github.com/OmerBentzi/agent-debate-arena.git
cd agent-debate-arena
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

export OPENROUTER_API_KEY=sk-or-...
./venv/bin/python server.py
```

Open <http://localhost:5050>, edit the question, hit **Start debate**.

### CLI mode (no dashboard)

```bash
./venv/bin/python debate.py --rounds 2 --model anthropic/claude-haiku-4.5
```

## Configuration

| Variable / flag      | Default                          | Description                                |
|----------------------|----------------------------------|--------------------------------------------|
| `OPENROUTER_API_KEY` | *(required)*                     | Your OpenRouter key                        |
| `PORT`               | `5050`                           | Server port                                |
| `?rounds=N`          | `2`                              | Full rounds; total turns = `N × 10`        |
| `?model=ID`          | `anthropic/claude-haiku-4.5`     | Any OpenRouter model id                    |
| `?question=…`        | Trump-vs-Biden default           | Debate prompt                              |

Personas live in [`debate.py`](debate.py) — edit the `AGENTS` list to swap voices, add domain experts, or change the persona count.

## Project structure

```
.
├── debate.py          # AGENTS list, build_messages, CLI entry point
├── server.py          # Flask + SSE + Three.js scene + synthesizer + stats
├── requirements.txt   # openai, flask
├── .gitignore
└── README.md
```

## Sample questions to try

| Question                                                                 | Why it's interesting |
|--------------------------------------------------------------------------|----------------------|
| *Trump or Biden — who should be President of the United States?*         | the default — sharp ideological splits |
| *Should we adopt Kubernetes for our small startup?*                      | tech vs. ops priorities clash |
| *React or Vue for a new SaaS frontend?*                                  | watch the engineers argue ergonomics vs. ecosystem |
| *Is remote-first work better than hybrid for early-stage companies?*     | culture vs. velocity |
| *Should public companies be required to disclose model training data?*   | regulation vs. innovation |
| *Centralized vs. federated identity for a new social platform?*          | security vs. control |

## Customizing personas

Open [`debate.py`](debate.py) and edit the `AGENTS` list. Each persona is one or two lines that fixes voice, worldview, and likely bias. The 3D scene, mention detection, and all dashboard panels adapt automatically to whatever number of agents you define (the layout is sized for ~10).

```python
Agent(
    "Maria — climate engineer",
    "You are a 35-year-old climate engineer at an NGO. You evaluate every "
    "decision through long-term planetary impact. Skeptical of short-term cost "
    "arguments. Speak with conviction, cite physics where relevant.",
),
```

## Security note

`OPENROUTER_API_KEY` is read from the environment only. It is never written to disk, never sent to the browser, and never committed. `.gitignore` excludes `.env`, `venv/`, and other local artifacts.

## License

MIT
