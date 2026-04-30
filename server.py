"""
Live dashboard: 10 agents debate Trump vs Biden in real time.

Flask + SSE server. Streams each agent's response token-by-token, then
computes statistics and asks a synthesizer model for a single verdict.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python server.py            # then open http://localhost:5050
"""

from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
from collections import Counter

from flask import Flask, Response, render_template_string, request
from openai import OpenAI

from debate import AGENTS, QUESTION, build_messages


app = Flask(__name__)


# Short tags used to detect cross-agent mentions. Skip honorifics.
_HONORIFICS = {"Dr.", "Dr", "Mr.", "Mr", "Mrs.", "Mrs", "Ms.", "Ms",
               "Colonel", "Col.", "Reverend", "Rev.", "Captain", "Capt."}


def _short_tag(full_name: str) -> str:
    for word in full_name.split(" "):
        if word and word not in _HONORIFICS and word != "—":
            return word
    return full_name.split(" ")[0]


SHORT_TAGS = [_short_tag(a.name) for a in AGENTS]


INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>10-Agent Debate — live dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0b0d12;
    --panel: #11141c;
    --card: #151821;
    --card-active: #1e2433;
    --border: #232838;
    --border-active: #ff7a45;
    --text: #e7eaf0;
    --muted: #8a93a6;
    --accent: #ff7a45;
    --accent2: #4cc4ff;
    --trump: #e74c3c;
    --biden: #3498db;
    --undecided: #95a5a6;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text);
               font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; }
  header { padding: 16px 22px; border-bottom: 1px solid var(--border);
           display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
  header h1 { margin: 0; font-size: 17px; font-weight: 600; }
  header .q { color: var(--muted); font-size: 13px; }
  header .controls { margin-left: auto; display: flex; gap: 8px; align-items: center; }
  button { background: var(--accent); color: #111; border: 0; padding: 8px 14px;
           border-radius: 6px; font-weight: 600; cursor: pointer; font-size: 14px; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  input[type=number] { background: var(--card); color: var(--text); border: 1px solid var(--border);
                       padding: 6px 8px; border-radius: 6px; width: 60px; font-size: 14px; }
  label { color: var(--muted); font-size: 13px; }
  .status { color: var(--muted); font-size: 13px; margin-left: 10px; }
  .speaker-banner { padding: 9px 22px; background: var(--panel); border-bottom: 1px solid var(--border);
                    color: var(--muted); font-size: 13px; }
  .speaker-banner b { color: var(--accent); }
  .qbar { padding: 12px 22px; background: var(--panel); border-bottom: 1px solid var(--border);
          display: flex; gap: 12px; align-items: flex-start; }
  .qbar label { font-size: 12px; color: var(--muted); text-transform: uppercase;
                letter-spacing: 0.05em; padding-top: 8px; min-width: 70px; }
  .qbar textarea { flex: 1; background: var(--card); color: var(--text); border: 1px solid var(--border);
                   border-radius: 6px; padding: 8px 10px; font-size: 14px; resize: vertical;
                   font-family: inherit; line-height: 1.4; }
  .qbar textarea:focus { outline: none; border-color: var(--accent); }

  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
          gap: 12px; padding: 16px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px;
          padding: 12px 14px; min-height: 170px; transition: border-color 0.15s, background 0.15s;
          display: flex; flex-direction: column; }
  .card.active { border-color: var(--border-active); background: var(--card-active);
                 box-shadow: 0 0 0 1px var(--border-active) inset; }
  .card .name { font-weight: 600; font-size: 13.5px; margin-bottom: 4px; color: var(--accent2); }
  .card.active .name { color: var(--accent); }
  .card .turn { font-size: 11px; color: var(--muted); margin-bottom: 6px;
                display: flex; justify-content: space-between; }
  .card .body { font-size: 13px; line-height: 1.5; white-space: pre-wrap;
                color: var(--text); flex: 1; overflow-y: auto; max-height: 320px; }
  .cursor { display: inline-block; width: 7px; background: var(--accent); height: 13px;
            margin-left: 2px; animation: blink 1s steps(2) infinite; vertical-align: middle; }
  @keyframes blink { 50% { opacity: 0; } }

  /* Results dashboard */
  #results { display: none; padding: 18px; border-top: 2px solid var(--border); }
  #results.show { display: block; }
  .verdict { background: linear-gradient(135deg, var(--panel), var(--card));
             border: 2px solid var(--accent); border-radius: 14px; padding: 22px 26px;
             margin-bottom: 18px; }
  .verdict h2 { margin: 0 0 6px 0; font-size: 14px; color: var(--muted);
                font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase; }
  .verdict .winner { font-size: 36px; font-weight: 700; margin-bottom: 4px; }
  .verdict .winner.trump { color: var(--trump); }
  .verdict .winner.biden { color: var(--biden); }
  .verdict .conf { color: var(--muted); font-size: 14px; margin-bottom: 10px; }
  .verdict .reasoning { font-size: 14px; line-height: 1.6; color: var(--text); }

  .panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
            gap: 14px; margin-bottom: 18px; }
  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
           padding: 14px 16px; }
  .panel h3 { margin: 0 0 12px 0; font-size: 14px; color: var(--accent2);
              font-weight: 600; letter-spacing: 0.02em; }
  .panel.wide { grid-column: 1 / -1; }
  canvas { max-height: 280px; }

  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 500; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }
  td.stance { font-weight: 600; }
  td.stance.trump { color: var(--trump); }
  td.stance.biden { color: var(--biden); }
  td.stance.undecided { color: var(--undecided); }
  td.quote { color: var(--muted); font-style: italic; }

  .heatmap { display: grid; gap: 2px; font-size: 11px; }
  .hm-cell { padding: 6px 4px; text-align: center; border-radius: 3px;
             background: var(--card); color: var(--muted); }
  .hm-row-label, .hm-col-label { padding: 4px; font-size: 10.5px; color: var(--muted); }
  .hm-col-label { writing-mode: vertical-rl; text-align: end; }

  .kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 14px; }
  .kpi { background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
         padding: 12px 14px; }
  .kpi .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
  .kpi .value { font-size: 22px; font-weight: 700; margin-top: 4px; }
</style>
</head>
<body>
<header>
  <h1>10-Agent Debate</h1>
  <div class="controls">
    <label>Rounds <input id="rounds" type="number" value="2" min="1" max="6"></label>
    <button id="start">Start debate</button>
    <span class="status" id="status">idle</span>
  </div>
</header>
<div class="qbar">
  <label for="question">Question</label>
  <textarea id="question" rows="2" placeholder="Ask the agents anything…">{{ question }}</textarea>
</div>
<div class="speaker-banner" id="banner">Press <b>Start debate</b> to begin.</div>
<div class="grid" id="grid"></div>

<section id="results">
  <div class="verdict" id="verdict">
    <h2>Final synthesized verdict</h2>
    <div class="winner" id="winner">—</div>
    <div class="conf" id="conf">—</div>
    <div class="reasoning" id="reasoning">—</div>
  </div>

  <div class="kpis" id="kpis"></div>

  <div class="panels">
    <div class="panel">
      <h3>Stance distribution</h3>
      <canvas id="chartStance"></canvas>
    </div>
    <div class="panel">
      <h3>Words per agent</h3>
      <canvas id="chartWords"></canvas>
    </div>
    <div class="panel">
      <h3>Topic breakdown</h3>
      <canvas id="chartTopics"></canvas>
    </div>
    <div class="panel">
      <h3 id="mentionsTitle">Position mention share by round</h3>
      <canvas id="chartMentions"></canvas>
    </div>

    <div class="panel wide">
      <h3>Per-agent final stance</h3>
      <table id="stanceTable">
        <thead><tr><th>Agent</th><th>Stance</th><th>Words</th><th>Key quote</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>

    <div class="panel wide">
      <h3>Cross-agent mention matrix</h3>
      <p style="color:var(--muted); font-size:12px; margin:0 0 8px 0;">
        Cell (row → col) = number of times the row's agent mentioned the column's agent by name.
      </p>
      <div id="mentionMatrix" class="heatmap"></div>
    </div>
  </div>
</section>

<script>
const AGENTS = {{ agents_json | safe }};
const SHORT_TAGS = {{ short_tags_json | safe }};
const grid = document.getElementById("grid");
const banner = document.getElementById("banner");
const statusEl = document.getElementById("status");
const startBtn = document.getElementById("start");
const roundsInput = document.getElementById("rounds");
const charts = {};

function render() {
  grid.innerHTML = "";
  AGENTS.forEach((a, i) => {
    const card = document.createElement("div");
    card.className = "card";
    card.id = `card-${i}`;
    card.innerHTML = `
      <div class="name">${a.name}</div>
      <div class="turn"><span id="turn-${i}">—</span><span id="state-${i}">waiting</span></div>
      <div class="body" id="body-${i}"></div>
    `;
    grid.appendChild(card);
  });
}
render();

function setActive(i) {
  document.querySelectorAll(".card").forEach(c => c.classList.remove("active"));
  if (i >= 0) document.getElementById(`card-${i}`).classList.add("active");
}

function colorFor(stance, posA, posB) {
  if (!stance || stance === "Undecided") return "#95a5a6";
  if (stance === posA) return "#e74c3c";
  if (stance === posB) return "#3498db";
  return "#95a5a6";
}

function renderResults(payload) {
  const { verdict, stats } = payload;
  const posA = verdict.position_a || "A";
  const posB = verdict.position_b || "B";
  document.getElementById("results").classList.add("show");

  const winnerEl = document.getElementById("winner");
  winnerEl.textContent = verdict.verdict;
  winnerEl.style.color = verdict.verdict === posA ? "#e74c3c"
                       : verdict.verdict === posB ? "#3498db" : "#ff7a45";
  winnerEl.className = "winner";
  document.getElementById("conf").textContent =
    `Confidence: ${verdict.confidence}% · positions: ${posA} vs ${posB}`;
  document.getElementById("reasoning").textContent = verdict.reasoning;

  // KPIs
  const totalWords = stats.words_per_agent.reduce((a,b) => a+b, 0);
  const stanceCounts = { [posA]: 0, [posB]: 0, "Undecided": 0 };
  verdict.agent_stances.forEach(s => {
    const k = (s.stance === posA || s.stance === posB) ? s.stance : "Undecided";
    stanceCounts[k]++;
  });
  const kpis = document.getElementById("kpis");
  kpis.innerHTML = `
    <div class="kpi"><div class="label">Total words</div><div class="value">${totalWords.toLocaleString()}</div></div>
    <div class="kpi"><div class="label">${posA} mentions</div><div class="value" style="color:var(--trump)">${stats.a_mentions}</div></div>
    <div class="kpi"><div class="label">${posB} mentions</div><div class="value" style="color:var(--biden)">${stats.b_mentions}</div></div>
    <div class="kpi"><div class="label">Cross-agent mentions</div><div class="value">${stats.total_cross_mentions}</div></div>
  `;

  // Stance distribution chart
  charts.stance?.destroy();
  charts.stance = new Chart(document.getElementById("chartStance"), {
    type: "bar",
    data: {
      labels: [posA, posB, "Undecided"],
      datasets: [{
        data: [stanceCounts[posA], stanceCounts[posB], stanceCounts.Undecided],
        backgroundColor: ["#e74c3c", "#3498db", "#95a5a6"],
      }],
    },
    options: { plugins: { legend: { display: false } },
               scales: { y: { beginAtZero: true, ticks: { stepSize: 1 } } } },
  });

  // Words per agent
  charts.words?.destroy();
  charts.words = new Chart(document.getElementById("chartWords"), {
    type: "bar",
    data: {
      labels: AGENTS.map(a => a.name.split(" — ")[0]),
      datasets: [{
        data: stats.words_per_agent,
        backgroundColor: AGENTS.map((_, i) => colorFor(verdict.agent_stances[i]?.stance, posA, posB)),
      }],
    },
    options: { indexAxis: "y", plugins: { legend: { display: false } },
               scales: { x: { beginAtZero: true } } },
  });

  // Topic breakdown
  charts.topics?.destroy();
  charts.topics = new Chart(document.getElementById("chartTopics"), {
    type: "doughnut",
    data: {
      labels: verdict.topic_breakdown.map(t => t.topic),
      datasets: [{
        data: verdict.topic_breakdown.map(t => t.weight),
        backgroundColor: ["#ff7a45", "#4cc4ff", "#9b59b6", "#2ecc71", "#f1c40f",
                          "#e67e22", "#1abc9c", "#e84393", "#74b9ff", "#a29bfe"],
      }],
    },
    options: { plugins: { legend: { position: "right", labels: { color: "#e7eaf0", font: { size: 11 } } } } },
  });

  // Mentions per round line
  charts.mentions?.destroy();
  charts.mentions = new Chart(document.getElementById("chartMentions"), {
    type: "line",
    data: {
      labels: stats.rounds.map(r => `Round ${r}`),
      datasets: [
        { label: posA, data: stats.a_per_round, borderColor: "#e74c3c",
          backgroundColor: "#e74c3c33", tension: 0.3, fill: true },
        { label: posB, data: stats.b_per_round, borderColor: "#3498db",
          backgroundColor: "#3498db33", tension: 0.3, fill: true },
      ],
    },
    options: { plugins: { legend: { labels: { color: "#e7eaf0" } } },
               scales: { y: { beginAtZero: true } } },
  });
  document.querySelector("#mentionsTitle").textContent =
    `${posA} vs ${posB} mention share by round`;

  // Stance table
  const tbody = document.querySelector("#stanceTable tbody");
  tbody.innerHTML = "";
  verdict.agent_stances.forEach((s, i) => {
    const tr = document.createElement("tr");
    const stance = s.stance || "Undecided";
    const color = colorFor(stance, posA, posB);
    tr.innerHTML = `
      <td>${AGENTS[i]?.name || s.name}</td>
      <td class="stance" style="color:${color}">${stance}</td>
      <td>${stats.words_per_agent[i]}</td>
      <td class="quote">"${s.key_quote}"</td>
    `;
    tbody.appendChild(tr);
  });

  // Mention matrix heatmap
  const mat = document.getElementById("mentionMatrix");
  mat.innerHTML = "";
  const n = AGENTS.length;
  mat.style.gridTemplateColumns = `120px repeat(${n}, 1fr)`;
  // header row
  mat.appendChild(Object.assign(document.createElement("div"), { className: "hm-cell", textContent: "" }));
  for (let j = 0; j < n; j++) {
    const c = document.createElement("div");
    c.className = "hm-col-label";
    c.textContent = SHORT_TAGS[j];
    mat.appendChild(c);
  }
  // body
  let maxCell = 1;
  for (const row of stats.mention_matrix) for (const v of row) if (v > maxCell) maxCell = v;
  for (let i = 0; i < n; i++) {
    const lab = document.createElement("div");
    lab.className = "hm-row-label";
    lab.textContent = SHORT_TAGS[i];
    mat.appendChild(lab);
    for (let j = 0; j < n; j++) {
      const v = stats.mention_matrix[i][j];
      const cell = document.createElement("div");
      cell.className = "hm-cell";
      cell.textContent = v || "·";
      const intensity = v / maxCell;
      cell.style.background = `rgba(255, 122, 69, ${intensity * 0.85})`;
      if (v > 0) cell.style.color = "#fff";
      mat.appendChild(cell);
    }
  }
}

function startDebate() {
  const rounds = parseInt(roundsInput.value, 10) || 2;
  const question = document.getElementById("question").value.trim();
  if (!question) {
    alert("Please enter a question for the agents to debate.");
    return;
  }
  AGENTS.forEach((_, i) => {
    document.getElementById(`body-${i}`).textContent = "";
    document.getElementById(`turn-${i}`).textContent = "—";
    document.getElementById(`state-${i}`).textContent = "waiting";
  });
  document.getElementById("results").classList.remove("show");
  startBtn.disabled = true;
  statusEl.textContent = "connecting…";
  banner.innerHTML = `Question: <i>${question}</i>`;

  const url = `/stream?rounds=${rounds}&question=${encodeURIComponent(question)}`;
  const es = new EventSource(url);

  es.addEventListener("turn_start", e => {
    const d = JSON.parse(e.data);
    setActive(d.agent);
    document.getElementById(`turn-${d.agent}`).textContent = `Round ${d.round}`;
    document.getElementById(`state-${d.agent}`).textContent = "speaking";
    const body = document.getElementById(`body-${d.agent}`);
    body.textContent = "";
    const cursor = document.createElement("span");
    cursor.className = "cursor";
    cursor.id = `cursor-${d.agent}`;
    body.appendChild(cursor);
    banner.innerHTML = `Speaking: <b>${AGENTS[d.agent].name}</b> · Round ${d.round}`;
    statusEl.textContent = `round ${d.round} · agent ${d.agent + 1}/${AGENTS.length}`;
  });

  es.addEventListener("token", e => {
    const d = JSON.parse(e.data);
    const body = document.getElementById(`body-${d.agent}`);
    const cursor = document.getElementById(`cursor-${d.agent}`);
    if (cursor) cursor.remove();
    body.appendChild(document.createTextNode(d.token));
    const c = document.createElement("span");
    c.className = "cursor"; c.id = `cursor-${d.agent}`;
    body.appendChild(c);
    body.scrollTop = body.scrollHeight;
  });

  es.addEventListener("turn_end", e => {
    const d = JSON.parse(e.data);
    document.getElementById(`state-${d.agent}`).textContent = "done";
    const cursor = document.getElementById(`cursor-${d.agent}`);
    if (cursor) cursor.remove();
  });

  es.addEventListener("synthesizing", () => {
    setActive(-1);
    banner.innerHTML = "Debate complete — synthesizing verdict…";
    statusEl.textContent = "analyzing";
  });

  es.addEventListener("results", e => {
    const payload = JSON.parse(e.data);
    renderResults(payload);
    banner.innerHTML = `Verdict: <b>${payload.verdict.verdict}</b> (${payload.verdict.confidence}% confidence)`;
    statusEl.textContent = "done";
    startBtn.disabled = false;
    es.close();
    document.getElementById("results").scrollIntoView({ behavior: "smooth" });
  });

  es.addEventListener("error_event", e => {
    const d = JSON.parse(e.data);
    banner.innerHTML = `Error: ${d.message}`;
    statusEl.textContent = "error";
    startBtn.disabled = false;
    es.close();
  });

  es.onerror = () => {
    statusEl.textContent = "disconnected";
    startBtn.disabled = false;
    es.close();
  };
}

startBtn.addEventListener("click", startDebate);
</script>
</body>
</html>
"""


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def compute_stats(
    transcript_per_round: list[list[str]],
    position_a: str,
    position_b: str,
) -> dict:
    """Compute statistics from a transcript shaped as [round][agent_idx] = text.

    position_a / position_b are short labels (e.g. "Trump", "Biden") whose
    occurrences are counted per round.
    """
    n = len(AGENTS)
    rounds = list(range(1, len(transcript_per_round) + 1))

    words_per_agent = [0] * n
    a_per_round: list[int] = []
    b_per_round: list[int] = []
    a_total = 0
    b_total = 0
    mention_matrix = [[0] * n for _ in range(n)]

    pat_a = re.compile(rf"\b{re.escape(position_a)}\b", flags=re.IGNORECASE)
    pat_b = re.compile(rf"\b{re.escape(position_b)}\b", flags=re.IGNORECASE)

    for round_texts in transcript_per_round:
        a_round = 0
        b_round = 0
        for i, text in enumerate(round_texts):
            words_per_agent[i] += len(text.split())
            a = len(pat_a.findall(text))
            b = len(pat_b.findall(text))
            a_round += a
            b_round += b
            a_total += a
            b_total += b
            for j, tag in enumerate(SHORT_TAGS):
                if i == j:
                    continue
                if re.search(rf"\b{re.escape(tag)}\b", text):
                    mention_matrix[i][j] += 1
        a_per_round.append(a_round)
        b_per_round.append(b_round)

    total_cross = sum(sum(row) for row in mention_matrix)

    return {
        "rounds": rounds,
        "words_per_agent": words_per_agent,
        "position_a": position_a,
        "position_b": position_b,
        "a_per_round": a_per_round,
        "b_per_round": b_per_round,
        "a_mentions": a_total,
        "b_mentions": b_total,
        "mention_matrix": mention_matrix,
        "total_cross_mentions": total_cross,
    }


SYNTHESIZER_SYSTEM = """You are a neutral debate analyst. You will receive a full transcript of a 10-agent debate on a single question. Your job is to:

1. Identify the TWO primary opposing positions in the debate. Give each a short label (1-3 words, e.g. "Trump", "Biden", "Yes", "Pro-React", "Public option"). Call them position_a and position_b.
2. Pick exactly ONE of those two labels as the verdict — based on the STRONGEST argument made, not on which side has more agents. This is "best argument wins", not a poll.
3. Estimate your confidence (0-100).
4. Write a 2-4 sentence reasoning paragraph explaining the synthesis.
5. Classify each agent's final stance as exactly one of: position_a's label, position_b's label, or "Undecided".
6. For each agent, extract a short verbatim key quote (max 25 words) capturing their core position.
7. Identify 6-10 dominant topics in the debate and assign each a weight (integer 1-100) reflecting how much debate time/energy went to that topic.

Respond with strictly valid JSON, no prose, no markdown, in this schema:
{
  "position_a": "<short label>",
  "position_b": "<short label>",
  "verdict": "<position_a label OR position_b label>",
  "confidence": <int 0-100>,
  "reasoning": "<string>",
  "agent_stances": [
    {"name": "<exact agent name>", "stance": "<position_a label | position_b label | Undecided>", "key_quote": "<short quote>"}
  ],
  "topic_breakdown": [
    {"topic": "<short topic name>", "weight": <int>}
  ]
}
"""


def synthesize_verdict(
    client: OpenAI,
    model: str,
    transcript: list[tuple[str, str]],
    question: str,
) -> dict:
    transcript_text = "\n\n".join(f"{name}: {text}" for name, text in transcript)
    agent_names = [a.name for a in AGENTS]
    user = (
        f"Question: {question}\n\n"
        f"Agents (in order): {json.dumps(agent_names)}\n\n"
        f"Full transcript:\n\n{transcript_text}\n\n"
        "Now produce the JSON verdict per the schema. Use the exact agent names above."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYNTHESIZER_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=2000,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        data = json.loads(match.group(0)) if match else {}

    # Defensive defaults
    data.setdefault("position_a", "Position A")
    data.setdefault("position_b", "Position B")
    data.setdefault("verdict", data["position_a"])
    data.setdefault("confidence", 0)
    data.setdefault("reasoning", "")
    data.setdefault("agent_stances", [])
    data.setdefault("topic_breakdown", [])

    # Reorder agent_stances to match AGENTS order so the UI can index by position.
    by_name = {s.get("name", ""): s for s in data["agent_stances"]}
    aligned = []
    for a in AGENTS:
        s = by_name.get(a.name) or {"name": a.name, "stance": "Undecided", "key_quote": ""}
        aligned.append({
            "name": a.name,
            "stance": s.get("stance", "Undecided"),
            "key_quote": s.get("key_quote", ""),
        })
    data["agent_stances"] = aligned
    return data


@app.route("/")
def index():
    agents_json = json.dumps([{"name": a.name} for a in AGENTS])
    short_tags_json = json.dumps(SHORT_TAGS)
    return render_template_string(
        INDEX_HTML,
        question=QUESTION,
        agents_json=agents_json,
        short_tags_json=short_tags_json,
    )


@app.route("/stream")
def stream():
    rounds = int(request.args.get("rounds", "2"))
    model = request.args.get("model", "anthropic/claude-haiku-4.5")
    question = (request.args.get("question") or "").strip() or QUESTION
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return Response(
            sse("error_event", {"message": "OPENROUTER_API_KEY not set"}),
            mimetype="text/event-stream",
        )

    q: "queue.Queue[str | None]" = queue.Queue()

    def run_debate():
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        transcript: list[tuple[str, str]] = []
        transcript_per_round: list[list[str]] = []
        try:
            for r in range(1, rounds + 1):
                round_texts: list[str] = []
                for idx, agent in enumerate(AGENTS):
                    q.put(sse("turn_start", {"agent": idx, "round": r, "name": agent.name}))
                    messages = build_messages(agent, transcript, question=question)
                    stream_resp = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.9,
                        max_tokens=300,
                        stream=True,
                    )
                    full = []
                    for chunk in stream_resp:
                        delta = chunk.choices[0].delta.content if chunk.choices else None
                        if delta:
                            full.append(delta)
                            q.put(sse("token", {"agent": idx, "token": delta}))
                    text = "".join(full).strip()
                    transcript.append((agent.name, text))
                    round_texts.append(text)
                    q.put(sse("turn_end", {"agent": idx}))
                    time.sleep(0.05)
                transcript_per_round.append(round_texts)

            q.put(sse("synthesizing", {}))
            verdict = synthesize_verdict(client, model, transcript, question)
            stats = compute_stats(
                transcript_per_round,
                verdict.get("position_a", "Position A"),
                verdict.get("position_b", "Position B"),
            )
            q.put(sse("results", {"verdict": verdict, "stats": stats, "question": question}))
        except Exception as e:
            q.put(sse("error_event", {"message": str(e)}))
        finally:
            q.put(None)

    threading.Thread(target=run_debate, daemon=True).start()

    def gen():
        while True:
            item = q.get()
            if item is None:
                return
            yield item

    return Response(
        gen(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    print(f"\nOpen http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
