"""
Agent Debate Arena — live dashboard.

Flask + SSE server. Streams each agent's response token-by-token, drives a
live 3D Three.js scene of the debate network, then computes statistics and
asks a synthesizer model for a single verdict plus persuasion / conviction
analysis.

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
<title>Agent Debate Arena — live</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
  }
}
</script>
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
    --pos-a: #e74c3c;
    --pos-b: #3498db;
    --undecided: #95a5a6;
    --good: #2ecc71;
    --warn: #f1c40f;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text);
               font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; }
  header { padding: 14px 22px; border-bottom: 1px solid var(--border);
           display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
           background: linear-gradient(180deg, var(--panel), var(--bg)); }
  header h1 { margin: 0; font-size: 17px; font-weight: 700; letter-spacing: 0.02em; }
  header h1 .accent { color: var(--accent); }
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

  /* 3D scene */
  #scene3D { width: 100%; height: 460px; background: radial-gradient(ellipse at center, #11151f 0%, #0b0d12 100%);
             border-bottom: 1px solid var(--border); position: relative; overflow: hidden; }
  #scene3D canvas { display: block; }
  .legend { position: absolute; top: 14px; right: 14px; background: rgba(17,20,28,0.88);
            border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px;
            font-size: 12px; max-width: 280px; backdrop-filter: blur(6px); pointer-events: none; }
  .legend .row { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
  .legend .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
  .legend hr { border: 0; border-top: 1px solid var(--border); margin: 8px 0; }
  .legend .small { color: var(--muted); font-size: 10.5px; line-height: 1.4; }
  .scene-hint { position: absolute; bottom: 12px; left: 14px; color: var(--muted); font-size: 11px;
                pointer-events: none; user-select: none; }
  .scene-title { position: absolute; top: 14px; left: 18px; color: var(--accent2); font-size: 12px;
                 letter-spacing: 0.08em; text-transform: uppercase; font-weight: 600; pointer-events: none; }

  /* Agent grid */
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
  th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border);
           vertical-align: top; }
  th { color: var(--muted); font-weight: 500; font-size: 12px; text-transform: uppercase;
       letter-spacing: 0.04em; }
  td.stance { font-weight: 600; }
  td.quote { color: var(--muted); font-style: italic; max-width: 380px; }
  td.outcome { font-weight: 600; }
  td.outcome.successful { color: var(--good); }
  td.outcome.rebuffed { color: var(--pos-a); }
  td.outcome.ignored { color: var(--undecided); }
  td.outcome.unclear { color: var(--warn); }

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

  .conv-bar { display: inline-block; height: 8px; background: var(--card); border-radius: 4px;
              width: 120px; vertical-align: middle; margin-right: 8px; overflow: hidden; }
  .conv-bar > span { display: block; height: 100%; background: var(--accent); }
  .pill { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px;
          font-weight: 600; border: 1px solid currentColor; }
</style>
</head>
<body>
<header>
  <h1><span class="accent">●</span> Agent Debate Arena</h1>
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

<div id="scene3D">
  <div class="scene-title">Live debate network</div>
  <div class="legend">
    <div class="row"><span class="dot" style="background:#4cc4ff"></span> idle agent</div>
    <div class="row"><span class="dot" style="background:#ff7a45"></span> active speaker</div>
    <div class="row"><span class="dot" style="background:#e74c3c"></span> position A (final)</div>
    <div class="row"><span class="dot" style="background:#3498db"></span> position B (final)</div>
    <div class="row"><span class="dot" style="background:#95a5a6"></span> undecided</div>
    <hr>
    <div class="row small"><b style="color:#ff7a45">Live</b>: orange arcs = mentions during a turn.</div>
    <div class="row small"><b style="color:#2ecc71">Final</b>: green = successful persuasion, red = rebuffed, gray = ignored.</div>
  </div>
  <div class="scene-hint">drag to rotate · scroll to zoom · double-click to reset</div>
</div>

<div class="grid" id="grid"></div>

<section id="results">
  <div class="verdict" id="verdict">
    <h2>Synthesized verdict</h2>
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
      <h3>Conviction scores</h3>
      <canvas id="chartConviction"></canvas>
    </div>
    <div class="panel">
      <h3>Persuasion outcomes</h3>
      <canvas id="chartOutcomes"></canvas>
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
      <h3>Persuasion events — who tried to convince whom</h3>
      <p style="color:var(--muted); font-size:12px; margin:0 0 8px 0;">
        Direct addressed challenges and rebuttals extracted by the analyst from the full transcript.
      </p>
      <table id="persuasionTable">
        <thead><tr>
          <th>Round</th><th>From</th><th>→</th><th>To</th>
          <th>Argument</th><th>Outcome</th>
        </tr></thead>
        <tbody></tbody>
      </table>
    </div>

    <div class="panel wide">
      <h3>Conviction leaderboard — who held firm, who shifted, and why</h3>
      <table id="convictionTable">
        <thead><tr>
          <th>Agent</th><th>Final stance</th><th>Conviction</th>
          <th>Held firm?</th><th>Stance journey</th><th>Why</th>
        </tr></thead>
        <tbody></tbody>
      </table>
    </div>

    <div class="panel wide">
      <h3>Per-agent final stance + key quote</h3>
      <table id="stanceTable">
        <thead><tr><th>Agent</th><th>Stance</th><th>Words</th><th>Key quote</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>

    <div class="panel wide">
      <h3>Cross-agent mention matrix</h3>
      <p style="color:var(--muted); font-size:12px; margin:0 0 8px 0;">
        Cell (row → col) = number of turns the row's agent mentioned the column's agent by name.
      </p>
      <div id="mentionMatrix" class="heatmap"></div>
    </div>
  </div>
</section>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const AGENTS = {{ agents_json | safe }};
const SHORT_TAGS = {{ short_tags_json | safe }};
const N = AGENTS.length;

const NODE_BASE = 0x4cc4ff;
const NODE_ACTIVE = 0xff7a45;
const POS_A_HEX = 0xe74c3c;
const POS_B_HEX = 0x3498db;
const UNDECIDED_HEX = 0x95a5a6;
const OUTCOME_COLORS = {
  successful: 0x2ecc71,
  rebuffed:   0xe74c3c,
  ignored:    0x666666,
  unclear:    0xf1c40f,
};

// ===== 3D scene =====
const sceneEl = document.getElementById("scene3D");
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, sceneEl.clientWidth / sceneEl.clientHeight, 0.1, 200);
camera.position.set(0, 5, 13);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio || 1);
renderer.setSize(sceneEl.clientWidth, sceneEl.clientHeight);
renderer.setClearColor(0x000000, 0);
sceneEl.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.minDistance = 6;
controls.maxDistance = 30;
controls.target.set(0, 0, 0);

renderer.domElement.addEventListener("dblclick", () => {
  camera.position.set(0, 5, 13);
  controls.target.set(0, 0, 0);
});

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const dl = new THREE.DirectionalLight(0xffffff, 0.7);
dl.position.set(8, 12, 6);
scene.add(dl);
const accentLight = new THREE.PointLight(0xff7a45, 0.6, 40);
accentLight.position.set(0, 6, 0);
scene.add(accentLight);
const grid = new THREE.GridHelper(24, 24, 0x232838, 0x161a24);
grid.position.y = -1.5;
scene.add(grid);

function makeTextSprite(text) {
  const canvas = document.createElement("canvas");
  canvas.width = 320; canvas.height = 64;
  const ctx = canvas.getContext("2d");
  ctx.font = "bold 28px -apple-system, BlinkMacSystemFont, sans-serif";
  ctx.fillStyle = "#e7eaf0";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.shadowColor = "rgba(0,0,0,0.7)";
  ctx.shadowBlur = 6;
  ctx.fillText(text, 160, 32);
  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
  sprite.scale.set(2.8, 0.56, 1);
  return sprite;
}

const RADIUS = 5.5;
const nodes = AGENTS.map((a, i) => {
  const angle = (i / N) * Math.PI * 2;
  const x = Math.cos(angle) * RADIUS;
  const z = Math.sin(angle) * RADIUS;
  const y = Math.sin(angle * 2) * 0.7;
  const sphere = new THREE.Mesh(
    new THREE.SphereGeometry(0.45, 32, 32),
    new THREE.MeshStandardMaterial({
      color: NODE_BASE,
      emissive: 0x0a2230,
      metalness: 0.35,
      roughness: 0.4,
    }),
  );
  sphere.position.set(x, y, z);
  scene.add(sphere);
  const label = makeTextSprite(a.name.split(" — ")[0]);
  label.position.set(x, y + 1.0, z);
  scene.add(label);
  return { sphere, label, position: new THREE.Vector3(x, y, z), wordCount: 0 };
});

const beams = [];
const persistentEdges = [];
let activeAgent = -1;

function fireBeam(fromIdx, toIdx, colorHex = 0xff7a45, lifetime = 1.0) {
  if (fromIdx === toIdx) return;
  const a = nodes[fromIdx].position;
  const b = nodes[toIdx].position;
  const mid = a.clone().add(b).multiplyScalar(0.5);
  mid.y += 2.2;
  const curve = new THREE.QuadraticBezierCurve3(a, mid, b);
  const geom = new THREE.BufferGeometry().setFromPoints(curve.getPoints(40));
  const mat = new THREE.LineBasicMaterial({ color: colorHex, transparent: true, opacity: 0.95 });
  const line = new THREE.Line(geom, mat);
  scene.add(line);
  beams.push({ line, mat, life: lifetime });
}

function drawPersistentEdge(fromIdx, toIdx, colorHex, opacity = 0.55, height = 1.4) {
  if (fromIdx === toIdx) return;
  const a = nodes[fromIdx].position;
  const b = nodes[toIdx].position;
  const mid = a.clone().add(b).multiplyScalar(0.5);
  mid.y += height;
  const curve = new THREE.QuadraticBezierCurve3(a, mid, b);
  const geom = new THREE.BufferGeometry().setFromPoints(curve.getPoints(36));
  const mat = new THREE.LineBasicMaterial({ color: colorHex, transparent: true, opacity });
  const line = new THREE.Line(geom, mat);
  scene.add(line);
  persistentEdges.push(line);
}

function clearPersistentEdges() {
  persistentEdges.forEach((l) => {
    scene.remove(l);
    l.geometry.dispose();
    l.material.dispose();
  });
  persistentEdges.length = 0;
}

function setNodeColor(idx, hex, emissiveHex) {
  const m = nodes[idx].sphere.material;
  m.color.setHex(hex);
  m.emissive.setHex(emissiveHex ?? hex);
  m.emissiveIntensity = 0.18;
}

function setActiveAgent(idx) {
  activeAgent = idx;
}

function nodeScaleFor(words) { return 1 + Math.min(1.2, words / 220); }

function setNodeWords(idx, words) {
  nodes[idx].wordCount = words;
  if (idx !== activeAgent) nodes[idx].sphere.scale.setScalar(nodeScaleFor(words));
}

function animate() {
  requestAnimationFrame(animate);
  for (let i = beams.length - 1; i >= 0; i--) {
    const b = beams[i];
    b.life -= 0.014;
    b.mat.opacity = Math.max(0, b.life);
    if (b.life <= 0) {
      scene.remove(b.line);
      b.line.geometry.dispose();
      b.mat.dispose();
      beams.splice(i, 1);
    }
  }
  if (activeAgent >= 0) {
    const t = Date.now() * 0.005;
    const base = nodeScaleFor(nodes[activeAgent].wordCount);
    nodes[activeAgent].sphere.scale.setScalar(base + 0.18 * Math.sin(t));
  }
  controls.update();
  renderer.render(scene, camera);
}
animate();

window.addEventListener("resize", () => {
  const w = sceneEl.clientWidth;
  const h = sceneEl.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
});

function resetScene() {
  clearPersistentEdges();
  nodes.forEach((n, i) => {
    setNodeColor(i, NODE_BASE, 0x0a2230);
    n.sphere.scale.setScalar(1);
    n.wordCount = 0;
  });
  beams.forEach((b) => { scene.remove(b.line); b.line.geometry.dispose(); b.mat.dispose(); });
  beams.length = 0;
  activeAgent = -1;
}

// ===== Agent grid =====
const gridEl = document.getElementById("grid");
const banner = document.getElementById("banner");
const statusEl = document.getElementById("status");
const startBtn = document.getElementById("start");
const roundsInput = document.getElementById("rounds");
const charts = {};

function renderGrid() {
  gridEl.innerHTML = "";
  AGENTS.forEach((a, i) => {
    const card = document.createElement("div");
    card.className = "card";
    card.id = `card-${i}`;
    card.innerHTML = `
      <div class="name">${a.name}</div>
      <div class="turn"><span id="turn-${i}">—</span><span id="state-${i}">waiting</span></div>
      <div class="body" id="body-${i}"></div>
    `;
    gridEl.appendChild(card);
  });
}
renderGrid();

function setActiveCard(i) {
  document.querySelectorAll(".card").forEach((c) => c.classList.remove("active"));
  if (i >= 0) document.getElementById(`card-${i}`).classList.add("active");
}

function colorForStance(stance, posA, posB) {
  if (!stance || stance === "Undecided") return "#95a5a6";
  if (stance === posA) return "#e74c3c";
  if (stance === posB) return "#3498db";
  return "#95a5a6";
}
function hexForStance(stance, posA, posB) {
  if (!stance || stance === "Undecided") return UNDECIDED_HEX;
  if (stance === posA) return POS_A_HEX;
  if (stance === posB) return POS_B_HEX;
  return UNDECIDED_HEX;
}

// ===== Render results dashboard =====
function renderResults(payload) {
  const { verdict, stats } = payload;
  const posA = verdict.position_a || "A";
  const posB = verdict.position_b || "B";
  document.getElementById("results").classList.add("show");

  const winnerEl = document.getElementById("winner");
  winnerEl.textContent = verdict.verdict;
  winnerEl.style.color = verdict.verdict === posA ? "#e74c3c"
                       : verdict.verdict === posB ? "#3498db" : "#ff7a45";
  document.getElementById("conf").textContent =
    `Confidence: ${verdict.confidence}% · positions: ${posA} vs ${posB}`;
  document.getElementById("reasoning").textContent = verdict.reasoning;

  // KPIs
  const totalWords = stats.words_per_agent.reduce((x, y) => x + y, 0);
  const stanceCounts = { [posA]: 0, [posB]: 0, "Undecided": 0 };
  verdict.agent_stances.forEach((s) => {
    const k = (s.stance === posA || s.stance === posB) ? s.stance : "Undecided";
    stanceCounts[k]++;
  });
  const heldFirm = verdict.agent_stances.filter((s) => s.held_firm).length;
  document.getElementById("kpis").innerHTML = `
    <div class="kpi"><div class="label">Total words</div><div class="value">${totalWords.toLocaleString()}</div></div>
    <div class="kpi"><div class="label">${posA} mentions</div><div class="value" style="color:var(--pos-a)">${stats.a_mentions}</div></div>
    <div class="kpi"><div class="label">${posB} mentions</div><div class="value" style="color:var(--pos-b)">${stats.b_mentions}</div></div>
    <div class="kpi"><div class="label">Held firm / total</div><div class="value">${heldFirm} / ${N}</div></div>
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

  // Words per agent (color by stance)
  charts.words?.destroy();
  charts.words = new Chart(document.getElementById("chartWords"), {
    type: "bar",
    data: {
      labels: AGENTS.map((a) => a.name.split(" — ")[0]),
      datasets: [{
        data: stats.words_per_agent,
        backgroundColor: AGENTS.map((_, i) => colorForStance(verdict.agent_stances[i]?.stance, posA, posB)),
      }],
    },
    options: { indexAxis: "y", plugins: { legend: { display: false } },
               scales: { x: { beginAtZero: true } } },
  });

  // Conviction scores chart
  charts.conv?.destroy();
  charts.conv = new Chart(document.getElementById("chartConviction"), {
    type: "bar",
    data: {
      labels: AGENTS.map((a) => a.name.split(" — ")[0]),
      datasets: [{
        data: verdict.agent_stances.map((s) => s.conviction_score ?? 50),
        backgroundColor: verdict.agent_stances.map((s) => s.held_firm ? "#ff7a45" : "#8a93a6"),
      }],
    },
    options: { indexAxis: "y", plugins: { legend: { display: false } },
               scales: { x: { beginAtZero: true, max: 100 } } },
  });

  // Persuasion outcomes
  const outcomeCounts = { successful: 0, rebuffed: 0, ignored: 0, unclear: 0 };
  verdict.persuasion_events.forEach((e) => {
    const k = e.outcome in outcomeCounts ? e.outcome : "unclear";
    outcomeCounts[k]++;
  });
  charts.outcomes?.destroy();
  charts.outcomes = new Chart(document.getElementById("chartOutcomes"), {
    type: "doughnut",
    data: {
      labels: ["Successful", "Rebuffed", "Ignored", "Unclear"],
      datasets: [{
        data: [outcomeCounts.successful, outcomeCounts.rebuffed, outcomeCounts.ignored, outcomeCounts.unclear],
        backgroundColor: ["#2ecc71", "#e74c3c", "#666666", "#f1c40f"],
      }],
    },
    options: { plugins: { legend: { position: "right", labels: { color: "#e7eaf0", font: { size: 11 } } } } },
  });

  // Topics
  charts.topics?.destroy();
  charts.topics = new Chart(document.getElementById("chartTopics"), {
    type: "doughnut",
    data: {
      labels: verdict.topic_breakdown.map((t) => t.topic),
      datasets: [{
        data: verdict.topic_breakdown.map((t) => t.weight),
        backgroundColor: ["#ff7a45", "#4cc4ff", "#9b59b6", "#2ecc71", "#f1c40f",
                          "#e67e22", "#1abc9c", "#e84393", "#74b9ff", "#a29bfe"],
      }],
    },
    options: { plugins: { legend: { position: "right", labels: { color: "#e7eaf0", font: { size: 11 } } } } },
  });

  // Mentions per round
  charts.mentions?.destroy();
  charts.mentions = new Chart(document.getElementById("chartMentions"), {
    type: "line",
    data: {
      labels: stats.rounds.map((r) => `Round ${r}`),
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
  document.querySelector("#mentionsTitle").textContent = `${posA} vs ${posB} mention share by round`;

  // Persuasion table
  const ptbody = document.querySelector("#persuasionTable tbody");
  ptbody.innerHTML = "";
  if (!verdict.persuasion_events.length) {
    ptbody.innerHTML = `<tr><td colspan="6" style="color:var(--muted)">No persuasion events extracted.</td></tr>`;
  } else {
    [...verdict.persuasion_events]
      .sort((a, b) => a.round - b.round)
      .forEach((e) => {
        const tr = document.createElement("tr");
        const oc = (e.outcome || "unclear").toLowerCase();
        tr.innerHTML = `
          <td>${e.round}</td>
          <td>${shortName(e.from_agent)}</td>
          <td style="color:var(--muted)">→</td>
          <td>${shortName(e.to_agent)}</td>
          <td>${escapeHtml(e.argument)}</td>
          <td class="outcome ${oc}">${e.outcome}</td>
        `;
        ptbody.appendChild(tr);
      });
  }

  // Conviction leaderboard
  const ctbody = document.querySelector("#convictionTable tbody");
  ctbody.innerHTML = "";
  [...verdict.agent_stances]
    .map((s, i) => ({ ...s, _i: i }))
    .sort((a, b) => (b.conviction_score ?? 0) - (a.conviction_score ?? 0))
    .forEach((s) => {
      const tr = document.createElement("tr");
      const color = colorForStance(s.stance, posA, posB);
      const cs = s.conviction_score ?? 50;
      const firm = s.held_firm
        ? `<span class="pill" style="color:#ff7a45">held firm</span>`
        : `<span class="pill" style="color:var(--muted)">shifted</span>`;
      tr.innerHTML = `
        <td>${AGENTS[s._i].name}</td>
        <td class="stance" style="color:${color}">${s.stance}</td>
        <td><span class="conv-bar"><span style="width:${cs}%;background:${color}"></span></span>${cs}</td>
        <td>${firm}</td>
        <td>${escapeHtml(s.stance_journey || "—")}</td>
        <td style="color:var(--muted)">${escapeHtml(s.conviction_reasoning || "—")}</td>
      `;
      ctbody.appendChild(tr);
    });

  // Stance / quote table
  const stbody = document.querySelector("#stanceTable tbody");
  stbody.innerHTML = "";
  verdict.agent_stances.forEach((s, i) => {
    const tr = document.createElement("tr");
    const color = colorForStance(s.stance, posA, posB);
    tr.innerHTML = `
      <td>${AGENTS[i]?.name || s.name}</td>
      <td class="stance" style="color:${color}">${s.stance}</td>
      <td>${stats.words_per_agent[i]}</td>
      <td class="quote">"${escapeHtml(s.key_quote || "")}"</td>
    `;
    stbody.appendChild(tr);
  });

  // Mention matrix heatmap
  const mat = document.getElementById("mentionMatrix");
  mat.innerHTML = "";
  mat.style.gridTemplateColumns = `120px repeat(${N}, 1fr)`;
  mat.appendChild(Object.assign(document.createElement("div"), { className: "hm-cell", textContent: "" }));
  for (let j = 0; j < N; j++) {
    const c = document.createElement("div");
    c.className = "hm-col-label";
    c.textContent = SHORT_TAGS[j];
    mat.appendChild(c);
  }
  let maxCell = 1;
  for (const row of stats.mention_matrix) for (const v of row) if (v > maxCell) maxCell = v;
  for (let i = 0; i < N; i++) {
    const lab = document.createElement("div");
    lab.className = "hm-row-label";
    lab.textContent = SHORT_TAGS[i];
    mat.appendChild(lab);
    for (let j = 0; j < N; j++) {
      const v = stats.mention_matrix[i][j];
      const cell = document.createElement("div");
      cell.className = "hm-cell";
      cell.textContent = v || "·";
      cell.style.background = `rgba(255, 122, 69, ${(v / maxCell) * 0.85})`;
      if (v > 0) cell.style.color = "#fff";
      mat.appendChild(cell);
    }
  }

  // Update 3D scene with final state
  clearPersistentEdges();
  verdict.agent_stances.forEach((s, i) => {
    const hex = hexForStance(s.stance, posA, posB);
    setNodeColor(i, hex);
    nodes[i].sphere.scale.setScalar(nodeScaleFor(stats.words_per_agent[i] || 0));
  });
  verdict.persuasion_events.forEach((e) => {
    const hex = OUTCOME_COLORS[e.outcome] ?? OUTCOME_COLORS.unclear;
    drawPersistentEdge(e.from_idx, e.to_idx, hex, 0.55, 1.4);
  });
}

function shortName(full) {
  if (!full) return "?";
  return full.split(" — ")[0];
}
function escapeHtml(s) {
  return (s || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// ===== SSE wiring =====
let currentTurn = { agent: -1, text: "", seen: new Set() };

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
  resetScene();
  startBtn.disabled = true;
  statusEl.textContent = "connecting…";
  banner.innerHTML = `Question: <i>${escapeHtml(question)}</i>`;

  const url = `/stream?rounds=${rounds}&question=${encodeURIComponent(question)}`;
  const es = new EventSource(url);

  es.addEventListener("turn_start", (e) => {
    const d = JSON.parse(e.data);
    currentTurn = { agent: d.agent, text: "", seen: new Set() };
    setActiveCard(d.agent);
    setActiveAgent(d.agent);
    setNodeColor(d.agent, NODE_ACTIVE);
    document.getElementById(`turn-${d.agent}`).textContent = `Round ${d.round}`;
    document.getElementById(`state-${d.agent}`).textContent = "speaking";
    const body = document.getElementById(`body-${d.agent}`);
    body.textContent = "";
    const cursor = document.createElement("span");
    cursor.className = "cursor";
    cursor.id = `cursor-${d.agent}`;
    body.appendChild(cursor);
    banner.innerHTML = `Speaking: <b>${AGENTS[d.agent].name}</b> · Round ${d.round}`;
    statusEl.textContent = `round ${d.round} · agent ${d.agent + 1}/${N}`;
  });

  es.addEventListener("token", (e) => {
    const d = JSON.parse(e.data);
    const body = document.getElementById(`body-${d.agent}`);
    const cursor = document.getElementById(`cursor-${d.agent}`);
    if (cursor) cursor.remove();
    body.appendChild(document.createTextNode(d.token));
    const c = document.createElement("span");
    c.className = "cursor"; c.id = `cursor-${d.agent}`;
    body.appendChild(c);
    body.scrollTop = body.scrollHeight;

    // 3D mention beam (live): detect newly mentioned agents in accumulated text
    if (d.agent === currentTurn.agent) {
      currentTurn.text += d.token;
      SHORT_TAGS.forEach((tag, j) => {
        if (j === currentTurn.agent || currentTurn.seen.has(j)) return;
        if (new RegExp(`\\b${tag}\\b`).test(currentTurn.text)) {
          currentTurn.seen.add(j);
          fireBeam(currentTurn.agent, j, NODE_ACTIVE, 1.0);
        }
      });
    }
  });

  es.addEventListener("turn_end", (e) => {
    const d = JSON.parse(e.data);
    document.getElementById(`state-${d.agent}`).textContent = "done";
    const cursor = document.getElementById(`cursor-${d.agent}`);
    if (cursor) cursor.remove();
    // Reset color to base; size by word count
    const words = (document.getElementById(`body-${d.agent}`).textContent || "").trim().split(/\s+/).filter(Boolean).length;
    setNodeColor(d.agent, NODE_BASE, 0x0a2230);
    setNodeWords(d.agent, words);
    setActiveAgent(-1);
  });

  es.addEventListener("synthesizing", () => {
    setActiveCard(-1);
    setActiveAgent(-1);
    banner.innerHTML = "Debate complete — synthesizing verdict, persuasion events, and conviction analysis…";
    statusEl.textContent = "analyzing";
  });

  es.addEventListener("results", (e) => {
    const payload = JSON.parse(e.data);
    renderResults(payload);
    banner.innerHTML = `Verdict: <b>${escapeHtml(payload.verdict.verdict)}</b> (${payload.verdict.confidence}% confidence)`;
    statusEl.textContent = "done";
    startBtn.disabled = false;
    es.close();
    document.getElementById("results").scrollIntoView({ behavior: "smooth" });
  });

  es.addEventListener("error_event", (e) => {
    const d = JSON.parse(e.data);
    banner.innerHTML = `Error: ${escapeHtml(d.message)}`;
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
    """Compute statistics from a transcript shaped as [round][agent_idx] = text."""
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


SYNTHESIZER_SYSTEM = """You are a neutral debate analyst. You will receive a full transcript of a multi-round, 10-agent debate on a single question. Each turn is labeled with the agent's name. Your job is to produce a deep analysis as strict JSON.

You must:

1. Identify the TWO primary opposing positions. Give each a short label (1-3 words, e.g. "Trump", "Biden", "Yes", "Pro-React"). Call them position_a and position_b.
2. Pick exactly ONE of those labels as the verdict — based on the STRONGEST argument made, not on which side has more agents. "Best argument wins", not a poll.
3. Estimate your confidence (0-100).
4. Write a 2-4 sentence reasoning paragraph explaining the synthesis.
5. For EACH agent (use exact names), produce:
   - stance: position_a label, position_b label, or "Undecided" (final stance at end of debate)
   - key_quote: short verbatim quote (max 25 words)
   - conviction_score: 0-100. 100 = held the same position with full force across all rounds. 0 = flipped position. ~50 = softened or hedged.
   - held_firm: boolean — true if conviction_score >= 70 AND no real position change.
   - stance_journey: one short sentence describing how their position evolved across rounds (e.g. "Held firm on Trump throughout, doubling down in round 2 after pushback from Jamal" or "Started leaning Biden, shifted toward Undecided in round 3 due to economic concerns").
   - conviction_reasoning: one short sentence on WHY they held firm or shifted (e.g. "Personal stake in manufacturing jobs made tariffs non-negotiable" or "Acknowledged Aisha's climate evidence and conceded ground").
6. Extract PERSUASION EVENTS — every clear instance where one agent directly tried to convince another agent (or sharply challenged their position):
   - from_agent: exact agent name (the persuader)
   - to_agent: exact agent name (the target — the agent they addressed or rebutted)
   - round: integer round number
   - argument: one short sentence (max 25 words) summarizing the persuasion attempt or rebuttal
   - outcome: one of "successful" (target acknowledged or shifted), "rebuffed" (target pushed back or refused), "ignored" (target did not respond), or "unclear"
   Aim for 5-20 events covering the major exchanges. Do not invent events — only include attempts that actually happened in the transcript.
7. Identify 6-10 dominant topics with weights (integer 1-100) reflecting debate time/energy.

Respond with strictly valid JSON ONLY, no prose, no markdown fences:
{
  "position_a": "<label>",
  "position_b": "<label>",
  "verdict": "<one of position_a or position_b>",
  "confidence": <int>,
  "reasoning": "<string>",
  "agent_stances": [
    {
      "name": "<exact agent name>",
      "stance": "<position_a | position_b | Undecided>",
      "key_quote": "<quote>",
      "conviction_score": <int 0-100>,
      "held_firm": <bool>,
      "stance_journey": "<short sentence>",
      "conviction_reasoning": "<short sentence>"
    }
  ],
  "persuasion_events": [
    {
      "from_agent": "<name>",
      "to_agent": "<name>",
      "round": <int>,
      "argument": "<short sentence>",
      "outcome": "successful | rebuffed | ignored | unclear"
    }
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
        max_tokens=4000,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        data = json.loads(match.group(0)) if match else {}

    data.setdefault("position_a", "Position A")
    data.setdefault("position_b", "Position B")
    data.setdefault("verdict", data["position_a"])
    data.setdefault("confidence", 0)
    data.setdefault("reasoning", "")
    data.setdefault("agent_stances", [])
    data.setdefault("topic_breakdown", [])
    data.setdefault("persuasion_events", [])

    by_name = {s.get("name", ""): s for s in data["agent_stances"]}
    aligned = []
    for a in AGENTS:
        s = by_name.get(a.name) or {}
        aligned.append({
            "name": a.name,
            "stance": s.get("stance", "Undecided"),
            "key_quote": s.get("key_quote", ""),
            "conviction_score": int(s.get("conviction_score", 50) or 50),
            "held_firm": bool(s.get("held_firm", False)),
            "stance_journey": s.get("stance_journey", ""),
            "conviction_reasoning": s.get("conviction_reasoning", ""),
        })
    data["agent_stances"] = aligned

    name_to_idx = {a.name: i for i, a in enumerate(AGENTS)}
    cleaned_events = []
    for ev in data.get("persuasion_events", []):
        f = ev.get("from_agent", "")
        t = ev.get("to_agent", "")
        fi = name_to_idx.get(f)
        ti = name_to_idx.get(t)
        if fi is None or ti is None or fi == ti:
            continue
        cleaned_events.append({
            "from_agent": f,
            "to_agent": t,
            "from_idx": fi,
            "to_idx": ti,
            "round": int(ev.get("round", 1) or 1),
            "argument": ev.get("argument", ""),
            "outcome": ev.get("outcome", "unclear"),
        })
    data["persuasion_events"] = cleaned_events
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
