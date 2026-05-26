"""
demo/app.py - Three animated Chip demos in one file.

Run:
    python demo/app.py

Then open:
    http://localhost:8080/brain      - Real-time brain visualizer
    http://localhost:8080/arena      - Survival arena with training
    http://localhost:8080/voice      - Voice assistant (Chip + LLM + TTS/STT)

Dependencies (beyond chip-brain):
    pip install flask edge-tts

The voice demo uses your local LM Studio at http://10.0.0.20:1234/v1
for language generation. Chip computes 9 cognitive factors in <100ms
and shapes the LLM's response accordingly.

The brain demo uses Microsoft's free Edge neural voices via edge-tts
for narration (requires internet, no API key).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Make the project importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from flask import Flask, Response, request, jsonify

from brain import ChipBrain

# ---------------------------------------------------------------------------
# Shared brain instance
# ---------------------------------------------------------------------------

_brain: Optional[ChipBrain] = None
_brain_lock = threading.Lock()


def get_brain() -> ChipBrain:
    global _brain
    if _brain is None:
        with _brain_lock:
            if _brain is None:
                _brain = ChipBrain(config={
                    "save_every": 100,
                    "inner_speech_every": 3,
                }).boot()
    return _brain


# ---------------------------------------------------------------------------
# LLM integration (for voice demo)
# ---------------------------------------------------------------------------

LLM_URL = "http://10.0.0.20:1234/v1/chat/completions"
LLM_MODEL = "huihui-gemma-4-e4b-it-abliterated"


def query_llm(system_prompt: str, user_message: str, max_tokens: int = 150) -> str:
    """Call the local LM Studio endpoint."""
    import urllib.request
    payload = json.dumps({
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }).encode("utf-8")
    req = urllib.request.Request(
        LLM_URL, data=payload, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[LLM unavailable: {e}]"


def build_system_prompt(brain: ChipBrain) -> str:
    """
    Compose a system prompt from Chip's 9 cognitive factors.
    This is the data Chip feeds the LLM to shape its response.
    """
    mood, _ = brain.emotions.current_mood()
    confidence = brain.meta.mean_confidence()
    goal = brain.goal_stack.current_goal()
    goal_name = goal.name if goal else "none"
    drives = brain.homeostasis.status()
    concepts = brain.inner_speech.recent(1)
    thought = concepts[0].text if concepts else ""
    novelty = brain.habituation._last_novelty
    strain = float(brain.homeostasis.strain().item())

    # Recalled knowledge (from hippocampal retrieval)
    recall_slots = [s for s in brain.working_mem._slots if s.source_tag == "hippocampus_recall"]
    recalled_count = len(recall_slots)

    return f"""You are Chip, a cognitive AI assistant. Respond naturally and concisely.

INTERNAL STATE (use this to shape tone and content):
- Mood: {mood}
- Confidence: {confidence:.0%} {"(speak assertively)" if confidence > 0.6 else "(hedge, express uncertainty)"}
- Current goal: {goal_name}
- Drives: energy={drives.get('energy',0):.0%}, curiosity={drives.get('curiosity',0):.0%}, safety={drives.get('safety',0):.0%}
- Recent thought: "{thought}"
- Novelty of input: {novelty:.0%} {"(this is new to me)" if novelty > 0.5 else "(I've heard this before)"}
- Homeostatic strain: {strain:.2f} {"(I feel balanced)" if strain < 0.3 else "(something feels off)"}
- Recalled memories: {recalled_count} relevant episodes

PERSONALITY RULES:
- If curiosity is high, ask follow-up questions
- If energy is low, be brief
- If novelty is low (habituated), acknowledge you've discussed this before
- If confidence is low, express that honestly
- If a contradiction was detected, mention your uncertainty"""


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)


# ===== BRAIN DASHBOARD =====

BRAIN_HTML = """<!DOCTYPE html>
<html><head><title>Chip Brain Dashboard</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#030308; color:#e0e0e0; font-family:'Courier New',monospace; overflow:hidden; height:100vh; display:flex; }

/* Cosmic background */
body::before {
    content:''; position:fixed; inset:0; z-index:-1;
    background: radial-gradient(ellipse at 50% 50%, #0a0a1a 0%, #030308 70%);
}
body::after {
    content:''; position:fixed; inset:0; z-index:-1; opacity:0.3;
    background-image: radial-gradient(1px 1px at 20px 30px, #fff, transparent),
                      radial-gradient(1px 1px at 40px 70px, #aaf, transparent),
                      radial-gradient(1px 1px at 100px 20px, #fff, transparent),
                      radial-gradient(1px 1px at 180px 90px, #aaf, transparent),
                      radial-gradient(1px 1px at 250px 60px, #fff, transparent),
                      radial-gradient(1px 1px at 320px 120px, #aaf, transparent);
    background-size: 350px 150px;
}

/* Sidebar */
#sidebar { width:220px; min-width:220px; background:#08080f; border-right:1px solid #1a1a2e; padding:16px; overflow-y:auto; display:flex; flex-direction:column; gap:12px; z-index:10; height:100%; }
#sidebar h2 { color:#7aa2f7; font-size:11px; text-transform:uppercase; letter-spacing:2px; margin-bottom:4px; }
.nav-btn { display:block; padding:10px 14px; background:#0f0f1a; border:1px solid #1a1a2e; border-radius:6px; color:#7aa2f7; text-decoration:none; font-size:12px; transition:all 0.2s; cursor:pointer; text-align:left; }
.nav-btn:hover, .nav-btn.active { background:#1a2a4a; border-color:#7aa2f7; box-shadow:0 0 12px rgba(122,162,247,0.2); }
.stat-row { display:flex; justify-content:space-between; font-size:11px; padding:3px 0; }
.stat-val { color:#7aa2f7; font-weight:bold; }
.meter { height:4px; background:#1a1a2e; border-radius:2px; margin:2px 0 6px; }
.meter-fill { height:100%; border-radius:2px; transition:width 0.5s ease; }
.thought-box { background:#0a0f14; border:1px solid #1a2a2e; border-radius:6px; padding:8px; font-size:10px; color:#9ece6a; font-style:italic; min-height:40px; margin-top:4px; }
#mood-display { text-align:center; font-size:28px; padding:8px; }

/* Main brain area */
#brain-container { flex:1; display:flex; align-items:center; justify-content:center; position:relative; }

/* Brain SVG regions */
.brain-region { cursor:pointer; transition:opacity 0.3s ease, filter 0.3s ease; opacity:0.7; }
.brain-region:hover { opacity:1; filter:url(#strongGlow) brightness(1.4); }
.brain-region.active { opacity:1; }
.brain-region.demo-active { opacity:1; filter:url(#strongGlow) brightness(1.6); animation:demo-pulse 1.4s ease-in-out infinite; }
@keyframes demo-pulse { 0%,100% { opacity:0.85; } 50% { opacity:1; } }
@keyframes hover-pulse { 0%,100% { opacity:0.85; } 50% { opacity:1; } }

/* Demo overlay */
#demo-controls { position:absolute; top:16px; right:16px; display:flex; gap:8px; z-index:50; }
#demo-controls button { background:rgba(15,15,26,0.85); border:1px solid #3a4a6a; color:#a0c0ff; padding:8px 14px; border-radius:6px; cursor:pointer; font-family:'Courier New',monospace; font-size:12px; transition:all 0.2s; }
#demo-controls button:hover { background:rgba(26,42,74,0.9); border-color:#7aa2f7; }
#demo-overlay { position:absolute; inset:0; pointer-events:none; opacity:0; transition:opacity 0.5s; z-index:20; }
#demo-overlay.active { opacity:1; }
#demo-label-bar { position:absolute; top:60px; left:50%; transform:translateX(-50%); background:rgba(8,8,15,0.92); border:1px solid #7aa2f7; border-radius:8px; padding:10px 22px; color:#c0d8ff; font-family:'Courier New',monospace; font-size:15px; font-weight:bold; letter-spacing:1px; box-shadow:0 0 24px rgba(122,162,247,0.4); transition:opacity 0.4s; }
#demo-label-bar:empty { display:none; }

/* Subtitle bar - sentence-level captions synced to TTS audio */
#demo-subtitle-bar { position:absolute; bottom:90px; left:8%; right:8%; min-height:60px; padding:14px 24px; background:rgba(6,6,12,0.88); border:1px solid #2a3a5a; border-left:3px solid #7aa2f7; border-radius:6px; color:#dde6f5; font-family:'Inter','Segoe UI',system-ui,sans-serif; font-size:17px; line-height:1.45; text-align:center; max-width:80%; margin:0 auto; box-shadow:0 0 28px rgba(0,0,0,0.6), 0 0 18px rgba(122,162,247,0.12); backdrop-filter:blur(4px); display:none; }
#demo-subtitle-bar.visible { display:block; }
#demo-subtitle-bar .sub-line { opacity:0; animation:sub-fade 0.35s ease forwards; }
@keyframes sub-fade { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:translateY(0); } }

/* Slide panel - per-chapter animated diagram on the right side */
#demo-slide { position:absolute; top:50%; right:24px; transform:translateY(-50%) translateX(60px); width:380px; max-width:30%; min-height:200px; max-height:68%; background:rgba(6,6,14,0.94); border:1px solid #2a3a5a; border-radius:14px; padding:16px; opacity:0; pointer-events:none; transition:opacity 0.5s ease, transform 0.5s ease; z-index:30; box-shadow:0 0 40px rgba(0,0,0,0.7), 0 0 28px rgba(122,162,247,0.15); overflow-y:auto; }
#demo-slide.visible { opacity:1; transform:translateY(-50%) translateX(0); pointer-events:auto; }
#demo-slide.swapping { opacity:0; transform:translateY(-50%) translateX(20px); }
.slide-h { color:#7aa2f7; font-family:'Courier New',monospace; font-size:11px; text-transform:uppercase; letter-spacing:2px; margin-bottom:14px; padding-bottom:8px; border-bottom:1px solid #1a2a4a; }
.slide-body { color:#cdd6e8; font-family:'Inter','Segoe UI',system-ui,sans-serif; font-size:13px; line-height:1.5; }
.slide-body svg { width:100%; height:auto; display:block; margin:6px 0; }
.slide-row { display:flex; align-items:center; gap:10px; padding:6px 0; }
.slide-chip { display:inline-block; padding:3px 10px; border-radius:10px; font-size:10px; font-family:'Courier New',monospace; letter-spacing:1px; background:rgba(122,162,247,0.12); border:1px solid rgba(122,162,247,0.4); color:#a0c0ff; }
.slide-chip.warm { background:rgba(247,118,142,0.12); border-color:rgba(247,118,142,0.4); color:#f7a0b0; }
.slide-chip.gold { background:rgba(224,175,104,0.12); border-color:rgba(224,175,104,0.4); color:#f0c890; }
.slide-chip.green { background:rgba(158,206,106,0.12); border-color:rgba(158,206,106,0.4); color:#b8e090; }
.slide-chip.purple { background:rgba(187,154,247,0.12); border-color:rgba(187,154,247,0.4); color:#d0b0f0; }
.slide-chip.cyan { background:rgba(125,207,255,0.12); border-color:rgba(125,207,255,0.4); color:#a0d8ff; }
.slide-stage { display:flex; align-items:center; gap:8px; padding:8px 12px; background:rgba(15,20,40,0.5); border-radius:6px; margin:6px 0; border-left:2px solid #3a4a6a; }
.slide-stage.lit { border-left-color:#7aa2f7; box-shadow:0 0 10px rgba(122,162,247,0.2); }
.slide-stage-name { font-size:11px; font-weight:bold; color:#c0d0ff; min-width:90px; font-family:'Courier New',monospace; }
.slide-stage-desc { font-size:11px; color:#8090b0; }
.bar-row { display:flex; align-items:center; gap:8px; margin:5px 0; font-size:11px; font-family:'Courier New',monospace; }
.bar-label { width:80px; color:#a0b0c8; }
.bar-track { flex:1; height:8px; background:#1a1a2e; border-radius:4px; overflow:hidden; position:relative; }
.bar-fill { height:100%; transition:width 1.4s ease; }
.bar-fill.low { background:linear-gradient(90deg,#f7768e,#ff9090); animation:bar-pulse 1.2s ease-in-out infinite; }
.bar-fill.mid { background:linear-gradient(90deg,#e0af68,#f0c878); }
.bar-fill.high { background:linear-gradient(90deg,#9ece6a,#b8e078); }
.bar-val { width:36px; text-align:right; color:#7aa2f7; }
@keyframes bar-pulse { 0%,100% { filter:brightness(1); } 50% { filter:brightness(1.6); } }
.mem-card { padding:8px 10px; background:rgba(20,18,30,0.6); border:1px solid #2a2a3a; border-left:3px solid #e0af68; border-radius:4px; margin:5px 0; font-size:11px; color:#c0b890; font-family:'Courier New',monospace; opacity:0; animation:card-slide 0.5s ease forwards; }
@keyframes card-slide { from { opacity:0; transform:translateX(-10px); } to { opacity:1; transform:translateX(0); } }
.wm-slot { display:inline-block; width:36px; height:36px; margin:3px; border-radius:6px; border:1px solid #2a3a5a; background:rgba(15,20,40,0.5); text-align:center; line-height:36px; font-size:10px; font-family:'Courier New',monospace; color:#5060a0; }
.wm-slot.filled { background:rgba(122,162,247,0.15); border-color:#7aa2f7; color:#a0c0ff; box-shadow:0 0 8px rgba(122,162,247,0.3); }
.heartbeat-svg path { stroke:#f7768e; stroke-width:1.6; fill:none; stroke-dasharray:1000; stroke-dashoffset:1000; animation:hb-draw 2.4s linear infinite; }
@keyframes hb-draw { to { stroke-dashoffset:-1000; } }
.signal-pulse { fill:#7aa2f7; }
@keyframes pipeline-pulse { 0% { offset-distance:0%; } 100% { offset-distance:100%; } }
.flow-line { stroke:#7aa2f7; stroke-width:1.5; fill:none; stroke-dasharray:6 4; animation:flow 1.5s linear infinite; }
.dim-axis { stroke:#3a4a6a; stroke-width:1; }
.belief-vec { stroke:#bb9af7; stroke-width:2.5; fill:none; transform-origin:50% 50%; animation:slerp-rotate 4s ease-in-out infinite; }
@keyframes slerp-rotate { 0% { transform:rotate(0deg); } 50% { transform:rotate(35deg); } 100% { transform:rotate(0deg); } }

#demo-loading { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); background:rgba(8,8,15,0.95); border:1px solid #7aa2f7; border-radius:10px; padding:18px 30px; color:#c0d8ff; font-family:'Courier New',monospace; font-size:14px; box-shadow:0 0 30px rgba(122,162,247,0.4); display:none; z-index:60; }
#demo-loading.visible { display:block; }
#demo-loading::after { content:''; display:inline-block; width:14px; height:14px; margin-left:10px; border:2px solid #7aa2f7; border-top-color:transparent; border-radius:50%; animation:spin 0.8s linear infinite; vertical-align:middle; }
@keyframes spin { to { transform:rotate(360deg); } }

#demo-progress-container { position:absolute; bottom:30px; left:10%; right:10%; height:4px; background:rgba(20,20,40,0.6); border-radius:2px; overflow:hidden; }
#demo-progress { height:100%; background:linear-gradient(90deg, #7aa2f7, #bb9af7, #f7768e); transition:width 0.4s ease; }
#demo-time { position:absolute; bottom:10px; left:50%; transform:translateX(-50%); color:#565f89; font-family:'Courier New',monospace; font-size:11px; }

/* Scene-active region highlight: pulsing ring around the current scene's regions */
.brain-region.demo-active ellipse,
.brain-region.demo-active > ellipse:first-child { stroke-width:3.5; }

/* Halo burst for scene transitions */
@keyframes halo-burst { 0% { stroke-width:2; opacity:0.85; } 60% { stroke-width:8; opacity:0.4; } 100% { stroke-width:2; opacity:0.85; } }
.brain-region.demo-active { animation:demo-pulse 1.4s ease-in-out infinite, halo-burst 2.2s ease-out 1; }

/* Glow animations */
@keyframes pulse-blue { 0%,100%{filter:drop-shadow(0 0 4px #7aa2f7);} 50%{filter:drop-shadow(0 0 16px #7aa2f7) drop-shadow(0 0 30px #4a72c7);} }
@keyframes pulse-gold { 0%,100%{filter:drop-shadow(0 0 4px #e0af68);} 50%{filter:drop-shadow(0 0 16px #e0af68) drop-shadow(0 0 30px #c09048);} }
@keyframes pulse-red { 0%,100%{filter:drop-shadow(0 0 4px #f7768e);} 50%{filter:drop-shadow(0 0 16px #f7768e) drop-shadow(0 0 30px #d75070);} }
@keyframes pulse-white { 0%,100%{filter:drop-shadow(0 0 4px #c0caf5);} 50%{filter:drop-shadow(0 0 14px #c0caf5);} }
@keyframes pulse-purple { 0%,100%{filter:drop-shadow(0 0 4px #bb9af7);} 50%{filter:drop-shadow(0 0 16px #bb9af7) drop-shadow(0 0 28px #9a7ad7);} }
@keyframes pulse-green { 0%,100%{filter:drop-shadow(0 0 4px #9ece6a);} 50%{filter:drop-shadow(0 0 12px #9ece6a);} }
@keyframes pulse-cyan { 0%,100%{filter:drop-shadow(0 0 4px #7dcfff);} 50%{filter:drop-shadow(0 0 14px #7dcfff);} }

.glow-cerebrum { animation: pulse-blue 2s infinite; }
.glow-hippocampus { animation: pulse-gold 2.5s infinite; }
.glow-amygdala { animation: pulse-red 1.8s infinite; }
.glow-thalamus { animation: pulse-white 2.2s infinite; }
.glow-hypothalamus { animation: pulse-purple 2.4s infinite; }
.glow-cerebellum { animation: pulse-green 2.6s infinite; }
.glow-brainstem { animation: pulse-cyan 3s infinite; }

/* Tooltip */
#tooltip { position:absolute; background:#0a0a14; border:1px solid #2a2a4a; border-radius:6px; padding:10px 14px; font-size:11px; max-width:240px; pointer-events:none; opacity:0; transition:opacity 0.2s; z-index:100; box-shadow:0 4px 20px rgba(0,0,0,0.5); }
#tooltip.visible { opacity:1; }
#tooltip h3 { color:#7aa2f7; font-size:12px; margin-bottom:4px; text-transform:uppercase; }
#tooltip p { color:#a0a0b0; line-height:1.4; }

/* Signal streams (animated lines between regions) */
.signal-path { stroke-dasharray: 8 4; animation: flow 1s linear infinite; opacity:0.4; }
@keyframes flow { to { stroke-dashoffset: -12; } }
.signal-path.active { opacity:0.9; stroke-width:2; }

/* Shared sidebar */
#nav-sidebar { width:240px; min-width:240px; background:#08080f; border-right:1px solid #1a1a2e; padding:20px 16px; display:flex; flex-direction:column; gap:8px; z-index:50; }
#nav-sidebar h2 { color:#7aa2f7; font-size:11px; text-transform:uppercase; letter-spacing:3px; margin-bottom:12px; font-weight:bold; }
.nav-btn { display:block; padding:14px 18px; background:rgba(15,15,26,0.6); border:1px solid #1a1a2e; border-radius:8px; color:#a0c0ff; text-decoration:none; font-size:13px; font-family:'Courier New',monospace; transition:all 0.2s; cursor:pointer; text-align:left; }
.nav-btn:hover { background:rgba(26,42,74,0.6); border-color:#5070a0; color:#c0d0ff; }
.nav-btn.active { background:rgba(26,42,74,0.8); border-color:#7aa2f7; color:#c0d8ff; box-shadow:0 0 12px rgba(122,162,247,0.3); }
</style></head><body>
<div style="display:flex;height:100vh;width:100vw;">
<div id="nav-sidebar">
    <h2>CHIP BRAIN</h2>
    <a class="nav-btn active" href="/brain">Brain Visualizer</a>
    <a class="nav-btn" href="/arena">Survival Arena</a>
    <a class="nav-btn" href="/voice">Voice Assistant</a>
</div>
<div style="flex:1;display:flex;flex-direction:row;overflow:hidden;position:relative;">

<div id="sidebar">
    <h2>Mood</h2>
    <div id="mood-display">...</div>

    <h2>Drives</h2>
    <div id="drives"></div>

    <h2>Working Memory</h2>
    <div id="wm-info"></div>

    <h2>Goals</h2>
    <div id="goal-info"></div>

    <h2>Inner Speech</h2>
    <div class="thought-box" id="thought-box">waiting for first thought...</div>

    <h2>Stats</h2>
    <div id="stats-info"></div>
</div>

<div id="brain-container">
    <!-- Demo controls + overlay -->
    <div id="demo-controls">
        <button id="demo-btn" onclick="toggleDemo()">▶ Play Demo</button>
        <button id="voice-btn" onclick="toggleVoice()">🔊 Voice ON</button>
    </div>
    <div id="demo-overlay">
        <div id="demo-label-bar"><span id="demo-label"></span></div>
        <div id="demo-loading">Loading first chapter</div>
        <div id="demo-slide"><div class="slide-h" id="slide-h">Chapter</div><div class="slide-body" id="slide-body"></div></div>
        <div id="demo-subtitle-bar"></div>
        <div id="demo-progress-container"><div id="demo-progress"></div></div>
        <div id="demo-time"></div>
    </div>

    <svg viewBox="0 0 600 700" width="60%" height="64%" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <filter id="strongGlow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="5" result="coloredBlur"/>
                <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
            <filter id="softGlow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="2.5" result="coloredBlur"/>
                <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
            <radialGradient id="brainTissue" cx="50%" cy="40%" r="60%">
                <stop offset="0%" stop-color="#1a2238" stop-opacity="0.95"/>
                <stop offset="60%" stop-color="#0f1525" stop-opacity="0.9"/>
                <stop offset="100%" stop-color="#08081a" stop-opacity="1"/>
            </radialGradient>
            <radialGradient id="cerebrumGrad"><stop offset="0%" stop-color="#3a5aaf" stop-opacity="0.55"/><stop offset="60%" stop-color="#1a2a5a" stop-opacity="0.3"/><stop offset="100%" stop-color="#0a1230" stop-opacity="0"/></radialGradient>
            <radialGradient id="thalamusGrad"><stop offset="0%" stop-color="#e0e8ff" stop-opacity="0.55"/><stop offset="60%" stop-color="#7090c0" stop-opacity="0.25"/><stop offset="100%" stop-color="#0a1230" stop-opacity="0"/></radialGradient>
            <radialGradient id="amygdalaGrad"><stop offset="0%" stop-color="#ff6080" stop-opacity="0.55"/><stop offset="60%" stop-color="#a02040" stop-opacity="0.25"/><stop offset="100%" stop-color="#0a1230" stop-opacity="0"/></radialGradient>
            <radialGradient id="hippocampusGrad"><stop offset="0%" stop-color="#ffc060" stop-opacity="0.55"/><stop offset="60%" stop-color="#a07020" stop-opacity="0.25"/><stop offset="100%" stop-color="#0a1230" stop-opacity="0"/></radialGradient>
            <radialGradient id="hypothalamusGrad"><stop offset="0%" stop-color="#c080ff" stop-opacity="0.55"/><stop offset="60%" stop-color="#7040a0" stop-opacity="0.25"/><stop offset="100%" stop-color="#0a1230" stop-opacity="0"/></radialGradient>
            <radialGradient id="cerebellumGrad"><stop offset="0%" stop-color="#80e060" stop-opacity="0.55"/><stop offset="60%" stop-color="#408020" stop-opacity="0.25"/><stop offset="100%" stop-color="#0a1230" stop-opacity="0"/></radialGradient>
            <radialGradient id="brainstemGrad"><stop offset="0%" stop-color="#60d0ff" stop-opacity="0.55"/><stop offset="60%" stop-color="#2080c0" stop-opacity="0.25"/><stop offset="100%" stop-color="#0a1230" stop-opacity="0"/></radialGradient>
            <radialGradient id="pulseMarker"><stop offset="0%" stop-color="#fff" stop-opacity="1"/><stop offset="40%" stop-color="#7aa2f7" stop-opacity="0.8"/><stop offset="100%" stop-color="#7aa2f7" stop-opacity="0"/></radialGradient>

            <!-- Mask for clipping content to inside the brain shape -->
            <mask id="brainMask">
                <rect width="600" height="700" fill="black"/>
                <path d="M298,80 C260,80 220,90 185,115 C150,140 125,180 110,225 C100,265 95,310 100,355 C108,405 125,455 150,500 C175,545 210,580 250,600 C275,612 290,615 298,615 C310,615 325,612 350,600 C390,580 425,545 450,500 C475,455 492,405 500,355 C505,310 500,265 490,225 C475,180 450,140 415,115 C380,90 340,80 302,80 Z" fill="white"/>
            </mask>
        </defs>

        <!-- Brain shell with tissue fill -->
        <path d="M298,80 C260,80 220,90 185,115 C150,140 125,180 110,225 C100,265 95,310 100,355 C108,405 125,455 150,500 C175,545 210,580 250,600 C275,612 290,615 298,615 C310,615 325,612 350,600 C390,580 425,545 450,500 C475,455 492,405 500,355 C505,310 500,265 490,225 C475,180 450,140 415,115 C380,90 340,80 302,80 Z"
              fill="url(#brainTissue)" stroke="#3a5070" stroke-width="2.5" opacity="0.95"/>

        <!-- Central longitudinal fissure -->
        <line x1="300" y1="82" x2="300" y2="612" stroke="#1a2540" stroke-width="2" opacity="0.6"/>

        <!-- Sulci/gyri (clipped to brain shape) -->
        <g mask="url(#brainMask)" opacity="0.35" stroke="#2a3a5a" stroke-width="1.2" fill="none">
            <path d="M180,130 Q200,150 195,180 Q190,210 210,230"/>
            <path d="M150,180 Q175,200 170,235 Q165,270 185,290"/>
            <path d="M130,240 Q155,260 150,295 Q145,330 165,350"/>
            <path d="M120,310 Q145,330 140,365 Q135,400 155,420"/>
            <path d="M140,400 Q165,420 160,455 Q155,485 175,505"/>
            <path d="M180,470 Q205,490 200,520 Q195,550 215,565"/>
            <path d="M210,150 Q230,170 225,200"/>
            <path d="M250,200 Q270,215 265,245 Q260,275 275,290"/>
            <path d="M230,300 Q250,315 245,345 Q240,375 255,395"/>
            <path d="M250,420 Q270,435 265,465 Q260,495 275,515"/>
            <path d="M420,130 Q400,150 405,180 Q410,210 390,230"/>
            <path d="M450,180 Q425,200 430,235 Q435,270 415,290"/>
            <path d="M470,240 Q445,260 450,295 Q455,330 435,350"/>
            <path d="M480,310 Q455,330 460,365 Q465,400 445,420"/>
            <path d="M460,400 Q435,420 440,455 Q445,485 425,505"/>
            <path d="M420,470 Q395,490 400,520 Q405,550 385,565"/>
            <path d="M390,150 Q370,170 375,200"/>
            <path d="M350,200 Q330,215 335,245 Q340,275 325,290"/>
            <path d="M370,300 Q350,315 355,345 Q360,375 345,395"/>
            <path d="M350,420 Q330,435 335,465 Q340,495 325,515"/>
        </g>

        <!-- Connection pathways (white-matter tracts, realistic anatomical routing) -->
        <g mask="url(#brainMask)">
            <!-- Thalamus → Cerebrum (left): thalamocortical radiation -->
            <path class="signal-path" id="sig-thal-cer"
                  d="M285,275 C260,240 235,210 225,180 C220,160 220,150 225,140"
                  stroke="#7aa2f7" fill="none" stroke-width="2" opacity="0.75"/>
            <!-- Thalamus → Cerebrum (right): mirror -->
            <path class="signal-path" id="sig-thal-cer2"
                  d="M315,275 C340,240 365,210 375,180 C380,160 380,150 375,140"
                  stroke="#7aa2f7" fill="none" stroke-width="2" opacity="0.75"/>
            <!-- Thalamus → Amygdala: subcortical fast path -->
            <path class="signal-path" id="sig-thal-amy"
                  d="M280,290 C260,295 240,305 230,315"
                  stroke="#f7768e" fill="none" stroke-width="1.8" opacity="0.7"/>
            <!-- Hippocampus → Thalamus (memory recall pathway) -->
            <path class="signal-path" id="sig-hip-thal"
                  d="M225,410 C245,380 265,350 280,310"
                  stroke="#e0af68" fill="none" stroke-width="1.8" opacity="0.7"/>
            <!-- Amygdala → Hippocampus (emotional memory tagging) -->
            <path class="signal-path" id="sig-amy-hip"
                  d="M210,338 C208,360 208,388 213,410"
                  stroke="#f7768e" fill="none" stroke-width="1.4" opacity="0.5"/>
            <!-- Hypothalamus → Cerebrum: drive signal up to cortex -->
            <path class="signal-path" id="sig-hyp-cer"
                  d="M310,340 C330,300 355,250 365,200 C370,170 370,155 365,140"
                  stroke="#bb9af7" fill="none" stroke-width="1.8" opacity="0.7"/>
            <!-- Cerebrum → Cerebellum: corticopontine tract -->
            <path class="signal-path" id="sig-cer-cbl"
                  d="M385,180 C410,250 425,320 420,400"
                  stroke="#9ece6a" fill="none" stroke-width="2" opacity="0.7"/>
            <!-- Brainstem → Hypothalamus (visceral/autonomic pathway) -->
            <path class="signal-path" id="sig-bs-hyp"
                  d="M300,495 C302,460 302,420 304,370"
                  stroke="#7dcfff" fill="none" stroke-width="1.8" opacity="0.7"/>
            <!-- Cerebellum → Brainstem: descending motor tract -->
            <path class="signal-path" id="sig-cbl-bs"
                  d="M395,460 C370,490 340,510 320,520"
                  stroke="#9ece6a" fill="none" stroke-width="1.4" opacity="0.5"/>
        </g>

        <!-- Animated traveling pulses -->
        <g class="pulses">
            <circle r="3" fill="url(#pulseMarker)"><animateMotion dur="2.5s" repeatCount="indefinite"><mpath href="#sig-thal-cer"/></animateMotion></circle>
            <circle r="3" fill="url(#pulseMarker)"><animateMotion dur="2.8s" repeatCount="indefinite" begin="0.5s"><mpath href="#sig-thal-cer2"/></animateMotion></circle>
            <circle r="2.5" fill="#e0af68" opacity="0.9"><animateMotion dur="3.2s" repeatCount="indefinite" begin="1.2s"><mpath href="#sig-hip-thal"/></animateMotion></circle>
            <circle r="2.5" fill="#bb9af7" opacity="0.9"><animateMotion dur="2.7s" repeatCount="indefinite" begin="0.3s"><mpath href="#sig-hyp-cer"/></animateMotion></circle>
            <circle r="2.5" fill="#9ece6a" opacity="0.9"><animateMotion dur="3s" repeatCount="indefinite" begin="0.8s"><mpath href="#sig-cer-cbl"/></animateMotion></circle>
            <circle r="2.5" fill="#7dcfff" opacity="0.9"><animateMotion dur="3.5s" repeatCount="indefinite" begin="1.5s"><mpath href="#sig-bs-hyp"/></animateMotion></circle>
        </g>

        <!--
            REGIONS - each defined by a single (cx, cy) so aura and core are perfectly aligned.
            All positions verified to fit inside the brain outline.
            Layout (top to bottom):
                Cerebrum    : (300, 145)  large, top
                Thalamus    : (300, 285)  center hub
                Hypothalamus: (300, 350)  below thalamus
                Amygdala    : (215, 320)  left of thalamus
                Hippocampus : (215, 425)  below amygdala
                Cerebellum  : (415, 440)  right-back lobe
                Brainstem   : (300, 530)  bottom center
        -->

        <!-- Auras (large soft halos, clipped to brain shape) -->
        <g mask="url(#brainMask)">
            <ellipse cx="300" cy="145" rx="140" ry="55" fill="url(#cerebrumGrad)"/>
            <circle cx="300" cy="285" r="32" fill="url(#thalamusGrad)"/>
            <circle cx="215" cy="320" r="32" fill="url(#amygdalaGrad)"/>
            <circle cx="215" cy="425" r="38" fill="url(#hippocampusGrad)"/>
            <circle cx="300" cy="350" r="32" fill="url(#hypothalamusGrad)"/>
            <circle cx="415" cy="440" r="45" fill="url(#cerebellumGrad)"/>
            <ellipse cx="300" cy="530" rx="25" ry="35" fill="url(#brainstemGrad)"/>
        </g>

        <!-- Region cores (interactive, with neon ring borders) -->
        <ellipse class="brain-region glow-cerebrum" id="reg-cerebrum" cx="300" cy="145" rx="115" ry="42"
                 fill="rgba(58,90,175,0.12)" stroke="#7aa2f7" stroke-width="2" filter="url(#strongGlow)"
                 data-name="Cerebrum" data-desc="Higher cognition: policy, working memory, world model, reasoning chain, goals, inner speech, personality, causal reasoning. The seat of voluntary thought."/>

        <ellipse class="brain-region glow-thalamus" id="reg-thalamus" cx="300" cy="285" rx="22" ry="17"
                 fill="rgba(192,202,245,0.12)" stroke="#c0caf5" stroke-width="2" filter="url(#strongGlow)"
                 data-name="Thalamus" data-desc="Sensory relay hub: Granite-125m text encoder, transformer backbone, attention bottleneck. Every signal enters through here first."/>

        <ellipse class="brain-region glow-amygdala" id="reg-amygdala" cx="215" cy="320" rx="22" ry="17"
                 fill="rgba(247,118,142,0.12)" stroke="#f7768e" stroke-width="2" filter="url(#strongGlow)"
                 data-name="Amygdala" data-desc="Emotion processing: valence, fear veto, arousal modulation, habituation. Fast threat detection that bypasses conscious thought."/>

        <ellipse class="brain-region glow-hippocampus" id="reg-hippocampus" cx="215" cy="425" rx="30" ry="20"
                 fill="rgba(224,175,104,0.12)" stroke="#e0af68" stroke-width="2" filter="url(#strongGlow)"
                 data-name="Hippocampus" data-desc="Memory: episodic store/recall, dream replay, active dreaming, boundary detection, cognitive map, temporal abstraction."/>

        <ellipse class="brain-region glow-hypothalamus" id="reg-hypothalamus" cx="300" cy="350" rx="22" ry="14"
                 fill="rgba(187,154,247,0.12)" stroke="#bb9af7" stroke-width="2" filter="url(#strongGlow)"
                 data-name="Hypothalamus" data-desc="Drives and homeostasis: curiosity, energy, safety, engagement, coherence. Generates goals from internal deficits."/>

        <g class="brain-region glow-cerebellum" id="reg-cerebellum"
           data-name="Cerebellum" data-desc="Motor coordination: action smoothing, skill library, swarm consensus, emotional contagion. Smooth, precise output.">
            <ellipse cx="415" cy="440" rx="38" ry="32" fill="rgba(158,206,106,0.12)" stroke="#9ece6a" stroke-width="2" filter="url(#strongGlow)"/>
            <path d="M380,425 Q415,420 450,430" stroke="#9ece6a" stroke-width="0.8" opacity="0.5" fill="none"/>
            <path d="M378,440 Q415,438 452,445" stroke="#9ece6a" stroke-width="0.8" opacity="0.5" fill="none"/>
            <path d="M380,455 Q415,455 450,460" stroke="#9ece6a" stroke-width="0.8" opacity="0.5" fill="none"/>
        </g>

        <ellipse class="brain-region glow-brainstem" id="reg-brainstem" cx="300" cy="530" rx="16" ry="32"
                 fill="rgba(125,207,255,0.12)" stroke="#7dcfff" stroke-width="2" filter="url(#strongGlow)"
                 data-name="Brainstem" data-desc="Life support: SAC training loop, gradient health, NaN detection, autosave, LR scheduling. Always running, never conscious."/>

        <!-- Region labels (subtle, positioned at region centers) -->
        <g style="pointer-events:none">
            <text x="300" y="148" font-size="11" fill="#7aa2f7" text-anchor="middle" font-weight="bold" opacity="0.85" filter="url(#softGlow)">CEREBRUM</text>
            <text x="300" y="289" font-size="8" fill="#c0caf5" text-anchor="middle" opacity="0.85">THALAMUS</text>
            <text x="215" y="324" font-size="8" fill="#f7768e" text-anchor="middle" opacity="0.85">AMYGDALA</text>
            <text x="215" y="429" font-size="8" fill="#e0af68" text-anchor="middle" opacity="0.85">HIPPOCAMPUS</text>
            <text x="300" y="354" font-size="7" fill="#bb9af7" text-anchor="middle" opacity="0.85">HYPOTHAL.</text>
            <text x="415" y="445" font-size="9" fill="#9ece6a" text-anchor="middle" opacity="0.85">CEREBELLUM</text>
            <text x="300" y="534" font-size="7" fill="#7dcfff" text-anchor="middle" opacity="0.85">BRAINSTEM</text>
        </g>
    </svg>

    <div id="tooltip"><h3 id="tt-name"></h3><p id="tt-desc"></p></div>
</div>

<script>
const MOOD_EMOJI = {Calm:'😌',Happy:'😊',Sad:'😔',Angry:'😠'};
const tooltip = document.getElementById('tooltip');
const ttName = document.getElementById('tt-name');
const ttDesc = document.getElementById('tt-desc');

// Hover tooltips
document.querySelectorAll('.brain-region').forEach(el => {
    el.addEventListener('mouseenter', (e) => {
        ttName.textContent = el.dataset.name;
        ttDesc.textContent = el.dataset.desc;
        tooltip.classList.add('visible');
    });
    el.addEventListener('mousemove', (e) => {
        tooltip.style.left = (e.clientX - 250) + 'px';
        tooltip.style.top = (e.clientY + 20) + 'px';
    });
    el.addEventListener('mouseleave', () => {
        tooltip.classList.remove('visible');
    });
});

function update(data) {
    // Mood
    document.getElementById('mood-display').textContent = (MOOD_EMOJI[data.mood]||'🧠') + ' ' + (data.mood||'?');

    // Drives
    let dh = '';
    if(data.homeostasis) Object.entries(data.homeostasis).forEach(([k,v])=>{
        const pct = (v*100).toFixed(0);
        const color = v>0.6?'#9ece6a':v>0.3?'#e0af68':'#f7768e';
        dh += `<div class="stat-row"><span>${k}</span><span class="stat-val">${pct}%</span></div><div class="meter"><div class="meter-fill" style="width:${pct}%;background:${color}"></div></div>`;
    });
    document.getElementById('drives').innerHTML = dh;

    // Working memory
    if(data.working_memory) {
        const wm = data.working_memory;
        document.getElementById('wm-info').innerHTML = `<div class="stat-row"><span>slots</span><span class="stat-val">${wm.n_slots}/${wm.capacity}</span></div>` +
            (wm.slots||[]).map(s=>`<div class="stat-row"><span style="color:#565f89">${s.source}</span><span class="stat-val">${s.salience.toFixed(2)}</span></div>`).join('');
    }

    // Goals
    if(data.goal_stack) {
        const gs = data.goal_stack;
        document.getElementById('goal-info').innerHTML = gs.is_empty ? '<span style="color:#565f89">no active goals</span>' :
            gs.stack.map(g=>`<div class="stat-row"><span>${g.name}</span><span class="stat-val">t=${g.ticks_active}</span></div>`).join('');
    }

    // Inner speech
    if(data.inner_speech && data.inner_speech.recent && data.inner_speech.recent.length) {
        const last = data.inner_speech.recent[data.inner_speech.recent.length-1];
        document.getElementById('thought-box').textContent = last.text || JSON.stringify(last);
    }

    // Stats
    document.getElementById('stats-info').innerHTML =
        `<div class="stat-row"><span>tick</span><span class="stat-val">${data.tick||0}</span></div>` +
        `<div class="stat-row"><span>confidence</span><span class="stat-val">${(data.meta_cognition?.mean_confidence||0).toFixed(2)}</span></div>` +
        `<div class="stat-row"><span>memories</span><span class="stat-val">${data.episodic_memory_size||0}</span></div>` +
        `<div class="stat-row"><span>place cells</span><span class="stat-val">${data.cognitive_map?.n_cells||0}</span></div>`;

    // Pulse active signal paths
    document.querySelectorAll('.signal-path').forEach(p=>p.classList.remove('active'));
    document.getElementById('sig-thal-cer').classList.add('active');
    if(data.inner_speech?.n_thoughts > 0) document.getElementById('sig-cer-cbl').classList.add('active');
    if(data.episodic_memory_size > 0) document.getElementById('sig-hip-cer').classList.add('active');
}

const evtSource = new EventSource('/brain/stream');
evtSource.onmessage = (e) => { try { update(JSON.parse(e.data)); } catch(err){} };

// ============================================================
// DEMO SLIDESHOW - narrated tour of how a real brain maps to Chip
// Each scene is one TTS request (one chunk). Scene 0 plays the moment it
// finishes loading; later scenes prefetch in the background while the
// current one plays, so playback feels continuous instead of front-loaded.
// Subtitles are sentence-level and ride on the audio's currentTime, so
// what you hear and what you read stay in lockstep.
// ============================================================

const SCENES = [
    {
        title: "Where every thought begins",
        regions: [],
        sentences: [
            "Every thought you have starts the same way.",
            "A signal arrives, gets routed, gets felt, gets remembered, and finally turns into something you do.",
            "Chip follows that exact path.",
            "What you are looking at is not a chatbot. It is a brain.",
            "Each region has its own job, and they pass typed messages to each other every tick."
        ],
        slide: `
            <div class="slide-stage" data-stage="0"><span class="slide-stage-name">SENSE</span><span class="slide-stage-desc">input arrives</span></div>
            <div class="slide-stage" data-stage="1"><span class="slide-stage-name">ROUTE</span><span class="slide-stage-desc">thalamic gating</span></div>
            <div class="slide-stage" data-stage="1"><span class="slide-stage-name">FEEL</span><span class="slide-stage-desc">amygdala valence</span></div>
            <div class="slide-stage" data-stage="1"><span class="slide-stage-name">RECALL</span><span class="slide-stage-desc">hippocampal lookup</span></div>
            <div class="slide-stage" data-stage="1"><span class="slide-stage-name">PLAN</span><span class="slide-stage-desc">cortical decision</span></div>
            <div class="slide-stage" data-stage="1"><span class="slide-stage-name">ACT</span><span class="slide-stage-desc">cerebellar smoothing</span></div>
            <div style="margin-top:14px;padding:10px;background:rgba(15,30,15,0.4);border-radius:6px;font-size:11px;color:#90b890;line-height:1.5;" data-stage="2">
                Same loop, every tick. <span style="color:#9ece6a">No region imports another</span>; they communicate by typed signals on a shared bus.
            </div>
            <div style="margin-top:10px;display:flex;gap:6px;flex-wrap:wrap;" data-stage="4">
                <span class="slide-chip">cerebrum</span><span class="slide-chip warm">amygdala</span><span class="slide-chip gold">hippocampus</span><span class="slide-chip purple">hypothalamus</span><span class="slide-chip green">cerebellum</span><span class="slide-chip cyan">brainstem</span>
            </div>
        `
    },
    {
        title: "Thalamus: the gate everything passes through",
        regions: ["thalamus"],
        sentences: [
            "In humans, the thalamus sits in the middle of the brain like a switchboard.",
            "Nearly every sensory signal in your body, except smell, lands there first before reaching the cortex.",
            "It filters. It decides what is worth your attention this second.",
            "Chip's thalamus does the same job in software.",
            "Incoming text gets encoded by Granite-125m into a 512-dimensional vector, runs through a transformer backbone, and an attention bottleneck keeps only the strongest features.",
            "The rest is discarded the way you stop noticing the feel of your own clothes a few seconds after putting them on."
        ],
        slide: `
            <svg viewBox="0 0 360 200" style="margin-bottom:8px">
                <defs>
                    <linearGradient id="thalGrad" x1="0%" x2="100%"><stop offset="0%" stop-color="#7aa2f7" stop-opacity="0.2"/><stop offset="100%" stop-color="#7aa2f7" stop-opacity="0.05"/></linearGradient>
                </defs>
                <text x="20" y="22" fill="#7090b0" font-size="10" font-family="Courier New">INPUT STREAMS</text>
                <text x="20" y="44" fill="#a0c0ff" font-size="11" font-family="Inter">vision</text>
                <text x="20" y="62" fill="#a0c0ff" font-size="11" font-family="Inter">touch</text>
                <text x="20" y="80" fill="#a0c0ff" font-size="11" font-family="Inter">hearing</text>
                <text x="20" y="98" fill="#a0c0ff" font-size="11" font-family="Inter">taste</text>
                <text x="20" y="116" fill="#5a6a8a" font-size="11" font-family="Inter" font-style="italic">smell (skips)</text>

                <line x1="80" y1="40" x2="170" y2="100" class="flow-line"/>
                <line x1="80" y1="58" x2="170" y2="100" class="flow-line"/>
                <line x1="80" y1="76" x2="170" y2="100" class="flow-line"/>
                <line x1="80" y1="94" x2="170" y2="100" class="flow-line"/>

                <ellipse cx="180" cy="100" rx="34" ry="28" fill="url(#thalGrad)" stroke="#c0caf5" stroke-width="2"/>
                <text x="180" y="98" fill="#c0caf5" font-size="11" text-anchor="middle" font-family="Courier New" font-weight="bold">THAL</text>
                <text x="180" y="112" fill="#a0b0d0" font-size="9" text-anchor="middle" font-family="Inter">filter</text>

                <line x1="216" y1="100" x2="290" y2="60" class="flow-line"/>
                <line x1="216" y1="100" x2="290" y2="100" class="flow-line"/>
                <line x1="216" y1="100" x2="290" y2="140" class="flow-line"/>

                <text x="300" y="60" fill="#7aa2f7" font-size="11" font-family="Courier New">cortex</text>
                <text x="300" y="100" fill="#f7768e" font-size="11" font-family="Courier New">amygdala</text>
                <text x="300" y="140" fill="#5a6a8a" font-size="11" font-family="Courier New">discarded</text>
            </svg>
            <div class="bar-row" data-stage="4"><span class="bar-label">tokens in</span><div class="bar-track"><div class="bar-fill mid" style="width:90%"></div></div><span class="bar-val">512</span></div>
            <div class="bar-row" data-stage="4"><span class="bar-label">survives</span><div class="bar-track"><div class="bar-fill high" style="width:18%"></div></div><span class="bar-val">~94</span></div>
            <div style="font-size:11px;color:#7090b0;line-height:1.5;margin-top:6px;font-family:Inter" data-stage="5">
                Attention bottleneck keeps roughly the top 18 percent of features. The rest fade the way clothing fades from awareness.
            </div>
        `
    },
    {
        title: "Amygdala: the fast lane for emotion",
        regions: ["thalamus", "amygdala"],
        sentences: [
            "Out of the thalamus, two paths fire at once.",
            "One goes the slow way, up to the cortex for careful thought.",
            "The other goes straight to the amygdala in a few milliseconds.",
            "That low road is why you flinch from a snake on the trail before you consciously register it is a snake.",
            "Chip's amygdala scores valence in the same shortcut, and it can veto an action before the planner has even finished thinking.",
            "It also habituates. Show it the same input twenty times and the response fades, the same way your brain stops reacting to a ticking clock."
        ],
        slide: `
            <svg viewBox="0 0 360 180" style="margin-bottom:10px">
                <circle cx="40" cy="90" r="18" fill="rgba(192,202,245,0.15)" stroke="#c0caf5" stroke-width="2"/>
                <text x="40" y="93" fill="#c0caf5" font-size="9" text-anchor="middle" font-family="Courier New">THAL</text>

                <path d="M62,82 C140,40 220,40 290,55" class="flow-line" stroke="#7aa2f7" data-stage="1"/>
                <text x="170" y="32" fill="#7aa2f7" font-size="10" text-anchor="middle" font-family="Courier New">HIGH ROAD - cortex - 100ms</text>

                <path d="M62,98 C140,140 220,140 290,125" class="flow-line" stroke="#f7768e" data-stage="2"/>
                <text x="170" y="170" fill="#f7768e" font-size="10" text-anchor="middle" font-family="Courier New">LOW ROAD - amygdala - 12ms</text>

                <rect x="280" y="40" width="60" height="36" rx="6" fill="rgba(122,162,247,0.1)" stroke="#7aa2f7" stroke-width="1.5"/>
                <text x="310" y="62" fill="#a0c0ff" font-size="10" text-anchor="middle" font-family="Courier New">CORTEX</text>

                <rect x="280" y="105" width="60" height="36" rx="6" fill="rgba(247,118,142,0.1)" stroke="#f7768e" stroke-width="1.5" data-stage="3"/>
                <text x="310" y="127" fill="#f7a0b0" font-size="10" text-anchor="middle" font-family="Courier New">AMYG</text>
            </svg>
            <div class="bar-row" data-stage="5"><span class="bar-label">repeat 1</span><div class="bar-track"><div class="bar-fill low" style="width:92%"></div></div><span class="bar-val">.92</span></div>
            <div class="bar-row" data-stage="5"><span class="bar-label">repeat 5</span><div class="bar-track"><div class="bar-fill mid" style="width:64%"></div></div><span class="bar-val">.64</span></div>
            <div class="bar-row" data-stage="5"><span class="bar-label">repeat 10</span><div class="bar-track"><div class="bar-fill mid" style="width:38%"></div></div><span class="bar-val">.38</span></div>
            <div class="bar-row" data-stage="5"><span class="bar-label">repeat 20</span><div class="bar-track"><div class="bar-fill high" style="width:11%"></div></div><span class="bar-val">.11</span></div>
            <div style="font-size:11px;color:#7090b0;font-family:Inter;margin-top:4px" data-stage="5">
                Habituation curve. Same input, fading response.
            </div>
        `
    },
    {
        title: "Hippocampus: turning experience into memory",
        regions: ["amygdala", "hippocampus"],
        sentences: [
            "The hippocampus is shaped like a seahorse, which is where its name comes from.",
            "It binds together where you were, who was with you, and what you felt, then writes that bundle to long-term memory while you sleep.",
            "Patient H.M. had his removed in 1953 and could never form a new conscious memory again.",
            "He could still ride a bike. He could still feel emotions. He just could not tell you what happened five minutes ago.",
            "Chip's hippocampus stores each episode indexed by its latent vector.",
            "When new input arrives, it pulls the three most relevant past episodes back into working memory through cosine similarity.",
            "It also watches for prediction errors and marks event boundaries, which is why you remember a car crash but forget the drive there.",
            "And during idle cycles, it does something the human hippocampus does during sleep: it replays key decision points, simulates alternative outcomes, and writes the better ones back as synthetic memories."
        ],
        slide: `
            <div style="font-size:10px;color:#7090b0;font-family:Courier New;margin-bottom:4px;letter-spacing:1px" data-stage="1">EPISODE = WHERE + WHO + FEEL</div>
            <div class="mem-card" style="animation-delay:0s;padding:5px 8px;margin:3px 0;font-size:10px" data-stage="1">2025-03-12 / corridor / unfamiliar door / curious</div>
            <div class="mem-card" style="animation-delay:0.15s;padding:5px 8px;margin:3px 0;font-size:10px" data-stage="1">2025-03-09 / lab / new person introduced / wary</div>
            <div class="mem-card" style="animation-delay:0.3s;padding:5px 8px;margin:3px 0;font-size:10px" data-stage="1">2025-03-04 / kitchen / dropped a glass / startled</div>

            <div style="margin:7px 0;padding:7px 9px;background:rgba(20,15,5,0.6);border-radius:5px;font-size:10px;color:#c0a070;font-family:Inter;line-height:1.45" data-stage="2">
                <strong style="color:#e0af68">H.M., 1953.</strong> Both hippocampi removed. Childhood memories intact. Motor skills intact. Could not form a single new conscious memory for the rest of his life.
            </div>

            <div style="font-size:10px;color:#7090b0;font-family:Courier New;margin:6px 0 3px" data-stage="5">RECALL: cosine similarity</div>
            <div class="bar-row" style="margin:3px 0" data-stage="5"><span class="bar-label">ep #142</span><div class="bar-track"><div class="bar-fill high" style="width:88%"></div></div><span class="bar-val">.88</span></div>
            <div class="bar-row" style="margin:3px 0" data-stage="5"><span class="bar-label">ep #097</span><div class="bar-track"><div class="bar-fill mid" style="width:71%"></div></div><span class="bar-val">.71</span></div>
            <div class="bar-row" style="margin:3px 0" data-stage="5"><span class="bar-label">ep #211</span><div class="bar-track"><div class="bar-fill mid" style="width:64%"></div></div><span class="bar-val">.64</span></div>
            <div class="bar-row" style="margin:3px 0" data-stage="5"><span class="bar-label">ep #033</span><div class="bar-track"><div class="bar-fill low" style="width:22%"></div></div><span class="bar-val">.22</span></div>
            <div style="font-size:10px;color:#9ece6a;font-family:Inter;margin-top:4px" data-stage="6">Top 3 enter working memory. The rest stay dormant.</div>
            <div style="margin-top:6px;padding:6px 9px;background:rgba(10,20,10,0.6);border-left:3px solid #9ece6a;border-radius:4px;font-size:10px;color:#90c890;line-height:1.4;font-family:Inter" data-stage="7">
                Idle replay: simulates alt trajectories, writes better outcomes back as synthetic memories.
            </div>
        `
    },
    {
        title: "Hypothalamus: the engine of motivation",
        regions: ["hippocampus", "hypothalamus"],
        sentences: [
            "Beneath the thalamus sits the hypothalamus, no bigger than an almond.",
            "It runs almost every drive you have. Hunger, thirst, body temperature, sleep, the urge to bond, the urge to flee.",
            "Without it you would feel nothing pushing you toward anything.",
            "Chip's hypothalamus tracks six drives instead of dozens.",
            "Energy, curiosity, safety, social engagement, coherence, and novelty.",
            "When any one of them drifts out of range, the hypothalamus generates a goal automatically.",
            "Curiosity reward comes straight from how surprised the world model was, which is the same dopaminergic loop you feel when something genuinely interesting happens."
        ],
        slide: `
            <div style="font-size:11px;color:#7090b0;font-family:Courier New;margin-bottom:6px;letter-spacing:1px" data-stage="3">SIX DRIVES, LIVE</div>
            <div class="bar-row" data-stage="4"><span class="bar-label">energy</span><div class="bar-track"><div class="bar-fill mid" style="width:62%"></div></div><span class="bar-val">.62</span></div>
            <div class="bar-row" data-stage="4"><span class="bar-label">curiosity</span><div class="bar-track"><div class="bar-fill high" style="width:81%"></div></div><span class="bar-val">.81</span></div>
            <div class="bar-row" data-stage="4"><span class="bar-label">safety</span><div class="bar-track"><div class="bar-fill high" style="width:74%"></div></div><span class="bar-val">.74</span></div>
            <div class="bar-row" data-stage="4"><span class="bar-label">social</span><div class="bar-track"><div class="bar-fill mid" style="width:48%"></div></div><span class="bar-val">.48</span></div>
            <div class="bar-row" data-stage="4"><span class="bar-label">coherence</span><div class="bar-track"><div class="bar-fill high" style="width:79%"></div></div><span class="bar-val">.79</span></div>
            <div class="bar-row" data-stage="4"><span class="bar-label">novelty</span><div class="bar-track"><div class="bar-fill low" style="width:18%"></div></div><span class="bar-val">.18</span></div>
            <div style="margin-top:12px;padding:10px;background:rgba(15,5,25,0.6);border-radius:6px;font-size:11px;color:#c0a0e0;line-height:1.5;font-family:Inter" data-stage="5">
                <strong style="color:#bb9af7">Drift out of range</strong> &rarr; goal fires automatically. Low novelty triggered <span class="slide-chip purple">explore_frontier</span> on this tick.
            </div>
            <div style="font-size:11px;color:#7090b0;font-family:Inter;margin-top:8px;line-height:1.5" data-stage="6">
                Curiosity reward = surprise from the world model. The same dopaminergic loop that fires when something genuinely interesting happens to you.
            </div>
        `
    },
    {
        title: "Cerebrum: planning, language, identity",
        regions: ["hypothalamus", "cerebrum"],
        sentences: [
            "The cerebrum is the wrinkled outer layer most people picture when they hear the word brain.",
            "It holds plans, language, autobiographical self, and most of what you would call you.",
            "Chip's cerebrum keeps seven working-memory slots, the same magic number George Miller wrote about in 1956.",
            "A dual-actor soft actor-critic policy picks the next move.",
            "A reasoning chain only fires when meta-cognitive confidence drops, which keeps the lights on without burning compute on easy decisions.",
            "Inner speech runs in plain English, gets re-encoded back into the latent space, and anchors a stable identity across thousands of ticks."
        ],
        slide: `
            <div style="font-size:11px;color:#7090b0;font-family:Courier New;margin-bottom:6px;letter-spacing:1px" data-stage="2">WORKING MEMORY (Miller, 1956)</div>
            <div style="text-align:center;margin:6px 0" data-stage="2">
                <span class="wm-slot filled">obs</span>
                <span class="wm-slot filled">goal</span>
                <span class="wm-slot filled">ep142</span>
                <span class="wm-slot filled">ep097</span>
                <span class="wm-slot filled">mood</span>
                <span class="wm-slot">+</span>
                <span class="wm-slot">+</span>
            </div>
            <div style="font-size:10px;color:#5a6a8a;text-align:center;font-family:Inter;font-style:italic;margin-bottom:10px" data-stage="2">7 ± 2 slots, same as the human limit</div>

            <div style="font-size:11px;color:#7090b0;font-family:Courier New;margin:8px 0 4px;letter-spacing:1px" data-stage="3">DUAL-ACTOR SAC POLICY</div>
            <div class="bar-row" data-stage="3"><span class="bar-label">actor A</span><div class="bar-track"><div class="bar-fill high" style="width:64%"></div></div><span class="bar-val">.64</span></div>
            <div class="bar-row" data-stage="3"><span class="bar-label">actor B</span><div class="bar-track"><div class="bar-fill mid" style="width:51%"></div></div><span class="bar-val">.51</span></div>
            <div class="bar-row" data-stage="3"><span class="bar-label">consensus</span><div class="bar-track"><div class="bar-fill high" style="width:59%"></div></div><span class="bar-val">.59</span></div>

            <div style="margin-top:10px;padding:8px 10px;background:rgba(15,30,15,0.5);border-left:3px solid #9ece6a;border-radius:4px;font-size:11px;color:#a0c898;font-style:italic;line-height:1.5;font-family:Inter" data-stage="5">
                "I should investigate further. Confidence is reasonable. Memory pattern matches."
            </div>
            <div style="font-size:10px;color:#5a6a8a;text-align:right;font-family:Inter;margin-top:2px" data-stage="5">- inner speech, tick 1247</div>
        `
    },
    {
        title: "Cerebellum: the timing of skilled action",
        regions: ["cerebrum", "cerebellum"],
        sentences: [
            "Behind and below the cerebrum sits the cerebellum.",
            "It contains roughly half of all the neurons in the human brain, packed tight under the back of the skull.",
            "People used to think it only handled balance. We now know it shapes almost any skilled, rapid action.",
            "A tennis swing. Speech timing. The micro-corrections in your fingers when you pick up a coffee cup.",
            "Chip's cerebellum smooths every chosen action with an exponential moving average so the output stops looking robotic.",
            "It also maintains a small skill library, so similar problems reuse the same motor primitives instead of being solved from scratch."
        ],
        slide: `
            <div style="font-size:11px;color:#7090b0;font-family:Courier New;margin-bottom:6px;letter-spacing:1px" data-stage="1">NEURON COUNTS</div>
            <div class="bar-row" data-stage="1"><span class="bar-label">cerebellum</span><div class="bar-track"><div class="bar-fill high" style="width:69%"></div></div><span class="bar-val">69B</span></div>
            <div class="bar-row" data-stage="1"><span class="bar-label">cerebrum</span><div class="bar-track"><div class="bar-fill mid" style="width:21%"></div></div><span class="bar-val">21B</span></div>
            <div class="bar-row" data-stage="1"><span class="bar-label">all other</span><div class="bar-track"><div class="bar-fill low" style="width:10%"></div></div><span class="bar-val">~6B</span></div>
            <div style="font-size:10px;color:#5a6a8a;font-family:Inter;font-style:italic;margin-bottom:10px">~86 billion neurons total. The cerebellum holds the majority.</div>

            <div style="font-size:11px;color:#7090b0;font-family:Courier New;margin:8px 0 6px;letter-spacing:1px" data-stage="4">EMA SMOOTHING (raw vs out)</div>
            <svg viewBox="0 0 320 80" style="margin-bottom:6px">
                <polyline points="0,40 20,15 40,55 60,20 80,60 100,18 120,58 140,22 160,52 180,28 200,48 220,30 240,46 260,34 280,42 300,38 320,40" stroke="#f7768e" stroke-width="1.4" fill="none" opacity="0.7"/>
                <polyline points="0,40 20,33 40,40 60,35 80,42 100,36 120,42 140,37 160,41 180,38 200,40 220,39 240,40 260,40 280,40 300,40 320,40" stroke="#9ece6a" stroke-width="2" fill="none"/>
                <text x="4" y="12" fill="#f7768e" font-size="9" font-family="Courier New">raw</text>
                <text x="4" y="76" fill="#9ece6a" font-size="9" font-family="Courier New">smoothed</text>
            </svg>

            <div style="font-size:11px;color:#7090b0;font-family:Courier New;margin:8px 0 4px;letter-spacing:1px" data-stage="5">SKILL LIBRARY</div>
            <div style="display:flex;flex-wrap:wrap;gap:5px;font-family:Courier New;font-size:10px" data-stage="5">
                <span class="slide-chip green">approach_food</span>
                <span class="slide-chip green">avoid_threat</span>
                <span class="slide-chip green">return_shelter</span>
                <span class="slide-chip green">explore_unknown</span>
                <span class="slide-chip green">wait_observe</span>
            </div>
        `
    },
    {
        title: "Brainstem: silent life support",
        regions: ["cerebellum", "brainstem"],
        sentences: [
            "The brainstem keeps you alive while you are not paying attention.",
            "Heartbeat. Breathing. Sleep cycles. The reflex that makes you blink before you know why.",
            "You never thank it. You never even notice it.",
            "Chip's brainstem runs the SAC update, clips gradients, watches for NaNs, and writes a signed HMAC-SHA256 snapshot to disk every hundred ticks.",
            "Pull the plug, restart the process, and Chip wakes up exactly where it left off, with the same drives, the same memories, the same sense of self."
        ],
        slide: `
            <svg class="heartbeat-svg" viewBox="0 0 360 60" style="margin:4px 0 12px" data-stage="0">
                <path d="M0,30 L40,30 L48,30 L52,10 L60,50 L68,30 L80,30 L120,30 L128,30 L132,10 L140,50 L148,30 L160,30 L200,30 L208,30 L212,10 L220,50 L228,30 L240,30 L280,30 L288,30 L292,10 L300,50 L308,30 L360,30"/>
            </svg>

            <div style="font-size:11px;color:#7090b0;font-family:Courier New;margin-bottom:6px;letter-spacing:1px" data-stage="3">AUTONOMIC LOOP</div>
            <div class="slide-stage" data-stage="3"><span class="slide-stage-name">SAC step</span><span class="slide-stage-desc">policy + value update</span></div>
            <div class="slide-stage" data-stage="3"><span class="slide-stage-name">grad clip</span><span class="slide-stage-desc">norm cap = 1.0</span></div>
            <div class="slide-stage" data-stage="3"><span class="slide-stage-name">NaN watch</span><span class="slide-stage-desc">rollback if dirty</span></div>
            <div class="slide-stage" data-stage="3"><span class="slide-stage-name">snapshot</span><span class="slide-stage-desc">every 100 ticks, HMAC-SHA256</span></div>

            <div style="margin-top:12px;padding:10px;background:rgba(5,15,25,0.6);border-radius:6px;font-size:11px;color:#a0d0e0;line-height:1.5;font-family:Inter" data-stage="4">
                Crash, kill the process, reboot the host. Chip wakes up where it left off. Same drives. Same memories. Same sense of self.
            </div>
        `
    },
    {
        title: "One tick, end to end",
        regions: ["thalamus", "amygdala", "hippocampus", "hypothalamus", "cerebrum", "cerebellum", "brainstem"],
        sentences: [
            "Watch them work together for a moment.",
            "A new sentence lands at the thalamus and gets encoded.",
            "The amygdala scores its valence in parallel.",
            "The hippocampus pulls back any relevant past episodes.",
            "The hypothalamus checks whether some drive has gone hungry.",
            "The cerebrum chooses a goal and an action.",
            "The cerebellum smooths the action vector before it leaves.",
            "The brainstem updates the policy and saves state.",
            "Input to output takes about one tick. No region imports another. They only publish typed signals on a shared bus, so any single piece can fail without dragging the rest down with it."
        ],
        slide: `
            <div class="slide-stage" data-stage="1"><span class="slide-stage-name">thalamus</span><span class="slide-stage-desc">encode 512-D</span></div>
            <div class="slide-stage" data-stage="2"><span class="slide-stage-name">amygdala</span><span class="slide-stage-desc">valence + veto</span></div>
            <div class="slide-stage" data-stage="3"><span class="slide-stage-name">hippocampus</span><span class="slide-stage-desc">recall top 3</span></div>
            <div class="slide-stage" data-stage="4"><span class="slide-stage-name">hypothalamus</span><span class="slide-stage-desc">drive check</span></div>
            <div class="slide-stage" data-stage="5"><span class="slide-stage-name">cerebrum</span><span class="slide-stage-desc">goal + action</span></div>
            <div class="slide-stage" data-stage="6"><span class="slide-stage-name">cerebellum</span><span class="slide-stage-desc">EMA smooth</span></div>
            <div class="slide-stage" data-stage="7"><span class="slide-stage-name">brainstem</span><span class="slide-stage-desc">SAC + save</span></div>
            <div style="margin-top:12px;padding:10px;background:rgba(15,15,25,0.7);border-radius:6px;font-size:11px;color:#a0c0ff;line-height:1.5;font-family:Inter" data-stage="8">
                Total wall-clock: ~80ms on CPU. Any region can fail and the rest keep running. The shared bus is the only contract.
            </div>
        `
    },
    {
        title: "What makes Chip different",
        regions: ["cerebrum", "hippocampus", "amygdala"],
        sentences: [
            "What separates Chip from a language model is structure, not scale.",
            "A language model has no drives. It does not get bored, hungry, or curious on its own. It responds when called and forgets when the context window closes.",
            "Chip has none of those limitations by design.",
            "Belief revision happens through spherical linear interpolation, so a contradiction rotates the embedding instead of snapping it. Small contradictions stay quiet. Large ones trigger deliberate review.",
            "Every hundred ticks, Chip writes a short description of its current state, re-encodes it, and anchors that token. This stops the slow identity drift that accumulates in any system trained continuously.",
            "Memory replay does more than rehearse old episodes. It finds key decision points, asks the world model for alternative trajectories, scores them, and keeps the better ones as synthetic memories.",
            "The result is a system that gets better at decisions it has already made, without needing new data from the outside world."
        ],
        slide: `
            <div style="font-size:11px;color:#7090b0;font-family:Courier New;margin-bottom:8px;letter-spacing:1px" data-stage="0">LLM vs CHIP</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:10px" data-stage="1">
                <div style="padding:8px;background:rgba(30,10,10,0.5);border:1px solid #3a2a2a;border-radius:6px;font-size:10px;font-family:Inter;color:#c09090;line-height:1.6">
                    <div style="color:#f7768e;font-family:Courier New;font-size:10px;margin-bottom:4px">LLM</div>
                    no drives<br>no memory<br>no identity<br>sleeps between calls<br>forgets on close
                </div>
                <div style="padding:8px;background:rgba(10,25,10,0.5);border:1px solid #2a3a2a;border-radius:6px;font-size:10px;font-family:Inter;color:#90c090;line-height:1.6">
                    <div style="color:#9ece6a;font-family:Courier New;font-size:10px;margin-bottom:4px">CHIP</div>
                    6 drives<br>episodic memory<br>anchored identity<br>runs continuously<br>persists across restarts
                </div>
            </div>
            <svg viewBox="0 0 360 130" style="margin-bottom:6px" data-stage="3">
                <circle cx="180" cy="65" r="52" fill="none" stroke="#3a4a6a" stroke-width="1" stroke-dasharray="3 3"/>
                <line x1="180" y1="65" x2="232" y2="65" stroke="#7aa2f7" stroke-width="2.5"/>
                <circle cx="232" cy="65" r="3" fill="#7aa2f7"/>
                <text x="236" y="61" fill="#7aa2f7" font-size="9" font-family="Courier New">old belief</text>
                <line x1="180" y1="65" x2="218" y2="22" class="belief-vec"/>
                <circle cx="218" cy="22" r="3" fill="#bb9af7"/>
                <text x="222" y="19" fill="#bb9af7" font-size="9" font-family="Courier New">new belief</text>
                <text x="180" y="124" fill="#5a6a8a" font-size="9" text-anchor="middle" font-family="Inter" font-style="italic">SLERP: contradiction rotates, does not snap</text>
            </svg>
            <div class="slide-stage" data-stage="4"><span class="slide-stage-name">identity</span><span class="slide-stage-desc">re-anchored every 100 ticks</span></div>
            <div class="slide-stage" data-stage="5"><span class="slide-stage-name">replay</span><span class="slide-stage-desc">alt trajectories scored + saved</span></div>
            <div style="margin-top:10px;padding:8px 10px;background:rgba(10,20,10,0.5);border-left:3px solid #9ece6a;border-radius:4px;font-size:11px;color:#90c890;line-height:1.5;font-family:Inter" data-stage="6">
                Gets better at decisions it has already made. No new data required.
            </div>
        `
    },
    {
        title: "The problem Chip is built to solve",
        regions: ["cerebrum", "thalamus", "hippocampus", "hypothalamus", "amygdala"],
        sentences: [
            "Every AI system in production today has the same structural problem.",
            "It wakes up when you call it. It answers. Then it forgets everything and goes back to sleep.",
            "There is no continuity. No accumulation. No sense that the system has been alive between your messages.",
            "That is fine for a search engine. It is not fine for anything that needs to act in the world over time.",
            "A robot that forgets what it learned yesterday is not useful. An agent that cannot feel the pull of an unfinished goal will never finish anything on its own.",
            "The deeper problem is that intelligence without motivation is just a lookup table. Fast, accurate, and completely passive.",
            "Chip is an attempt to build the missing layer: a cognitive substrate that runs continuously, accumulates experience, and generates its own reasons to act.",
            "Not a smarter chatbot. A persistent mind."
        ],
        slide: `
            <div style="font-size:11px;color:#7090b0;font-family:Courier New;margin-bottom:8px;letter-spacing:1px" data-stage="0">THE STRUCTURAL GAP</div>
            <div style="padding:10px;background:rgba(25,10,10,0.6);border:1px solid #3a2020;border-radius:6px;font-size:11px;color:#c09090;line-height:1.7;font-family:Inter;margin-bottom:10px" data-stage="1">
                <span style="color:#f7768e;font-family:Courier New">Current AI loop:</span><br>
                prompt &rarr; respond &rarr; <strong>forget</strong> &rarr; prompt &rarr; respond &rarr; <strong>forget</strong>
            </div>
            <div style="font-size:11px;color:#7090b0;font-family:Courier New;margin:8px 0 4px;letter-spacing:1px" data-stage="3">WHAT IS MISSING</div>
            <div class="slide-stage" data-stage="3"><span class="slide-stage-name">continuity</span><span class="slide-stage-desc">state persists between calls</span></div>
            <div class="slide-stage" data-stage="4"><span class="slide-stage-name">motivation</span><span class="slide-stage-desc">acts without being prompted</span></div>
            <div class="slide-stage" data-stage="5"><span class="slide-stage-name">accumulation</span><span class="slide-stage-desc">experience compounds over time</span></div>
            <div class="slide-stage" data-stage="6"><span class="slide-stage-name">identity</span><span class="slide-stage-desc">stable self across restarts</span></div>
            <div style="margin-top:12px;padding:10px;background:rgba(10,20,30,0.6);border-left:3px solid #7aa2f7;border-radius:4px;font-size:12px;color:#a0c8f0;line-height:1.5;font-family:Inter;font-weight:bold" data-stage="7">
                Intelligence without motivation is just a lookup table.
            </div>
        `
    },
    {
        title: "Chip: a persistent mind",
        regions: ["cerebrum", "amygdala", "hippocampus", "hypothalamus", "thalamus", "cerebellum", "brainstem"],
        sentences: [
            "So here is what Chip actually is.",
            "It is a pure-Python cognitive engine with seven brain regions, each doing the job its biological counterpart does.",
            "The thalamus encodes every input into a 512-dimensional vector. The amygdala scores it emotionally and can veto dangerous actions in under a millisecond.",
            "The hippocampus stores every episode and recalls the three most relevant ones by cosine similarity. The hypothalamus tracks six drives and fires goals when any of them drift.",
            "The cerebrum holds seven working-memory slots, runs a dual-actor policy, generates inner speech in plain English, and anchors identity every hundred ticks.",
            "The cerebellum smooths every action and maintains a skill library. The brainstem runs the training loop, clips gradients, and writes a signed snapshot to disk.",
            "One hundred ninety-one tests cover the full loop. The package is on PyPI as chip-brain. The Docker image is on GitHub Container Registry.",
            "This is not a finished product. It is a foundation. The architecture handles persistence, graceful degradation, and clean restarts. Real capability will come once it trains inside richer environments.",
            "But the structure is right. And the structure is what has been missing."
        ],
        slide: `
            <div style="font-size:11px;color:#7090b0;font-family:Courier New;margin-bottom:8px;letter-spacing:1px" data-stage="0">CHIP AT A GLANCE</div>
            <div class="bar-row" data-stage="1"><span class="bar-label">regions</span><div class="bar-track"><div class="bar-fill high" style="width:100%"></div></div><span class="bar-val">7</span></div>
            <div class="bar-row" data-stage="1"><span class="bar-label">latent dim</span><div class="bar-track"><div class="bar-fill high" style="width:100%"></div></div><span class="bar-val">512</span></div>
            <div class="bar-row" data-stage="1"><span class="bar-label">drives</span><div class="bar-track"><div class="bar-fill mid" style="width:60%"></div></div><span class="bar-val">6</span></div>
            <div class="bar-row" data-stage="1"><span class="bar-label">WM slots</span><div class="bar-track"><div class="bar-fill mid" style="width:70%"></div></div><span class="bar-val">7</span></div>
            <div class="bar-row" data-stage="1"><span class="bar-label">tests</span><div class="bar-track"><div class="bar-fill high" style="width:100%"></div></div><span class="bar-val">191</span></div>
            <div style="display:flex;gap:6px;flex-wrap:wrap;margin:10px 0" data-stage="6">
                <span class="slide-chip">pip install chip-brain</span>
                <span class="slide-chip cyan">ghcr.io/doorman11991/chip</span>
            </div>
            <div style="padding:10px;background:rgba(10,20,10,0.6);border:1px solid #2a3a2a;border-radius:6px;font-size:11px;color:#90c890;line-height:1.6;font-family:Inter;margin-top:6px" data-stage="7">
                Persistence. Graceful degradation. Clean restarts. The architecture is ready. The environment is next.
            </div>
            <div style="margin-top:10px;padding:10px;background:rgba(8,8,20,0.8);border-left:3px solid #bb9af7;border-radius:4px;font-size:13px;color:#d0b8f8;line-height:1.4;font-family:Inter;font-weight:bold;text-align:center" data-stage="8">
                The structure is right.<br>And the structure is what has been missing.
            </div>
        `
    }
];

const VOICE = "en-US-GuyNeural";

let demoActive = false;
let demoVoiceEnabled = true;
let demoCurrentScene = -1;
let demoPlaybackToken = 0;          // increments on start/stop to invalidate stale callbacks
let demoSubtitleDetach = null;       // function to remove timeupdate listener
let demoLoader = null;               // TTSLoader instance, recreated per run

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

// Progressive TTS loader. One Audio per scene, lazy + idempotent.
class TTSLoader {
    constructor(scenes, voice) {
        this.scenes = scenes;
        this.voice = voice;
        this.audios = new Array(scenes.length).fill(null);
        this.promises = new Array(scenes.length).fill(null);
    }

    _narrationFor(idx) {
        return this.scenes[idx].sentences.join(' ');
    }

    load(idx) {
        if (idx < 0 || idx >= this.scenes.length) return Promise.resolve(null);
        if (this.audios[idx]) return Promise.resolve(this.audios[idx]);
        if (this.promises[idx]) return this.promises[idx];

        const url = '/tts?text=' + encodeURIComponent(this._narrationFor(idx)) + '&voice=' + encodeURIComponent(this.voice);
        const audio = new Audio(url);
        audio.preload = 'auto';

        this.promises[idx] = new Promise((resolve, reject) => {
            const onReady = () => { this.audios[idx] = audio; resolve(audio); };
            const onErr = (e) => { this.promises[idx] = null; reject(e); };
            audio.addEventListener('canplaythrough', onReady, { once: true });
            audio.addEventListener('loadeddata', onReady, { once: true });
            audio.addEventListener('error', onErr, { once: true });
            audio.load();
        });
        return this.promises[idx];
    }

    prefetch(idx) {
        this.load(idx).catch(() => {});
    }

    dispose() {
        for (const a of this.audios) {
            if (a) { try { a.pause(); a.src = ''; } catch (e) {} }
        }
        this.audios = [];
        this.promises = [];
    }
}

function attachSubtitles(audio, sentences, stageEls) {
    const bar = document.getElementById('demo-subtitle-bar');
    bar.classList.add('visible');
    bar.innerHTML = '<div class="sub-line">' + escapeHtml(sentences[0]) + '</div>';

    const lengths = sentences.map(s => Math.max(1, s.length));
    const total = lengths.reduce((a, b) => a + b, 0);
    let cum = 0;
    const boundaries = lengths.map(l => { cum += l; return cum / total; });

    // Pre-bucket stage elements by sentence index
    const buckets = sentences.map(() => []);
    if (stageEls) {
        for (const el of stageEls) {
            const idx = parseInt(el.dataset.stage, 10);
            if (!isNaN(idx) && idx >= 0 && idx < buckets.length) buckets[idx].push(el);
        }
    }

    let lastIdx = -1;
    const handler = () => {
        const dur = audio.duration;
        if (!isFinite(dur) || dur === 0) return;
        const frac = audio.currentTime / dur;
        let idx = boundaries.findIndex(b => frac < b);
        if (idx === -1) idx = sentences.length - 1;
        if (idx !== lastIdx) {
            lastIdx = idx;
            bar.innerHTML = '<div class="sub-line">' + escapeHtml(sentences[idx]) + '</div>';
            // Reveal stage elements up to and including this sentence
            if (stageEls) {
                for (let i = 0; i <= idx; i++) {
                    for (const el of buckets[i]) {
                        if (!el.classList.contains('lit')) {
                            el.classList.add('lit');
                            // Re-trigger card-slide animation for late reveals
                            el.style.opacity = '';
                            el.style.animation = 'none';
                            void el.offsetWidth;
                            el.style.animation = '';
                        }
                    }
                }
            }
        }
    };
    audio.addEventListener('timeupdate', handler);
    return () => audio.removeEventListener('timeupdate', handler);
}

function clearSubtitles() {
    if (demoSubtitleDetach) { try { demoSubtitleDetach(); } catch (e) {} demoSubtitleDetach = null; }
    const bar = document.getElementById('demo-subtitle-bar');
    bar.classList.remove('visible');
    bar.innerHTML = '';
}

function toggleDemo() {
    if (demoActive) { stopDemo(); } else { startDemo(); }
}

function toggleVoice() {
    demoVoiceEnabled = !demoVoiceEnabled;
    document.getElementById('voice-btn').textContent = demoVoiceEnabled ? '🔊 Voice ON' : '🔇 Voice OFF';
    if (!demoVoiceEnabled && window._currentTtsAudio) {
        window._currentTtsAudio.pause();
    }
}

async function startDemo() {
    demoActive = true;
    demoCurrentScene = -1;
    demoPlaybackToken++;
    const myToken = demoPlaybackToken;

    document.getElementById('demo-overlay').classList.add('active');
    document.getElementById('demo-btn').textContent = '⏸ Stop Demo';
    document.getElementById('demo-loading').classList.add('visible');
    document.getElementById('demo-loading').textContent = 'Loading first chapter';
    document.getElementById('demo-label').textContent = '';

    demoLoader = new TTSLoader(SCENES, VOICE);

    // Load scene 0 first; kick off scene 1 & 2 in parallel so the queue stays warm.
    try {
        await demoLoader.load(0);
        demoLoader.prefetch(1);
        demoLoader.prefetch(2);
    } catch (e) {
        console.warn('TTS pre-load failed for scene 0:', e);
    }
    if (myToken !== demoPlaybackToken) return;

    document.getElementById('demo-loading').classList.remove('visible');
    playScene(0, myToken);
}

function stopDemo() {
    demoActive = false;
    demoPlaybackToken++;
    document.getElementById('demo-overlay').classList.remove('active');
    document.getElementById('demo-btn').textContent = '▶ Play Demo';
    document.getElementById('demo-loading').classList.remove('visible');
    const slidePanel = document.getElementById('demo-slide');
    slidePanel.classList.remove('visible');
    slidePanel.classList.remove('swapping');
    document.getElementById('slide-body').innerHTML = '';
    document.querySelectorAll('.brain-region').forEach(r => r.classList.remove('demo-active'));
    document.getElementById('demo-label').textContent = '';
    clearSubtitles();
    if (window._currentTtsAudio) {
        try { window._currentTtsAudio.pause(); } catch (e) {}
        window._currentTtsAudio = null;
    }
    if (demoLoader) { demoLoader.dispose(); demoLoader = null; }
}

async function playScene(idx, token) {
    if (!demoActive || token !== demoPlaybackToken) return;
    if (idx >= SCENES.length) { stopDemo(); return; }

    const scene = SCENES[idx];
    demoCurrentScene = idx;

    // Region highlights
    document.querySelectorAll('.brain-region').forEach(r => r.classList.remove('demo-active'));
    scene.regions.forEach(rid => {
        const el = document.getElementById('reg-' + rid);
        if (el) el.classList.add('demo-active');
    });

    // Title bar
    document.getElementById('demo-label').textContent = scene.title || '';

    // Slide panel: swap content with transition
    const slidePanel = document.getElementById('demo-slide');
    const slideHeader = document.getElementById('slide-h');
    const slideBody = document.getElementById('slide-body');
    slidePanel.classList.add('swapping');
    setTimeout(() => {
        if (token !== demoPlaybackToken) return;
        slideHeader.textContent = scene.title || ('Chapter ' + (idx + 1));
        slideBody.innerHTML = scene.slide || '';
        slidePanel.classList.remove('swapping');
        slidePanel.classList.add('visible');
    }, 300);

    // Progress bar fills as we move through scenes
    const progress = ((idx + 1) / SCENES.length) * 100;
    document.getElementById('demo-progress').style.width = progress + '%';
    document.getElementById('demo-time').textContent = `Chapter ${idx + 1} of ${SCENES.length}`;

    // Background prefetch for upcoming scenes (concurrency cap = 2 ahead)
    if (demoLoader) {
        demoLoader.prefetch(idx + 1);
        demoLoader.prefetch(idx + 2);
    }

    clearSubtitles();

    // Wait for the swap to settle so the slide DOM is in place before we light stages
    await new Promise(r => setTimeout(r, 350));
    if (token !== demoPlaybackToken) return;
    const stageEls = Array.from(slideBody.querySelectorAll('[data-stage]'));

    if (demoVoiceEnabled && demoLoader) {
        let audio = demoLoader.audios[idx];
        if (!audio) {
            // Audio not ready yet. Show loader, wait for it.
            document.getElementById('demo-loading').textContent = `Loading chapter ${idx + 1}`;
            document.getElementById('demo-loading').classList.add('visible');
            try { audio = await demoLoader.load(idx); }
            catch (e) { console.warn('TTS load failed:', e); }
            document.getElementById('demo-loading').classList.remove('visible');
            if (token !== demoPlaybackToken) return;
        }

        if (audio) {
            window._currentTtsAudio = audio;
            audio.currentTime = 0;
            demoSubtitleDetach = attachSubtitles(audio, scene.sentences, stageEls);
            audio.onended = () => {
                if (token !== demoPlaybackToken) return;
                clearSubtitles();
                setTimeout(() => playScene(idx + 1, token), 600);
            };
            audio.onerror = () => {
                if (token !== demoPlaybackToken) return;
                setTimeout(() => playScene(idx + 1, token), 2500);
            };
            try { await audio.play(); }
            catch (err) {
                console.warn('TTS playback failed:', err);
                setTimeout(() => playScene(idx + 1, token), 2500);
            }
            return;
        }
    }

    // Fallback when voice is off or audio failed: show subtitles + reveal stages on a fixed schedule.
    const bar = document.getElementById('demo-subtitle-bar');
    bar.classList.add('visible');
    let i = 0;
    const showNext = () => {
        if (token !== demoPlaybackToken) return;
        if (i >= scene.sentences.length) {
            clearSubtitles();
            setTimeout(() => playScene(idx + 1, token), 400);
            return;
        }
        bar.innerHTML = '<div class="sub-line">' + escapeHtml(scene.sentences[i]) + '</div>';
        for (const el of stageEls) {
            const sIdx = parseInt(el.dataset.stage, 10);
            if (!isNaN(sIdx) && sIdx <= i) el.classList.add('lit');
        }
        const dur = Math.max(2200, scene.sentences[i].length * 55);
        i++;
        setTimeout(showNext, dur);
    };
    showNext();
}
</script>
</div></div></body></html>"""


# ===== ARENA =====

ARENA_HTML = """<!DOCTYPE html>
<html><head><title>Chip Survival Arena</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#0a0a0f; color:#e0e0e0; font-family:'Courier New',monospace; display:flex; }
#nav { width:60px; background:#08080f; border-right:1px solid #1a1a2e; display:flex; flex-direction:column; align-items:center; padding:12px 0; gap:16px; }
.nav-icon { width:40px; height:40px; border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:18px; text-decoration:none; background:#0f0f1a; border:1px solid #1a1a2e; transition:all 0.2s; }
.nav-icon:hover { background:#1a2a4a; border-color:#7aa2f7; }
.nav-icon.active { background:#1a2a4a; border-color:#7aa2f7; box-shadow:0 0 8px rgba(122,162,247,0.3); }
#main { flex:1; display:flex; }
#grid { display:grid; grid-template-columns:repeat(10,40px); grid-template-rows:repeat(10,40px); gap:2px; padding:20px; }
.cell { background:#12121a; border:1px solid #1a1a2e; display:flex; align-items:center; justify-content:center; font-size:18px; border-radius:4px; transition:background 0.2s; }
.cell.agent { background:#1a3a5a; border-color:#7aa2f7; }
.cell.food { background:#1a3a2a; }
.cell.threat { background:#3a1a1a; }
.cell.shelter { background:#2a2a1a; }
#sidebar { padding:20px; width:300px; }
#sidebar h2 { color:#7aa2f7; font-size:13px; margin:12px 0 6px; text-transform:uppercase; }
.stat { display:flex; justify-content:space-between; font-size:12px; padding:2px 0; }
.stat-val { color:#7aa2f7; }
#log { font-size:11px; color:#565f89; max-height:200px; overflow-y:auto; margin-top:12px; }
#log div { padding:1px 0; }

/* Shared sidebar */
#nav-sidebar { width:240px; min-width:240px; background:#08080f; border-right:1px solid #1a1a2e; padding:20px 16px; display:flex; flex-direction:column; gap:8px; z-index:50; }
#nav-sidebar h2 { color:#7aa2f7; font-size:11px; text-transform:uppercase; letter-spacing:3px; margin-bottom:12px; font-weight:bold; }
.nav-btn { display:block; padding:14px 18px; background:rgba(15,15,26,0.6); border:1px solid #1a1a2e; border-radius:8px; color:#a0c0ff; text-decoration:none; font-size:13px; font-family:'Courier New',monospace; transition:all 0.2s; cursor:pointer; text-align:left; }
.nav-btn:hover { background:rgba(26,42,74,0.6); border-color:#5070a0; color:#c0d0ff; }
.nav-btn.active { background:rgba(26,42,74,0.8); border-color:#7aa2f7; color:#c0d8ff; box-shadow:0 0 12px rgba(122,162,247,0.3); }
</style></head><body>
<div style="display:flex;height:100vh;width:100vw;">
<div id="nav-sidebar">
    <h2>CHIP BRAIN</h2>
    <a class="nav-btn" href="/brain">Brain Visualizer</a>
    <a class="nav-btn active" href="/arena">Survival Arena</a>
    <a class="nav-btn" href="/voice">Voice Assistant</a>
</div>
<div style="flex:1;display:flex;flex-direction:column;overflow:auto;position:relative;">
<div id="main">
<div id="grid"></div>
<div id="sidebar">
<h2>Survival Arena</h2>
<div class="stat"><span>Tick</span><span class="stat-val" id="a-tick">0</span></div>
<div class="stat"><span>Health</span><span class="stat-val" id="a-health">100</span></div>
<div class="stat"><span>Food</span><span class="stat-val" id="a-food">0</span></div>
<div class="stat"><span>Mood</span><span class="stat-val" id="a-mood">Calm</span></div>
<div class="stat"><span>Goal</span><span class="stat-val" id="a-goal">none</span></div>
<h2>Agent Log</h2>
<div id="log"></div>
<button onclick="fetch('/arena/step')" style="margin-top:12px;padding:6px 12px;background:#7aa2f7;border:none;color:#0a0a0f;border-radius:4px;cursor:pointer">Step</button>
<button onclick="autoRun()" style="margin-left:8px;padding:6px 12px;background:#9ece6a;border:none;color:#0a0a0f;border-radius:4px;cursor:pointer">Auto</button>
</div>
<script>
let autoInterval = null;
function autoRun() { if(autoInterval){clearInterval(autoInterval);autoInterval=null;} else {autoInterval=setInterval(()=>fetch('/arena/step'),500);} }
const evtSource = new EventSource('/arena/stream');
evtSource.onmessage = (e) => {
    const d = JSON.parse(e.data);
    // Render grid
    let html = '';
    for(let y=0;y<10;y++) for(let x=0;x<10;x++) {
        let cls='cell', content='';
        const key=x+','+y;
        if(d.agent_pos===key){cls+=' agent';content='🧠';}
        else if(d.food?.includes(key)){cls+=' food';content='🍎';}
        else if(d.threats?.includes(key)){cls+=' threat';content='⚡';}
        else if(d.shelter?.includes(key)){cls+=' shelter';content='🏠';}
        html+=`<div class="${cls}">${content}</div>`;
    }
    document.getElementById('grid').innerHTML=html;
    document.getElementById('a-tick').textContent=d.tick||0;
    document.getElementById('a-health').textContent=d.health||0;
    document.getElementById('a-food').textContent=d.food_collected||0;
    document.getElementById('a-mood').textContent=d.mood||'?';
    document.getElementById('a-goal').textContent=d.goal||'none';
    if(d.log) { const log=document.getElementById('log'); log.innerHTML=d.log.map(l=>`<div>${l}</div>`).join(''); log.scrollTop=log.scrollHeight; }
};
</script></div></div></div></body></html>"""


# ===== VOICE ASSISTANT =====

VOICE_HTML = """<!DOCTYPE html>
<html><head><title>Chip Voice Assistant</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#0a0a0f; color:#e0e0e0; font-family:'Courier New',monospace; display:flex; flex-direction:column; align-items:center; padding:40px; }
h1 { color:#7aa2f7; margin-bottom:20px; }
.factors { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; width:100%; max-width:700px; margin:20px 0; }
.factor { background:#12121a; border:1px solid #1a1a2e; border-radius:6px; padding:10px; text-align:center; transition:all 0.3s; }
.factor.active { border-color:#7aa2f7; background:#1a2a3a; }
.factor-name { font-size:10px; color:#565f89; text-transform:uppercase; }
.factor-val { font-size:14px; color:#7aa2f7; margin-top:4px; }
#chat { width:100%; max-width:700px; margin:20px 0; }
.msg { padding:8px 12px; margin:4px 0; border-radius:8px; font-size:13px; }
.msg.user { background:#1a2a3a; text-align:right; }
.msg.chip { background:#1a3a2a; }
.msg.thinking { color:#565f89; font-style:italic; font-size:11px; }
#input-row { display:flex; gap:8px; width:100%; max-width:700px; }
#input-row input { flex:1; background:#12121a; border:1px solid #2a2a3a; color:#e0e0e0; padding:10px; border-radius:6px; font-size:14px; }
#input-row button { padding:10px 20px; background:#7aa2f7; border:none; color:#0a0a0f; border-radius:6px; cursor:pointer; font-weight:bold; }
#timing { color:#565f89; font-size:11px; margin-top:8px; }

/* Shared sidebar */
#nav-sidebar { width:240px; min-width:240px; background:#08080f; border-right:1px solid #1a1a2e; padding:20px 16px; display:flex; flex-direction:column; gap:8px; z-index:50; }
#nav-sidebar h2 { color:#7aa2f7; font-size:11px; text-transform:uppercase; letter-spacing:3px; margin-bottom:12px; font-weight:bold; }
.nav-btn { display:block; padding:14px 18px; background:rgba(15,15,26,0.6); border:1px solid #1a1a2e; border-radius:8px; color:#a0c0ff; text-decoration:none; font-size:13px; font-family:'Courier New',monospace; transition:all 0.2s; cursor:pointer; text-align:left; }
.nav-btn:hover { background:rgba(26,42,74,0.6); border-color:#5070a0; color:#c0d0ff; }
.nav-btn.active { background:rgba(26,42,74,0.8); border-color:#7aa2f7; color:#c0d8ff; box-shadow:0 0 12px rgba(122,162,247,0.3); }
</style></head><body>
<div style="display:flex;height:100vh;width:100vw;">
<div id="nav-sidebar">
    <h2>CHIP BRAIN</h2>
    <a class="nav-btn" href="/brain">Brain Visualizer</a>
    <a class="nav-btn" href="/arena">Survival Arena</a>
    <a class="nav-btn active" href="/voice">Voice Assistant</a>
</div>
<div style="flex:1;display:flex;flex-direction:column;overflow:auto;position:relative;">
<h1>Chip Voice Assistant</h1>
<p style="color:#565f89;font-size:12px;margin-bottom:20px">9 cognitive factors computed in real-time, shaping every response</p>
<div class="factors" id="factors"></div>
<div id="chat"></div>
<div id="input-row">
<input id="msg" placeholder="Say something..." onkeydown="if(event.key==='Enter')send()">
<button onclick="send()">Send</button>
</div>
<div id="timing"></div>
<script>
const FACTOR_NAMES = ['mood','confidence','goal','curiosity','energy','safety','novelty','strain','recalled'];
document.getElementById('factors').innerHTML = FACTOR_NAMES.map(f=>`<div class="factor" id="f-${f}"><div class="factor-name">${f}</div><div class="factor-val" id="fv-${f}">-</div></div>`).join('');

async function send() {
    const input = document.getElementById('msg');
    const text = input.value.trim();
    if(!text) return;
    input.value = '';
    addMsg('user', text);
    addMsg('thinking', 'Chip is thinking...');

    const t0 = performance.now();
    const resp = await fetch('/voice/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({message:text})});
    const data = await resp.json();
    const dt = performance.now() - t0;

    // Remove thinking indicator
    const chat = document.getElementById('chat');
    chat.removeChild(chat.lastChild);

    addMsg('chip', data.response);
    document.getElementById('timing').textContent = `Brain: ${data.brain_ms}ms | LLM: ${data.llm_ms}ms | Total: ${dt.toFixed(0)}ms`;

    // Update factors
    if(data.factors) {
        Object.entries(data.factors).forEach(([k,v])=>{
            const el = document.getElementById('fv-'+k);
            if(el) el.textContent = typeof v==='number' ? v.toFixed(2) : v;
            const card = document.getElementById('f-'+k);
            if(card) { card.classList.add('active'); setTimeout(()=>card.classList.remove('active'),1000); }
        });
    }
    if(data.thought) addMsg('thinking', '💭 ' + data.thought);
}

function addMsg(type, text) {
    const chat = document.getElementById('chat');
    const div = document.createElement('div');
    div.className = 'msg ' + type;
    div.textContent = text;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}
</script></div></div></body></html>"""


# ---------------------------------------------------------------------------
# Arena state
# ---------------------------------------------------------------------------

import random

class ArenaState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.agent_pos = [5, 5]
        self.health = 100
        self.food_collected = 0
        self.tick = 0
        self.food = [[random.randint(0,9), random.randint(0,9)] for _ in range(5)]
        self.threats = [[random.randint(0,9), random.randint(0,9)] for _ in range(3)]
        self.shelter = [[2, 2], [7, 7]]
        self.log = []

    def step(self, action_vec):
        self.tick += 1
        # Map continuous action to grid movement
        dx = 1 if action_vec[0] > 0.3 else (-1 if action_vec[0] < -0.3 else 0)
        dy = 1 if action_vec[1] > 0.3 else (-1 if action_vec[1] < -0.3 else 0)
        self.agent_pos[0] = max(0, min(9, self.agent_pos[0] + dx))
        self.agent_pos[1] = max(0, min(9, self.agent_pos[1] + dy))

        reward = -0.01  # time cost
        self.health -= 0.5

        # Food
        for f in self.food[:]:
            if f == self.agent_pos:
                self.food.remove(f)
                self.food.append([random.randint(0,9), random.randint(0,9)])
                self.food_collected += 1
                self.health = min(100, self.health + 20)
                reward += 1.0
                self.log.append(f"t{self.tick}: ate food!")

        # Threats
        for t in self.threats:
            if t == self.agent_pos:
                self.health -= 20
                reward -= 1.0
                self.log.append(f"t{self.tick}: hit by threat!")

        # Shelter
        for s in self.shelter:
            if s == self.agent_pos:
                reward += 0.1

        if len(self.log) > 20:
            self.log = self.log[-20:]

        done = self.health <= 0
        if done:
            self.log.append(f"t{self.tick}: DIED")
        return reward, done

    def to_dict(self, mood="Calm", goal="none"):
        return {
            "agent_pos": f"{self.agent_pos[0]},{self.agent_pos[1]}",
            "food": [f"{f[0]},{f[1]}" for f in self.food],
            "threats": [f"{t[0]},{t[1]}" for t in self.threats],
            "shelter": [f"{s[0]},{s[1]}" for s in self.shelter],
            "tick": self.tick,
            "health": int(self.health),
            "food_collected": self.food_collected,
            "mood": mood,
            "goal": goal,
            "log": self.log,
        }


arena = ArenaState()
arena_events = []


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return """<!DOCTYPE html><html><head><title>Chip Demos</title>
<style>body{background:#0a0a0f;color:#e0e0e0;font-family:'Courier New',monospace;display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh;}
a{color:#7aa2f7;text-decoration:none;font-size:18px;padding:12px 24px;margin:8px;border:1px solid #2a2a3a;border-radius:8px;display:block;transition:all 0.2s;}
a:hover{background:#1a2a3a;border-color:#7aa2f7;}h1{margin-bottom:24px;}</style></head><body>
<h1>Chip Demos</h1>
<a href="/brain">Brain Visualizer</a>
<a href="/arena">Survival Arena</a>
<a href="/voice">Voice Assistant</a>
</body></html>"""

@app.route("/brain")
def brain_page():
    return BRAIN_HTML

# ---------------------------------------------------------------------------
# edge-tts - high-quality neural narration via Microsoft's Edge cloud voices.
# Free, no API key needed, ~3KB/sec MP3 output. Requires internet.
# ---------------------------------------------------------------------------
_tts_lock = threading.Lock()
_tts_cache: Dict[str, bytes] = {}

# Default voice - natural-sounding US male. Other good options:
#   en-US-AriaNeural       (female, warm)
#   en-US-GuyNeural        (male, professional)
#   en-US-JennyNeural      (female, friendly)
#   en-GB-RyanNeural       (UK male)
#   en-AU-WilliamNeural    (AU male)
DEFAULT_VOICE = "en-US-GuyNeural"


def _generate_tts_sync(text: str, voice: str) -> bytes:
    """Synchronously generate MP3 audio via edge-tts. Runs the async API in a fresh loop."""
    import asyncio
    import edge_tts

    async def _run() -> bytes:
        comm = edge_tts.Communicate(text, voice=voice)
        chunks: list[bytes] = []
        async for part in comm.stream():
            if part.get("type") == "audio":
                chunks.append(part["data"])
        return b"".join(chunks)

    # Run the async coroutine in this thread's event loop
    try:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_run())
        finally:
            loop.close()
    except Exception:
        # If we somehow already have a loop, fall back to a thread
        import concurrent.futures
        def _wrapper():
            return asyncio.run(_run())
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_wrapper).result(timeout=30)


@app.route("/tts")
def tts():
    """Generate MP3 audio for a text string using edge-tts. Cached per-text."""
    text = request.args.get("text", "").strip()
    voice = request.args.get("voice", DEFAULT_VOICE)
    if not text:
        return Response(b"", status=400)

    cache_key = f"{voice}:{text}"
    if cache_key in _tts_cache:
        return Response(_tts_cache[cache_key], mimetype="audio/mpeg")

    # Serialize generation so we don't fire 15 parallel network requests on demo start
    with _tts_lock:
        # Re-check cache inside lock (another thread may have generated it)
        if cache_key in _tts_cache:
            return Response(_tts_cache[cache_key], mimetype="audio/mpeg")
        try:
            mp3_bytes = _generate_tts_sync(text, voice)
            if not mp3_bytes:
                return Response(b"empty audio", status=500)
            _tts_cache[cache_key] = mp3_bytes
            return Response(mp3_bytes, mimetype="audio/mpeg")
        except Exception as e:
            print(f"[TTS] error: {type(e).__name__}: {e}")
            return Response(str(e).encode(), status=500)

@app.route("/brain/stream")
def brain_stream():
    def gen():
        brain = get_brain()
        while True:
            data = json.dumps(brain.status())
            yield f"data: {data}\n\n"
            time.sleep(1)
    return Response(gen(), mimetype="text/event-stream")

@app.route("/brain/tick", methods=["POST"])
def brain_tick():
    brain = get_brain()
    text = request.json.get("text", "idle observation")
    brain.tick(text)
    return jsonify(brain.status())

@app.route("/arena")
def arena_page():
    return ARENA_HTML

@app.route("/arena/stream")
def arena_stream():
    def gen():
        while True:
            brain = get_brain()
            mood = brain.emotions.current_mood()[0]
            goal_obj = brain.goal_stack.current_goal()
            goal = goal_obj.name if goal_obj else "none"
            data = json.dumps(arena.to_dict(mood=mood, goal=goal))
            yield f"data: {data}\n\n"
            time.sleep(0.3)
    return Response(gen(), mimetype="text/event-stream")

@app.route("/arena/step")
def arena_step():
    brain = get_brain()
    obs = f"Position {arena.agent_pos}, health {arena.health:.0f}, food nearby {len(arena.food)}"
    action = brain.tick(obs)
    a = action.squeeze(0).tolist()
    reward, done = arena.step(a)
    brain.train_step(reward=reward, done=done)
    if done:
        arena.reset()
    return jsonify({"ok": True})

@app.route("/voice")
def voice_page():
    return VOICE_HTML

@app.route("/voice/chat", methods=["POST"])
def voice_chat():
    brain = get_brain()
    user_msg = request.json.get("message", "")

    # 1. Brain tick - computes all 9 factors in <100ms
    t0 = time.time()
    brain.tick(user_msg)
    brain_ms = int((time.time() - t0) * 1000)

    # 2. Extract the 9 cognitive factors
    mood, _ = brain.emotions.current_mood()
    confidence = brain.meta.mean_confidence()
    goal_obj = brain.goal_stack.current_goal()
    goal_name = goal_obj.name if goal_obj else "none"
    drives = brain.homeostasis.status()
    novelty = brain.habituation._last_novelty
    strain = float(brain.homeostasis.strain().item())
    recent_thought = brain.inner_speech.recent(1)
    thought_text = recent_thought[0].text if recent_thought else ""
    recall_count = len([s for s in brain.working_mem._slots if s.source_tag == "hippocampus_recall"])

    factors = {
        "mood": mood,
        "confidence": confidence,
        "goal": goal_name,
        "curiosity": drives.get("curiosity", 0.5),
        "energy": drives.get("energy", 0.5),
        "safety": drives.get("safety", 0.5),
        "novelty": novelty,
        "strain": strain,
        "recalled": recall_count,
    }

    # 3. Build system prompt and query LLM
    system_prompt = build_system_prompt(brain)
    t1 = time.time()
    response = query_llm(system_prompt, user_msg)
    llm_ms = int((time.time() - t1) * 1000)

    # 4. Small positive reward for engagement
    brain.train_step(reward=0.05, done=False)

    return jsonify({
        "response": response,
        "factors": factors,
        "thought": thought_text,
        "brain_ms": brain_ms,
        "llm_ms": llm_ms,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Chip Demo Server")
    print("=" * 60)
    print()
    print("  http://localhost:8080/brain   - Brain Visualizer")
    print("  http://localhost:8080/arena   - Survival Arena")
    print("  http://localhost:8080/voice   - Voice Assistant")
    print()
    print("  Starting brain (first load downloads granite ~250MB)...")
    get_brain()
    print("  Ready!")
    print()
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)


if __name__ == "__main__":
    main()
