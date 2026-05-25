"""
demo/app.py — Three animated Chip demos in one file.

Run:
    python demo/app.py

Then open:
    http://localhost:8080/brain      — Real-time brain visualizer
    http://localhost:8080/arena      — Survival arena with training
    http://localhost:8080/voice      — Voice assistant (Chip + LLM + TTS/STT)

Dependencies (beyond chip-brain):
    pip install flask

Optional (for voice demo):
    pip install RealtimeSTT KittenTTS

The voice demo uses your local LM Studio at http://10.0.0.20:1234/v1
for language generation. Chip computes 9 cognitive factors in <100ms
and shapes the LLM's response accordingly.
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
#sidebar { width:280px; background:#08080f; border-right:1px solid #1a1a2e; padding:16px; overflow-y:auto; display:flex; flex-direction:column; gap:12px; z-index:10; }
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
.brain-region { cursor:pointer; transition:all 0.3s ease; opacity:0.6; }
.brain-region:hover { opacity:1; transform:scale(1.05); }
.brain-region.active { opacity:1; }

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
</style></head><body>

<div id="sidebar">
    <h2>Chip Brain</h2>
    <a class="nav-btn active" href="/brain">Brain Visualizer</a>
    <a class="nav-btn" href="/arena">Survival Arena</a>
    <a class="nav-btn" href="/voice">Voice Assistant</a>

    <h2 style="margin-top:16px">Mood</h2>
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
    <svg viewBox="0 0 600 700" width="85%" height="92%" xmlns="http://www.w3.org/2000/svg">
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

        <!-- Connection paths (smooth curves, clipped to brain) -->
        <g mask="url(#brainMask)">
            <path class="signal-path" id="sig-thal-cer" d="M285,275 Q260,240 230,210 Q210,185 220,160" stroke="#7aa2f7" fill="none" stroke-width="1.6" opacity="0.7"/>
            <path class="signal-path" id="sig-thal-cer2" d="M315,275 Q340,240 370,210 Q390,185 380,160" stroke="#7aa2f7" fill="none" stroke-width="1.6" opacity="0.7"/>
            <path class="signal-path" id="sig-thal-amy" d="M285,295 Q260,310 235,330" stroke="#f7768e" fill="none" stroke-width="1.6" opacity="0.7"/>
            <path class="signal-path" id="sig-hip-thal" d="M225,415 Q255,360 285,310" stroke="#e0af68" fill="none" stroke-width="1.6" opacity="0.7"/>
            <path class="signal-path" id="sig-hyp-cer" d="M310,355 Q335,310 360,250 Q380,210 360,170" stroke="#bb9af7" fill="none" stroke-width="1.6" opacity="0.7"/>
            <path class="signal-path" id="sig-cer-cbl" d="M380,200 Q415,260 415,330 Q420,395 405,420" stroke="#9ece6a" fill="none" stroke-width="1.6" opacity="0.7"/>
            <path class="signal-path" id="sig-bs-hyp" d="M300,495 Q300,440 305,395" stroke="#7dcfff" fill="none" stroke-width="1.6" opacity="0.7"/>
            <path class="signal-path" id="sig-amy-hip" d="M210,355 Q210,385 215,415" stroke="#f7768e" fill="none" stroke-width="1.2" opacity="0.4"/>
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
            REGIONS — each defined by a single (cx, cy) so aura and core are perfectly aligned.
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
</script>
</body></html>"""


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
</style></head><body>
<div id="nav">
<a class="nav-icon" href="/brain" title="Brain">🧠</a>
<a class="nav-icon active" href="/arena" title="Arena">⚔️</a>
<a class="nav-icon" href="/voice" title="Voice">🎙️</a>
</div>
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
</script></div></body></html>"""


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
</style></head><body>
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
</script></body></html>"""


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

    # 1. Brain tick — computes all 9 factors in <100ms
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
    print("  http://localhost:8080/brain   — Brain Visualizer")
    print("  http://localhost:8080/arena   — Survival Arena")
    print("  http://localhost:8080/voice   — Voice Assistant")
    print()
    print("  Starting brain (first load downloads granite ~250MB)...")
    get_brain()
    print("  Ready!")
    print()
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)


if __name__ == "__main__":
    main()
