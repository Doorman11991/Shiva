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
body { background:#0a0a0f; color:#e0e0e0; font-family:'Courier New',monospace; overflow:hidden; }
.container { display:grid; grid-template-columns:1fr 1fr; grid-template-rows:1fr 1fr; height:100vh; gap:4px; padding:4px; }
.panel { background:#12121a; border:1px solid #2a2a3a; border-radius:8px; padding:12px; overflow-y:auto; }
.panel h2 { color:#7aa2f7; font-size:13px; margin-bottom:8px; text-transform:uppercase; letter-spacing:1px; }
.meter { height:6px; background:#1a1a2e; border-radius:3px; margin:4px 0; }
.meter-fill { height:100%; border-radius:3px; transition:width 0.3s; }
.mood { font-size:24px; text-align:center; padding:8px; }
.thought { color:#9ece6a; font-style:italic; padding:4px 0; border-bottom:1px solid #1a1a2e; font-size:12px; }
.signal { color:#565f89; font-size:11px; padding:2px 0; }
.stat { display:flex; justify-content:space-between; padding:2px 0; font-size:12px; }
.stat-val { color:#7aa2f7; }
#regions { display:flex; flex-wrap:wrap; gap:8px; justify-content:center; padding:8px; }
.region { padding:8px 12px; border-radius:6px; background:#1a1a2e; font-size:11px; transition:all 0.3s; }
.region.active { background:#1a3a5a; border:1px solid #7aa2f7; box-shadow:0 0 8px rgba(122,162,247,0.3); }
</style></head><body>
<div class="container">
<div class="panel">
<h2>Brain Regions</h2>
<div id="regions"></div>
<div class="mood" id="mood">...</div>
<div id="drives"></div>
</div>
<div class="panel">
<h2>Inner Speech</h2>
<div id="thoughts"></div>
</div>
<div class="panel">
<h2>Working Memory</h2>
<div id="wm"></div>
<h2 style="margin-top:12px">Goals</h2>
<div id="goals"></div>
</div>
<div class="panel">
<h2>Signals</h2>
<div id="signals"></div>
<h2 style="margin-top:12px">Stats</h2>
<div id="stats"></div>
</div>
</div>
<script>
const REGIONS = ['thalamus','amygdala','hippocampus','hypothalamus','cerebrum','cerebellum','brainstem'];
const MOOD_EMOJI = {Calm:'😌',Happy:'😊',Sad:'😔',Angry:'😠'};
document.getElementById('regions').innerHTML = REGIONS.map(r=>`<div class="region" id="r-${r}">${r}</div>`).join('');

function update(data) {
    document.getElementById('mood').textContent = (MOOD_EMOJI[data.mood]||'🧠') + ' ' + data.mood;
    // Drives
    let dh = '';
    if(data.homeostasis) Object.entries(data.homeostasis).forEach(([k,v])=>{
        const pct = (v*100).toFixed(0);
        const color = v>0.6?'#9ece6a':v>0.3?'#e0af68':'#f7768e';
        dh += `<div class="stat"><span>${k}</span><span class="stat-val">${pct}%</span></div><div class="meter"><div class="meter-fill" style="width:${pct}%;background:${color}"></div></div>`;
    });
    document.getElementById('drives').innerHTML = dh;
    // Thoughts
    if(data.inner_speech && data.inner_speech.recent) {
        let th = data.inner_speech.recent.map(t=>`<div class="thought">${t.text||JSON.stringify(t)}</div>`).join('');
        document.getElementById('thoughts').innerHTML = th;
    }
    // Working memory
    if(data.working_memory) {
        let wm = (data.working_memory.slots||[]).map(s=>`<div class="stat"><span>${s.source}</span><span class="stat-val">${s.salience.toFixed(2)}</span></div>`).join('');
        document.getElementById('wm').innerHTML = wm || '<div class="signal">empty</div>';
    }
    // Goals
    if(data.goal_stack && data.goal_stack.stack) {
        let g = data.goal_stack.stack.map(s=>`<div class="stat"><span>${s.name}</span><span class="stat-val">t=${s.ticks_active}</span></div>`).join('');
        document.getElementById('goals').innerHTML = g || '<div class="signal">no active goals</div>';
    }
    // Stats
    let st = `<div class="stat"><span>tick</span><span class="stat-val">${data.tick}</span></div>`;
    st += `<div class="stat"><span>confidence</span><span class="stat-val">${(data.meta_cognition?.mean_confidence||0).toFixed(2)}</span></div>`;
    st += `<div class="stat"><span>memories</span><span class="stat-val">${data.episodic_memory_size||0}</span></div>`;
    st += `<div class="stat"><span>curiosity</span><span class="stat-val">${(data.curiosity_beta||0).toFixed(4)}</span></div>`;
    document.getElementById('stats').innerHTML = st;
    // Pulse active regions
    REGIONS.forEach(r=>document.getElementById('r-'+r).classList.remove('active'));
    ['thalamus','cerebrum','amygdala'].forEach(r=>document.getElementById('r-'+r).classList.add('active'));
}

const evtSource = new EventSource('/brain/stream');
evtSource.onmessage = (e) => { try { update(JSON.parse(e.data)); } catch(err){} };
</script></body></html>"""


# ===== ARENA =====

ARENA_HTML = """<!DOCTYPE html>
<html><head><title>Chip Survival Arena</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#0a0a0f; color:#e0e0e0; font-family:'Courier New',monospace; display:flex; }
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
</script></body></html>"""


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
