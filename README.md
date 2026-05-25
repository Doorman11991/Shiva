# Chip — A Brain-Anatomical Proto-AGI

Chip is a pure-Python cognitive engine modelled after the human brain. Every module lives in the brain region it belongs to, communicates through typed signals on a central bus, and can be disabled without crashing the system.

It doesn't generate text. It doesn't call APIs. It *thinks* — in a 512-dimensional latent space where observations, emotions, memories, goals, and actions all share the same geometry.

## Installation

```bash
# Clone the repo
git clone https://github.com/Doorman11991/Chip.git
cd Chip

# Install dependencies (just two)
pip install torch transformers

# (Optional) Copy the env template
cp .env.example .env

# Verify everything works
python -c "from brain import ChipBrain; print(ChipBrain())"
```

**Requirements:**
- Python 3.10–3.13 (3.14 has known numpy issues on Windows)
- PyTorch 2.0+
- HuggingFace Transformers 4.30+

**Optional (for GPU acceleration):**
- CUDA toolkit (NVIDIA)
- DirectML (AMD on Windows)
- MPS (Apple Silicon — automatic)

The first call to the text encoder downloads IBM Granite-125m (~250MB) from HuggingFace. After that it's cached locally and loads in ~5 seconds.

## Quick Start

```python
from brain import ChipBrain

brain = ChipBrain().boot()

# One cognitive tick from a text observation:
action = brain.tick("I see an unfamiliar door at the end of the corridor.")

# Feed reward, advance training:
brain.train_step(reward=0.5, done=False)

# Save state to disk:
brain.shutdown()
```

## Requirements

```
pip install torch transformers
```

That's it. Two dependencies. Everything else is built from scratch.

## Architecture

```
Chip/
├── brain.py                 ← Consciousness loop (top-level orchestrator)
├── interfaces/              ← White matter: ABCs, SignalBus, plugin slots
├── thalamus/                ← Sensory relay: granite encoder, transformer backbone, attention bottleneck
├── amygdala/                ← Emotion: valence, fear veto, habituation, arousal modulation
├── hippocampus/             ← Memory: episodic store, recall, dreams, boundary detection, cognitive maps
├── hypothalamus/            ← Drives: homeostasis, curiosity, energy, entropy temperature
├── cerebrum/                ← Cognition: policy, working memory, world model, reasoning, goals, inner speech
├── cerebellum/              ← Coordination: action smoothing, skill library, swarm consensus, emotional contagion
├── brainstem/               ← Life support: training loop, health monitor, persistence, scheduling
├── locomotion/              ← Network migration (cognitive snapshot serialisation)
└── parasite/                ← Parasitic knowledge extraction from black-box models
```

## How It Works

Everything is a 512-D vector on a unit sphere. Text, emotions, memories, actions, goals — all in the same space. Cosine similarity = semantic relatedness.

**One tick flows like this:**

1. **Thalamus** — Text enters via IBM Granite-125m (768→512 projection). Transformer backbone filters and routes tokens. Attention bottleneck selects the top-k most salient. Top-down query from previous tick's cerebrum biases what passes through.

2. **Amygdala** — Fast emotional assessment (valence network). Habituation dampens repeated stimuli. Arousal gain signal sent to thalamus. Fear assessor can veto dangerous actions.

3. **Hippocampus** — Retrieves top-3 relevant past episodes into working memory. Boundary detector auto-segments the stream via prediction-error spikes. Temporal abstractor compresses across timescales. Cognitive map tracks explored latent regions.

4. **Hypothalamus** — 6-dim drive vector (arousal, energy, safety, engagement, curiosity, coherence). Curiosity reward from world model prediction error. Drive arbitrator picks the most urgent need. Entropy temperature adjusts exploration.

5. **Cerebrum** — Working memory (7 slots). Policy selects action via dual-actor SAC with personality bias. Meta-cognition checks confidence; if low, fires 3-step reasoning chain. Inner speech surfaces thoughts in language. Contradiction detector checks new evidence against core beliefs. Goal stack manages hierarchical sub-goal planning.

6. **Cerebellum** — Action smoothing (EMA). Skill library retrieval. Swarm consensus (if multi-node). Emotional contagion across nodes.

7. **Brainstem** — SAC training update. Gradient clipping. Health monitoring (NaN detection). Periodic autosave to disk with HMAC-signed snapshots.

## Key Design Decisions

**One latent space.** No translation layers between modalities. Everything projects into the same 512-D sphere so any two things can be compared by dot product.

**Signal bus, not imports.** Brain regions never import each other. They publish typed `NeuralSignal` objects on a priority-ordered bus. This makes the system testable, observable, and gracefully degradable.

**Stop-gradient boundaries.** The world model trains on detached latents so it can't fight the policy optimizer for the backbone's representation.

**Periodic language grounding.** Every 100 ticks, the brain translates its internal state into English ("I feel calm and curious"), encodes it with granite, and anchors the identity token to it. The subconscious drifts in language-grounded space, not arbitrary latent drift.

**SLERP belief revision.** When new evidence contradicts a core belief, the belief embedding rotates on the unit sphere via spherical linear interpolation. Small contradictions → quiet revision. Large contradictions → narrative crisis that forces deliberation.

**Active dreaming.** The hippocampus doesn't just replay memories — it identifies key decision points, imagines alternative actions via the world model, evaluates the counterfactual trajectories, and stores the best as synthetic memories. Creativity from imagination.

## Plugin Slots

Chip is designed to be embedded in a larger system. The host application provides:

```python
from interfaces.plugins import ITool, ToolRegistry, IEnvironment

class MyTool(ITool):
    name = "search"
    def call(self, args): ...

brain = ChipBrain(tool_registry=ToolRegistry()).boot()
```

Available extension points:
- `ITool` / `ToolRegistry` — external tool dispatch
- `IEnvironment` — step-based environment loop
- `ISensor` — custom sensory modality
- `IRewardSource` — external reward signal
- `HookRegistry` — observe brain events (inner speech, contradictions, boundaries)

## Persistence

The brain autosaves to `.chip_state/` every N ticks:
- HMAC-SHA256 signed snapshots (tamper-proof)
- Atomic writes (crash-safe)
- Rolling backups (corruption recovery)
- Auto-restore on boot

```python
brain.save()      # manual save
brain.shutdown()  # save + cleanup
```

## Tests

```
python scripts/e2e_brain_test.py                    # full brain against LM Studio endpoint
python scripts/example_granite_integration.py       # granite embedder demo
python scripts/test_feature_episodic_recall.py      # inference-time memory retrieval
python scripts/test_feature_topdown_attention.py    # corticothalamic feedback loop
python scripts/test_feature_persistence.py          # save/restore/crash recovery
python scripts/test_feature_inner_speech.py         # internal monologue
python scripts/test_feature_self_consistency.py     # contradiction detection + belief revision
python scripts/test_feature_stability_fixes.py      # stop-gradient + Platt calibration
python scripts/test_feature_goal_stack.py           # hierarchical sub-goal planning
python scripts/test_feature_habit_boundary.py       # habituation + episodic boundaries
python scripts/test_feature_contagion_dreaming.py   # emotional contagion + active dreaming
```

191 tests. All passing.

## What This Is Not

- Not a chatbot. It doesn't generate text.
- Not a LangChain wrapper. No API calls in the loop.
- Not a research toy. It persists state, handles crashes, and scales.
- Not finished. It needs a real training environment to become intelligent.

The architecture is complete. The intelligence emerges from training.

## License

See [LICENSE](LICENSE).
