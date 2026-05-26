# Chip: A Brain-Anatomical Proto-AGI

[![PyPI](https://img.shields.io/pypi/v/chip-brain?label=chip-brain&color=blue)](https://pypi.org/project/chip-brain)

Chip runs as a pure-Python cognitive engine that follows the layout of a real brain. Modules sit inside their matching regions, pass typed signals across a shared bus, and can drop out without taking the rest of the system down.

The engine does not output text or hit external APIs during its core loop. It builds internal states in one 512-dimensional space so observations, valence, stored episodes, active goals, and chosen actions can be compared directly with cosine similarity.

## Installation

Install the package the usual way:

```bash
pip install chip-brain
```

Run the pre-built image if you prefer containers:

```bash
docker run -it ghcr.io/doorman11991/chip:latest
```

Clone the repo when you want to modify things:

```bash
git clone https://github.com/Doorman11991/Chip.git
cd Chip
pip install torch transformers
cp .env.example .env
```

The Granite-125m embedding model downloads on first use and stays cached. The Docker build already includes it.

You need Python 3.10 through 3.13, PyTorch 2.0 or newer, and Transformers 4.30+. GPU paths work for NVIDIA via CUDA, AMD on Windows through DirectML, and Apple Silicon through MPS.

## Quick Start

Launch the interactive console:

```bash
python run.py
```

A short session looks like this:

```
you > I notice something strange in the corner of the room.

  [tick 1] mood=Calm, confidence=0.49, |action|=0.42
  thought: "This feels new. Also kind of risky."

you > thoughts

  [Calm] This feels new. Also kind of risky.

you > status

  tick:       1
  mood:       Calm
  top goal:   explore_frontier
  confidence: 0.49
  memories:   0
  wm slots:   3/7
```

Use it from Python code the same way:

```python
from brain import ChipBrain

brain = ChipBrain().boot()
action = brain.tick("I see an unfamiliar door at the end of the corridor.")
brain.train_step(reward=0.5, done=False)
brain.shutdown()
```

## Project Layout

```
Chip/
├── brain.py
├── interfaces/
├── thalamus/
├── amygdala/
├── hippocampus/
├── hypothalamus/
├── cerebrum/
├── cerebellum/
├── brainstem/
├── locomotion/
└── parasite/
```

Each folder holds the logic that belongs to that brain area.

## One Tick, End to End

Text arrives at the thalamus and gets turned into a 512-D vector by Granite-125m. A transformer backbone plus attention bottleneck keeps only the strongest signals while top-down queries from the cerebrum steer focus.

The amygdala scores valence quickly, dampens repeats through habituation, and can block actions that look dangerous.

The hippocampus pulls the three most relevant past episodes into working memory, watches for sudden prediction errors that mark event boundaries, and keeps a running map of explored regions in latent space.

The hypothalamus tracks six drives and picks which one matters most right now. Curiosity reward comes straight from how surprised the world model is.

Inside the cerebrum, seven working-memory slots hold the current context. A dual-actor SAC policy picks the next move while a light reasoning chain fires only when confidence drops. Inner speech gets generated in plain English, re-encoded, and used to keep identity stable. Goals stack hierarchically so high-level aims break into concrete sub-steps.

The cerebellum smooths the chosen action with exponential moving average and pulls matching skills from a small library.

The brainstem runs the SAC update, clips gradients, watches for NaNs, and writes a signed snapshot to disk every N ticks.

## Design Choices That Actually Matter

Everything shares one latent sphere. No extra projection layers sit between modalities.

Regions never import one another. They only publish `NeuralSignal` objects onto a priority bus. That keeps the code testable and lets you disable pieces without side effects.

World-model training uses detached latents so it cannot fight the policy optimizer for the same representation.

Every hundred ticks the brain writes a short English description of its current state, encodes it again, and anchors the identity token. This stops slow drift in the latent space.

When fresh evidence contradicts a stored belief, the embedding rotates on the sphere with spherical linear interpolation instead of snapping. Small contradictions stay quiet. Large ones trigger deliberate review.

The hippocampus does more than replay. It finds key decision points, asks the world model for alternative trajectories, scores them, and keeps the better ones as new synthetic memories.

## Adding Your Own Pieces

Drop in tools, environments, sensors, or reward sources through the plugin interfaces.

```python
from interfaces.plugins import ITool, ToolRegistry

class MyTool(ITool):
    name = "search"
    def call(self, args): ...

brain = ChipBrain(tool_registry=ToolRegistry()).boot()
```

## Saving State

Snapshots land in `.chip_state/` with HMAC-SHA256 signatures, atomic writes, and rolling backups. Boot restores the last clean state automatically. Call `brain.save()` or `brain.shutdown()` whenever you want manual control.

## Tests

One hundred ninety-one tests cover the full loop, memory retrieval, contradiction handling, persistence after crashes, inner speech, and active dreaming.

Run the main suite with:

```bash
python scripts/e2e_brain_test.py
```

## What Chip Is Not

It is not a chatbot that produces replies on demand.

It is not a thin wrapper around LangChain or any API-calling framework.

It is not a throwaway research script that loses state the moment the process ends.

The architecture already handles persistence, graceful degradation, and clean restarts. Real capability will come once it trains inside richer environments.

## License

MIT. See [LICENSE](LICENSE).
