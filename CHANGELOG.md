# Changelog

All notable changes to Chip are documented here.

## [1.0.1] - 2026-05-25

### Demo
- Brain visualizer: 10-chapter narrated slideshow with per-chapter animated slide panels
- Each chapter has sentence-level subtitles synced to TTS audio via `timeupdate`
- Progressive chunked TTS loading: chapter 1 plays immediately, later chapters prefetch in background
- Slide panels reveal content progressively as narration hits each sentence (`data-stage` system)
- Chapters cover: thalamus gating, amygdala dual-route, hippocampal H.M. case, hypothalamic drives, cerebrum WM + SAC, cerebellum EMA + skill library, brainstem autonomic loop, full tick pipeline, structural comparison vs LLMs, problem statement, full recap
- Replaced KittenTTS with edge-tts (Microsoft Edge neural voices, free, no API key, MP3 output)
- TTS requests serialized with lock + per-text cache to prevent parallel hammering
- Brain page layout fixed: stats sidebar (Mood, Drives, WM, Goals, Inner Speech) sits beside brain, not above it
- Shared nav sidebar on all three demo pages

### Repo hygiene
- Added `*.db`, `*.db-shm`, `*.db-wal`, `.omnimemory/`, `.memory/`, `.code-graph/` to `.gitignore`
- Removed tracked `.omnimemory/graph.db*` files from history

### CI/CD
- Fixed GHCR Docker tag: repository path now fully lowercase (`doorman11991/chip`)
- Added `--skip-existing` to twine upload so re-tagging an already-published version does not fail
- Added `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24` to suppress Node.js 20 deprecation warnings

## [1.0.0]  - 2026-05-25

### Architecture
- Restructured entire codebase from flat `core/` to 9 brain-anatomical packages
- Added `brain.py`  - single-entry consciousness loop orchestrating all regions
- Added `interfaces/signals.py`  - typed SignalBus with priority-ordered pub/sub
- Added `interfaces/plugins.py`  - extension slots for tools, environments, sensors, rewards
- Added backward-compat shims in `core/` so old imports still work
- Renamed project from Shiva to Chip
- MIT license

### Thalamus (Sensory Relay)
- IBM Granite-125m-English text encoder with singleton lazy-load (`granite_embedder.py`)
- 768→512 linear projection, double L2 normalization, batch encoding
- Unified multi-modal SensoryEncoder with auto-registered text modality
- Attention bottleneck with salience gating and top-k token selection
- Top-down corticothalamic query bias from cerebrum (closes the feedback loop)

### Amygdala (Emotion)
- Fear assessor with veto threshold and safe-zone EMA tracking
- Arousal modulator  - attention gain control based on emotional state
- Emotional memory tagger  - significance scoring for hippocampal storage
- Habituation filter  - EMA novelty decay, dishabituation on novel input
- Affective forecaster  - GRU predicts future valence of trajectories

### Hippocampus (Memory)
- Episodic memory with `store_text()` and `query_by_text()` via granite
- Inference-time episodic recall  - top-K cosine retrieval into working memory
- Dream cycle  - counterfactual noise injection + reconstruction loss
- Active dreaming  - decision-point perturbation, world model rollouts, synthetic memory storage
- Memory consolidation  - replay episodes into world model weights
- Temporal abstraction  - hierarchical compression (episode/session/epoch)
- Cognitive map  - latent place cells, frontier detection, transition graph
- Boundary detector  - auto-segment streams via prediction-error spikes

### Hypothalamus (Drives)
- 6-dim homeostatic regulator (arousal, energy, safety, engagement, curiosity, coherence)
- Curiosity drive  - prediction-error intrinsic reward with beta decay
- Energy manager  - compute budget, fatigue modeling, wall-clock recovery
- Drive arbitrator  - priority queue with safety override
- Entropy temperature regulator  - SAC alpha with homeostatic correction

### Cerebrum (Cognition)
- Working memory  - 7 slots, salience-gated, exponential decay, soft attention
- World model  - latent dynamics MLP with residual, stop-gradient training
- Meta-cognition  - confidence estimator + Platt calibration from outcomes
- Reasoning chain  - 3-step latent thought refinement conditioned on goals + WM
- Tree-search planner  - MCTS-lite: K candidates × H steps, pick best trajectory
- Goal generator  - goals from homeostatic deficits + curiosity frontier
- Hierarchical goal stack  - LIFO with sub-goals, pop-on-completion, replan-on-failure
- Narrative self  - core beliefs (SLERP-updated), periodic mood grounding via granite
- Inner speech  - concepts → templated sentence → granite encode → working memory
- Self-consistency  - contradiction detection, severity scoring, belief revision or crisis
- Personality traits  - learnable risk/curiosity/persistence + task conditioning
- Concept grounding  - 16-concept vocabulary, bidirectional latent↔symbol binding
- Causal engine  - attribution, counterfactual reasoning, soft causal graph
- Attention query builder  - top-down query from goal + WM + self for thalamus bias
- Theory of mind  - per-peer LSTM predicting state, action, intent; deception detection

### Cerebellum (Coordination)
- Swarm coordinator  - Global Workspace cross-attention consensus + diversity loss
- Action smoother  - EMA/window/learned-conv temporal smoothing
- Skill library  - cosine retrieval, success tracking, LRU eviction
- Emotional contagion  - valence spreads across swarm nodes, extreme dampening

### Brainstem (Life Support)
- Full SAC training loop with PER, double-Q critics, auto-alpha
- Gradient clipper  - global norm clip, spike detection, statistics
- Health monitor  - NaN/Inf detection, divergence trends, throughput tracking
- Scheduler  - warmup-cosine LR + training phase manager
- Cryostasis  - HMAC-signed atomic disk snapshots, rolling backups, auto-restore
- Circadian cycle  - sleep/wake gating by energy + reward plateau + prediction error
- EWC forgetting prevention  - diagonal Fisher, multi-task parameter protection

### Locomotion
- Cognitive snapshot serialization (HMAC-SHA256, schema-versioned)
- HTTP + gRPC transport stubs for cross-node migration

### Parasite
- Forward-hook parasitic extraction from black-box models
- InfoNCE distillation with EMA target encoder

### Developer Experience
- `run.py`  - interactive REPL for observing brain state
- `.env.example`  - optional config template
- 11 e2e test scripts, 191+ tests passing
- `requirements.txt`  - just `torch` + `transformers`
