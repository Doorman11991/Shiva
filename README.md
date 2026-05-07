
# Shiva AGI: Universal Gated Cognitive Engine

## 1. Overview
**Shiva** is a domain-agnostic Artificial General Intelligence (AGI) framework designed to achieve human-like cognition by bridging heterogeneous data streams—Robotics, Finance, Edge Systems—into a unified **Universal Latent Space ($z \in \mathbb{R}^{512}$)**. 

Beyond functional task-solving, Shiva incorporates **Synthetic Consciousness** and **Emotional Feedback Loops**, allowing the agent to evaluate the "internal state" and "ethical weight" of its actions across physical and digital domains.

## 2. Core Architecture
*   **Dynamic Gated Transformer**: A custom-scratch transformer backbone utilizing **GateHyperNetworks** and **Pre-LayerNorm** stability for high-performance feature extraction.
*   **Dual-Actor Soft-Gate SAC**: A specialized Reinforcement Learning policy utilizing two expert actors (e.g., Stability vs. Objective) mixed dynamically by a soft-gating mechanism.
*   **Prioritized Experience Replay (PER)**: A **Torch-based SumTree** memory buffer that utilizes **Importance Sampling** to ensure unbiased, high-efficiency learning from critical experiences.
*   **Synthetic Affective Layer**: Encodes "emotions" (e.g., urgency, caution, curiosity) as latent priors that modulate the **Soft Gate**, effectively steering the AGI's personality and decision-making style.

## 3. Key Capabilities
*   **Zero-Shot Domain Transfer**: Knowledge learned in robotics (physical constraints) can be instantly projected into software or financial domains via latent alignment.
*   **Weight Incorporation**: Capability to "hot-swap" and adapt pre-trained LLM weights (e.g., Llama-3) to act as a linguistic and logical world-model backbone.
*   **Autonomous Purpose**: Utilizes an **Automatic Reward Constructor** to synthesize its own optimization goals from high-level natural language intent.
*   **Universal Portability**: Engineered for deployment on high-compute servers, autonomous quadwalking bots, and resource-constrained edge devices.

## 4. Technical Specifications
*   **Latent Dimension**: 512-dim Hyper-sphere
*   **Policy**: Entropy-Regularized Soft Actor-Critic (SAC)
*   **Optimizer**: Integrated `torch.optim.Adam` for rapid convergence
*   **Memory**: $O(\log N)$ SumTree with Importance Sampling correction

## 5. Development Roadmap
1.  **Phase 1**: Scratch Transformer & Universal Latent Alignment.
2.  **Phase 2**: Dual-Actor SAC & Gating Policy Implementation.
3.  **Phase 3**: Integration of Affective (Emotional) State Modulators.
4.  **Phase 4**: Deployment and Zero-Shot Benchmarking across heterogenous domains.


```plaintext
shiva/
├── assets/                    # Project diagrams (e.g., image_bf54e6.png)
├── config/                    # Global & Domain configurations
│   ├── domains/               # robotics.yaml, quant.yaml, edge.yaml
│   ├── model/                 # sac_params.yaml, llm_config.yaml
│   └── trainer.yaml           # Global orchestration settings
├── core/                      # The "Shiva" AGI Engine
│   ├── __init__.py
│   ├── latent_alignment.py    # d=512 space + contrastive loss (InfoNCE)
│   ├── shiva_policy.py        # Universal SAC Policy π(a|z)
│   ├── weight_manager.py      # LLM weight injection & LoRA steering
│   ├── reward_constructor.py  # LLM-based automatic reward generation
│   ├── online_trainer.py      # SAC training loop + replay buffer
│   └── evaluator.py           # Self-correction & Directional analysis
├── monitoring/                # Autonomous Self-Supervision
│   ├── divergence_tracker.py  # Monitors policy drift from goals
│   ├── reward_critic.py       # Validates LLM rewards against physics
│   └── intrinsic_curiosity.py # Exploration driver for sparse domains
├── weights/                   # Model & Adapter Storage
│   ├── pretrained_llms/       # Frozen backbones (Llama, Mistral)
│   ├── checkpoints/           # Saved Shiva Core iterations
│   └── adapters/              # Domain-specific LoRA weights
├── domains/                   # Environment Abstractions
│   ├── __init__.py
│   ├── base_domain.py         # Abstract class for all environments
│   ├── robotics_domain.py     # Quadwalking/Drone simulations
│   ├── quant_domain.py        # Financial signal processing
│   └── edge_domain.py         # Hardware register/memory simulators
├── encoders/                  # Domain-to-Latent Translation
│   ├── __init__.py
│   ├── robotics.py            # Physics/IMU encoders
│   ├── language.py            # Tokenizer + LLM Embedding projection
│   ├── timeseries.py          # Quant signal processing
│   └── hardware.py            # Low-level system encoders
├── experiments/               # Research Validation
│   ├── zero_shot_transfer.py  # Cross-domain evaluation
│   ├── weight_injection.py    # LLM-backbone performance tests
│   └── latent_viz.py          # t-SNE & Mutual Info alignment plots
├── utils/                     # Engineering Helpers
│   ├── logger.py              # W&B / TensorBoard integration
│   ├── metrics.py             # MI, Silhouette, and Sample Efficiency math
│   └── registry.py            # Dynamic loading for modularity
├── Dockerfile                 # Reproducible environment
├── requirements.txt           # Torch, Transformers, PEFT, etc.
└── README.md                  # Project Documentation
```
