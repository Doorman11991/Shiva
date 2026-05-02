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
