"""
brain.py — The Chip consciousness loop.

This is the top-level assembly and runtime for the entire brain. It:

  1. Instantiates every region with sensible defaults.
  2. Wires them together through the SignalBus.
  3. Runs the cognitive tick loop: thalamus → amygdala → hippocampus →
     hypothalamus → cerebrum → cerebellum → brainstem, each region
     publishing and consuming NeuralSignals.

Biological analogy
~~~~~~~~~~~~~~~~~~
The brain doesn't have a single "main loop" — regions fire asynchronously
at different frequencies. We approximate this with a synchronous tick where
each region runs in the correct anatomical order, but regions can declare
their own tick_every rate to simulate different processing speeds:

    brainstem   → every tick      (heartbeat, always running)
    thalamus    → every tick      (sensory relay, always active)
    amygdala    → every tick      (threat detection is fast)
    hippocampus → every tick      (encoding, but consolidation is slower)
    hypothalamus→ every tick      (drive monitoring)
    cerebrum    → every tick      (thinking)
    cerebellum  → every tick      (action refinement)

    dream cycle → every 50 ticks  (offline replay during low-load)
    mood grounding → every 100 ticks (subconscious semantic anchoring)

Usage
~~~~~
    from brain import ChipBrain

    brain = ChipBrain()
    brain.boot()

    # Single cognitive tick from a text observation:
    action = brain.tick("I see an unfamiliar door at the end of the corridor.")

    # Training update (call after collecting environment feedback):
    brain.train_step(reward=0.5, done=False)

    # Inspect internal state:
    print(brain.status())
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------
from interfaces.signals import SignalBus, NeuralSignal
from interfaces.plugins import ToolRegistry, HookRegistry

# ---------------------------------------------------------------------------
# Brainstem
# ---------------------------------------------------------------------------
from brainstem.device import pick_device, describe_device
from brainstem.running_stats import RunningMeanStd
from brainstem.health_monitor import HealthMonitor
from brainstem.gradient_clipper import GradientClipper
from brainstem.scheduler import WarmupCosineScheduler, TrainingPhaseManager
from brainstem.online_trainer import PrioritizedReplayBuffer, ChipTrainer
from brainstem.cryostasis import Cryostasis
from brainstem.circadian import CircadianCycle
from brainstem.forgetting_prevention import EWC

# ---------------------------------------------------------------------------
# Thalamus
# ---------------------------------------------------------------------------
from thalamus.transformer_backbone import TransformerEncoderBlock
from thalamus.latent_alignment import LatentAligner
from thalamus.sensory_encoder import SensoryEncoder
from thalamus.attention_bottleneck import AttentionBottleneck
from thalamus.merge_strategies import RapidFrankenmergeStrategy

# ---------------------------------------------------------------------------
# Amygdala
# ---------------------------------------------------------------------------
from amygdala.emotional_core import EmotionalCore
from amygdala.fear_assessment import FearAssessor
from amygdala.arousal_modulator import ArousalModulator
from amygdala.emotional_memory import EmotionalMemoryTagger
from amygdala.habituation import HabituationFilter
from amygdala.affective_forecast import AffectiveForecaster, AffectiveForecasterTrainer

# ---------------------------------------------------------------------------
# Hippocampus
# ---------------------------------------------------------------------------
from hippocampus.episodic_memory import EpisodicMemory
from hippocampus.dream_cycle import DreamCycle
from hippocampus.active_dreaming import ActiveDreamer
from hippocampus.temporal_abstraction import TemporalAbstractor
from hippocampus.spatial_map import CognitiveMap
from hippocampus.memory_consolidation import MemoryConsolidator
from hippocampus.episodic_recall import EpisodicRecall
from hippocampus.boundary_detector import BoundaryDetector

# ---------------------------------------------------------------------------
# Hypothalamus
# ---------------------------------------------------------------------------
from hypothalamus.homeostasis import HomeostaticRegulator
from hypothalamus.curiosity_drive import CuriosityDrive
from hypothalamus.energy_manager import EnergyManager
from hypothalamus.drive_arbitrator import Drive, DriveArbitrator
from hypothalamus.entropy_temperature import EntropyTemperatureRegulator

# ---------------------------------------------------------------------------
# Cerebrum
# ---------------------------------------------------------------------------
from cerebrum.chip_policy import (
    ContinuousActor, ContinuousSACPolicy, DoubleQCritic,
)
from cerebrum.working_memory import WorkingMemory
from cerebrum.world_model import LatentDynamicsModel, WorldModelTrainer
from cerebrum.meta_cognition import MetaCognitionMonitor
from cerebrum.reasoning import ReasoningChain, PlanEvaluator
from cerebrum.concept_grounding import ConceptGrounder
from cerebrum.narrative_self import NarrativeSelf
from cerebrum.goal_generator import GoalGenerator
from cerebrum.goal_stack import GoalStack, GoalFrame
from cerebrum.personality import PersonalityTraits
from cerebrum.causal_engine import CausalEngine
from cerebrum.attention_query import AttentionQueryBuilder
from cerebrum.inner_speech import InnerSpeech
from cerebrum.self_consistency import ConsistencyChecker
from cerebrum.planner import TreeSearchPlanner

# ---------------------------------------------------------------------------
# Cerebellum
# ---------------------------------------------------------------------------
from cerebellum.swarm_coordinator import SwarmCoordinator
from cerebellum.action_smoother import ActionSmoother
from cerebellum.skill_library import SkillLibrary
from cerebellum.emotional_contagion import EmotionalContagion


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    # Latent space
    "d_model": 512,
    "action_dim": 4,
    "num_heads": 8,
    "max_seq_len": 128,

    # Memory
    "episodic_capacity": 10_000,
    "sequence_length": 16,
    "narrative_window": 8,

    # Training
    "replay_capacity": 100_000,
    "lr_actor": 3e-4,
    "lr_critic": 3e-4,
    "lr_alpha": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "grad_clip": 1.0,
    "batch_size": 64,

    # Tick rates
    "dream_every": 50,
    "mood_grounding_every": 100,
    "world_model_update_every": 5,
    "consolidation_every": 200,
    "ewc_consolidate_every": 1000,
    "affective_train_every": 20,

    # Circadian
    "circadian_sleep_threshold": 0.2,
    "circadian_wake_threshold": 0.6,
    "circadian_min_wake_ticks": 50,
    "circadian_max_sleep_ticks": 20,

    # Persistence
    "state_dir": ".chip_state",
    "save_every": 500,
    "auto_restore": True,

    # Inner speech
    "inner_speech_every": 25,

    # Rate limiting — minimum seconds between ticks (0 = unlimited).
    # Set to ~0.1 to cap at ~10 ticks/sec and prevent GPU saturation.
    "min_tick_interval": 0.1,
}


# ---------------------------------------------------------------------------
# ChipBrain
# ---------------------------------------------------------------------------

class ChipBrain:
    """
    The complete Chip brain — all regions assembled and wired.

    This is the single entry point for running Chip. It owns all region
    instances, the SignalBus, and the main tick loop.

    Args:
        config:  Override any default configuration values.
        device:  Force a specific device string. Auto-detected if None.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        tool_registry: Optional[ToolRegistry] = None,
        hooks: Optional[HookRegistry] = None,
        cryostasis: Optional[Cryostasis] = None,
    ) -> None:
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.device_str = device or pick_device()
        self.device = torch.device(self.device_str)
        self._tick = 0
        self._booted = False

        # Plugin slots — left empty by default. The host wires them at boot.
        self.tools = tool_registry or ToolRegistry()
        self.hooks = hooks or HookRegistry()

        # Persistence — default to disk-backed Cryostasis under the configured
        # state_dir. Pass cryostasis=Cryostasis(save_every=0) to disable autosave.
        self.cryo = cryostasis or Cryostasis(
            state_dir=self.cfg["state_dir"],
            save_every=self.cfg["save_every"],
        )

        # Will be populated by boot()
        self.bus: Optional[SignalBus] = None
        self._last_action: Optional[torch.Tensor] = None
        self._last_z: Optional[torch.Tensor] = None
        self._last_obs_token: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Boot
    # ------------------------------------------------------------------

    def boot(self) -> "ChipBrain":
        """
        Instantiate and wire all brain regions. Call once before tick().

        Returns self for chaining: brain = ChipBrain().boot()
        """
        if self._booted:
            return self

        D = self.cfg["d_model"]
        A = self.cfg["action_dim"]

        print(f"[ChipBrain] booting on {describe_device(self.device_str)}")

        # First-run GPU setup: if .chip_device doesn't exist, run detection now.
        from pathlib import Path as _Path
        if not _Path(".chip_device").exists():
            try:
                from setup_device import setup as _setup_device
                print("[ChipBrain] First run: detecting GPU...")
                _setup_device()
            except Exception as _e:
                print(f"[ChipBrain] GPU setup skipped: {_e}")

        # ---- Signal bus ------------------------------------------------
        self.bus = SignalBus()
        for region in ("thalamus", "amygdala", "hippocampus", "hypothalamus",
                       "cerebrum", "cerebellum", "brainstem"):
            self.bus.subscribe(region, ["*"])

        # ---- Brainstem -------------------------------------------------
        self.health = HealthMonitor(window=50, divergence_threshold=5.0)
        self.clipper = GradientClipper(max_norm=self.cfg["grad_clip"])
        self.reward_rms = RunningMeanStd()

        # ---- Thalamus --------------------------------------------------
        self.sensory = SensoryEncoder(d_model=D, enable_text=True)
        self.backbone = TransformerEncoderBlock(
            D, self.cfg["num_heads"], self.cfg["max_seq_len"]
        ).to(self.device)
        self.bottleneck = AttentionBottleneck(D, top_k=8)
        self.aligner = LatentAligner(
            encoders=nn.ModuleDict({"text": nn.Linear(D, D)}),
            d_model=D,
        )

        # ---- Amygdala --------------------------------------------------
        self.emotions = EmotionalCore(
            latent_aligner=self.aligner, hidden_dim=D
        ).to(self.device)
        self.fear = FearAssessor(D, A).to(self.device)
        self.arousal_mod = ArousalModulator(D).to(self.device)
        self.emo_tagger = EmotionalMemoryTagger()
        self.habituation = HabituationFilter(latent_dim=D, decay=0.95, floor=0.05, dishabit_threshold=0.2)

        # ---- Hippocampus -----------------------------------------------
        self.memory = EpisodicMemory(
            latent_dim=D,
            capacity=self.cfg["episodic_capacity"],
            sequence_length=self.cfg["sequence_length"],
            narrative_window=self.cfg["narrative_window"],
        ).to(self.device)
        self.dream = DreamCycle(noise_scale=0.05, counterfactual_k=3)
        self.temporal = TemporalAbstractor(latent_dim=D).to(self.device)
        self.cog_map = CognitiveMap(latent_dim=D, max_cells=512).to(self.device)
        self.recall = EpisodicRecall(self.memory, mode="endpoint", top_k=3, min_similarity=0.3)
        self.boundary = BoundaryDetector(sensitivity=2.0, min_episode_len=8, warmup=10)

        # ---- Hypothalamus ----------------------------------------------
        self.homeostasis = HomeostaticRegulator().to(self.device)
        self.curiosity = CuriosityDrive(latent_dim=D).to(self.device)
        self.energy = EnergyManager()
        self.drive_arb = DriveArbitrator()
        self.entropy_reg = EntropyTemperatureRegulator(action_dim=A).to(self.device)

        # ---- Cerebrum --------------------------------------------------
        self.working_mem = WorkingMemory(latent_dim=D, capacity=7)
        self.world_model = LatentDynamicsModel(latent_dim=D, action_dim=A).to(self.device)
        self.wm_trainer = WorldModelTrainer(self.world_model)
        self.meta = MetaCognitionMonitor(d_model=D, action_dim=A).to(self.device)
        self.reasoning = ReasoningChain(latent_dim=D, n_steps=3).to(self.device)
        self.plan_eval = PlanEvaluator(latent_dim=D).to(self.device)
        self.concepts = ConceptGrounder(d_model=D)
        self.narrative = NarrativeSelf(latent_dim=D).to(self.device)
        self.goals = GoalGenerator(latent_dim=D).to(self.device)
        self.goal_stack = GoalStack(max_depth=5, default_max_ticks=100)
        self.personality = PersonalityTraits(latent_dim=D).to(self.device)
        self.causal = CausalEngine(latent_dim=D, action_dim=A).to(self.device)
        self.attn_query = AttentionQueryBuilder(d_model=D).to(self.device)
        self.inner_speech = InnerSpeech(
            period=self.cfg["inner_speech_every"],
            history_len=64,
        )
        self.consistency = ConsistencyChecker(
            contradiction_threshold=0.5,   # granite text lives in a positive cone; 
                                           # sim < 0.5 = "semantically dissimilar enough to conflict"
            deep_scan_every=10,
            revision_threshold=0.6,
        )

        actor1 = ContinuousActor(D, A).to(self.device)
        actor2 = ContinuousActor(D, A).to(self.device)
        critic = DoubleQCritic(D, A).to(self.device)
        self.policy = ContinuousSACPolicy(
            backbone=self.backbone,
            actor1=actor1,
            actor2=actor2,
            memory=self.memory,
            critic=critic,
            d_model=D,
        ).to(self.device)

        # ---- Cerebellum ------------------------------------------------
        self.swarm = SwarmCoordinator(latent_dim=D, n_nodes=3)
        self.smoother = ActionSmoother(action_dim=A, method="ema", alpha=0.7)
        self.skills = SkillLibrary(latent_dim=D)
        self.emo_contagion = EmotionalContagion()

        # ---- Brainstem trainer -----------------------------------------
        self.buffer = PrioritizedReplayBuffer(capacity=self.cfg["replay_capacity"])
        self.trainer = ChipTrainer(
            policy=self.policy,
            buffer=self.buffer,
            emotional_core=self.emotions,
            merge_strategy=RapidFrankenmergeStrategy(),
            gamma=self.cfg["gamma"],
            tau=self.cfg["tau"],
            action_dim=A,
            lr_actor=self.cfg["lr_actor"],
            lr_critic=self.cfg["lr_critic"],
            lr_alpha=self.cfg["lr_alpha"],
            grad_clip=self.cfg["grad_clip"],
            device=self.device_str,
        )

        # ---- Brainstem: EWC + Circadian + LR scheduler -----------------
        self.ewc = EWC(
            model=self.policy,
            consolidation_strength=100.0,
            fisher_samples=200,
        )
        self.circadian = CircadianCycle(
            sleep_threshold=self.cfg["circadian_sleep_threshold"],
            wake_threshold=self.cfg["circadian_wake_threshold"],
            min_wake_ticks=self.cfg["circadian_min_wake_ticks"],
            max_sleep_ticks=self.cfg["circadian_max_sleep_ticks"],
        )
        self.lr_scheduler = WarmupCosineScheduler(
            optimizer=self.trainer.actor_optimizer,
            warmup_steps=500,
            total_steps=50_000,
            lr_max=self.cfg["lr_actor"],
            lr_min=1e-5,
        )

        # ---- Active dreamer (replaces the no-op DreamCycle) ------------
        self.active_dreamer = ActiveDreamer(
            world_model=self.world_model,
            plan_evaluator=self.plan_eval,
            action_dim=A,
            horizon=5,
            n_alternatives=4,
        )

        # ---- Tree-search planner ---------------------------------------
        self.planner = TreeSearchPlanner(
            world_model=self.world_model,
            plan_evaluator=self.plan_eval,
            action_dim=A,
            n_candidates=8,
            horizon=5,
        )

        # ---- Affective forecaster + trainer ----------------------------
        self.affect_forecaster = AffectiveForecaster(latent_dim=D).to(self.device)
        self.affect_trainer = AffectiveForecasterTrainer(self.affect_forecaster)

        self._booted = True

        # Auto-restore from disk if a snapshot is present.
        if self.cfg.get("auto_restore", True):
            restored = self.cryo.restore_if_available(
                policy=self.policy,
                episodic_memory=self.memory,
                emotional_core=self.emotions,
                device=self.device_str,
            )
            if restored is not None:
                print(f"[ChipBrain] restored from {restored['path']} "
                      f"({restored['bytes']:,} bytes, "
                      f"schema={restored['schema_version']})")
                self.hooks.fire("restore", restored)

        # Warm up the async granite encoder so the first tick doesn't block.
        from thalamus.granite_embedder import get_embedder
        get_embedder().warmup("chip brain online")

        print(f"[ChipBrain] all regions online. tick() ready.")
        return self

    # ------------------------------------------------------------------
    # Main cognitive tick
    # ------------------------------------------------------------------

    def tick(
        self,
        observation: Any,
        task_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        One full cognitive cycle: sense → feel → remember → think → act.

        Args:
            observation: A string (text observation) or (B, T, D) tensor.
            task_id:     Optional task index for personality conditioning.

        Returns:
            (1, action_dim) smoothed action tensor.
        """
        assert self._booted, "Call brain.boot() before brain.tick()"

        # Rate-limit: don't fire more than one tick per MIN_TICK_INTERVAL seconds.
        # Prevents rapid HTTP requests from pegging the GPU at 100%.
        now = time.time()
        elapsed = now - getattr(self, '_last_tick_time', 0.0)
        if elapsed < self.cfg.get("min_tick_interval", 0.0):
            # Return the last action without running a full forward pass.
            if self._last_action is not None:
                return self._last_action
        self._last_tick_time = time.time()

        self._tick += 1
        self.memory.set_current_step(self._tick)
        self.health.tick()

        # ----------------------------------------------------------------
        # 1. THALAMUS — sensory encoding
        # ----------------------------------------------------------------
        if isinstance(observation, str):
            obs_token = self.sensory.encode(observation, modality="text")
        elif isinstance(observation, list) and isinstance(observation[0], str):
            obs_token = self.sensory.encode(observation, modality="text")
        else:
            obs_token = observation  # already a (B, T, D) tensor

        obs_token = obs_token.to(self.device)
        self.energy.spend("forward_pass")
        with torch.no_grad():
            z_encoded = self.backbone.forward_pass(obs_token)       # (B, T, D)

        # Top-down attention query from previous tick's cerebrum state.
        # On tick 1 this is None and the bottleneck operates purely
        # bottom-up. From tick 2 onward, the cerebrum biases what the
        # thalamus passes through — corticothalamic feedback.
        td_query = self.attn_query.get_for_thalamus(batch_size=z_encoded.shape[0])
        if td_query is not None:
            td_query = td_query.to(z_encoded.device)
            self.bus.publish(NeuralSignal(
                "cerebrum", "thalamus", "attention_query",
                td_query, priority=0.7,
            ))

        with torch.no_grad():
            z_filtered, salience = self.bottleneck(z_encoded, top_down_query=td_query)
        z_pooled = z_filtered.mean(dim=1)                           # (B, D)

        self.bus.publish(NeuralSignal(
            "thalamus", "*", "sensory_tokens", z_pooled, priority=0.9
        ))

        # ----------------------------------------------------------------
        # 2. AMYGDALA — fast emotional assessment (bypasses cerebrum)
        # ----------------------------------------------------------------
        valence = self.emotions.get_valence(z_pooled)               # (B, 1)
        arousal_val = float(self.emotions._homeostasis.vector[0].item())
        # Habituation: dampen arousal for repeated/familiar observations.
        arousal_val = self.habituation.modulate_arousal(arousal_val, z_pooled.squeeze(0))
        arousal_gain = self.arousal_mod(torch.tensor([[arousal_val]]).to(self.device))
        mood_name, _ = self.emotions.current_mood()

        self.bus.publish(NeuralSignal(
            "amygdala", "thalamus", "arousal_gain", arousal_gain, priority=0.8
        ))
        self.bus.publish(NeuralSignal(
            "amygdala", "cerebrum", "valence_update", valence, priority=0.6
        ))

        # ----------------------------------------------------------------
        # 3. HIPPOCAMPUS — identity context + temporal abstraction
        # ----------------------------------------------------------------
        identity = self.memory.get_identity_context(z_pooled)       # (B, D)
        self.temporal.push(z_pooled.squeeze(0).detach().to('cpu'))
        cell_idx, novelty = self.cog_map.update(z_pooled.squeeze(0).detach())

        self.bus.publish(NeuralSignal(
            "hippocampus", "cerebrum", "memory_retrieve", identity, priority=0.5
        ))

        # Inference-time episodic recall: retrieve top-K relevant past
        # episodes for the current observation and inject them into
        # working memory as recall slots.
        #
        # Important: the recall query must live in the same latent space
        # as the stored episodes. memory.store_text() uses the granite
        # embedder directly (no modality offset), so we do the same here
        # rather than reusing obs_token (which has the modality_embed
        # added to it).
        recall_query: Optional[torch.Tensor] = None
        if isinstance(observation, str):
            from thalamus.granite_embedder import get_embedder
            # encode() returns the cached result from the async pipeline —
            # same text as the thalamus encode, so this is a free cache hit.
            recall_query = get_embedder().encode(observation).detach()
        elif isinstance(observation, list) and observation and isinstance(observation[0], str):
            from thalamus.granite_embedder import get_embedder
            recall_query = get_embedder().encode(observation).mean(dim=0).detach()
        else:
            recall_query = z_pooled.squeeze(0).detach()

        n_recalled = self.recall.inject_into_working_memory(
            query=recall_query,
            working_memory=self.working_mem,
            salience_scale=0.7,
        )
        if n_recalled > 0:
            self.bus.publish(NeuralSignal(
                "hippocampus", "cerebrum", "episodic_recall",
                {"n_recalled": n_recalled}, priority=0.55,
            ))
            self.hooks.fire("episodic_recall", {"n_recalled": n_recalled, "tick": self._tick})

        # ----------------------------------------------------------------
        # 4. HYPOTHALAMUS — drive signals
        # ----------------------------------------------------------------
        # Curiosity from world model prediction error (if we have a prior state)
        curiosity_reward = torch.zeros(1, 1, device=self.device)
        if self._last_z is not None and self._last_action is not None:
            with torch.no_grad():
                z_pred = self.world_model(self._last_z, self._last_action)
            curiosity_reward = self.curiosity.compute_reward(z_pred, z_pooled)
            self.curiosity.step()

        # Episodic boundary detection: feed prediction error to the boundary
        # detector. If it spikes above the recent baseline, auto-segment
        # the current episode and store it in memory.
        pred_err_scalar = float(curiosity_reward.mean().item())
        boundary_fired = self.boundary.tick(
            prediction_error=pred_err_scalar,
            z_current=z_pooled.squeeze(0),
            valence=float(valence.mean().item()),
        )
        if boundary_fired:
            episode_data = self.boundary.flush_episode()
            if episode_data is not None:
                ep_states, ep_valences = episode_data
                self.memory.store_episode(
                    state_sequence=ep_states,
                    valence_sequence=ep_valences,
                    empowerment_score=pred_err_scalar * 0.5,
                )
            self.bus.publish(NeuralSignal(
                "hippocampus", "*", "boundary_detected",
                {"tick": self._tick, "pred_error": pred_err_scalar},
                priority=0.7,
            ))
            self.hooks.fire("boundary_detected", {"tick": self._tick})
            # Partial WM decay on boundary (context partially carries over)
            self.working_mem.decay_step()

        # Emotional memory tagging: compute significance for this observation
        # so the hippocampus can weight replay sampling appropriately.
        emo_significance = self.emo_tagger.compute_significance(
            valence=valence.detach().to('cpu'),
            arousal=torch.tensor([[arousal_val]]),
            surprise=torch.tensor([[pred_err_scalar]]),
        )
        self._last_emo_significance = float(emo_significance.mean().item())

        # Circadian: record reward for plateau detection, advance tick.
        self.circadian.record_reward(pred_err_scalar)
        self.circadian.tick()

        # Energy passively recovers each tick (resting metabolism)
        self.homeostasis.update({
            "arousal": float(novelty) * 0.1,
            "curiosity": float(curiosity_reward.mean().item()) * 0.05,
            "energy": 0.005,  # passive recovery each tick
        })

        # Collect drives and arbitrate
        errors = self.homeostasis.per_dim_error()
        for drive_name, error in errors.items():
            if abs(error) > 0.15:
                self.drive_arb.submit(Drive(
                    name=drive_name,
                    urgency=min(abs(error), 1.0),
                    valence=1.0 if error > 0 else -1.0,
                    source="hypothalamus",
                ))

        winning_drive = self.drive_arb.arbitrate()
        if winning_drive is not None:
            self.bus.publish(NeuralSignal(
                "hypothalamus", "cerebrum", "drive_signal",
                {"name": winning_drive.name, "urgency": winning_drive.urgency},
                priority=winning_drive.urgency,
            ))

        # ----------------------------------------------------------------
        # 5. CEREBRUM — working memory + reasoning + policy
        # ----------------------------------------------------------------
        # Write sensory and identity into working memory
        self.working_mem.write(z_pooled.squeeze(0).detach(), salience=float(salience.detach().mean()), source_tag="thalamus")
        self.working_mem.write(identity.squeeze(0).detach(), salience=0.6, source_tag="hippocampus")
        self.working_mem.decay_step()

        # Personality bias
        pers_bias = self.personality.get_personality_bias().unsqueeze(0)  # (1, D)
        z_conditioned = z_pooled + 0.1 * pers_bias

        # Mood grounding (periodic — every mood_grounding_every ticks)
        self.narrative.ground_mood_in_language(
            homeostasis_vector=self.homeostasis.as_vector(),
            mood_name=mood_name,
            valence=float(valence.mean().item()),
            step=self._tick,
            grounding_interval=self.cfg["mood_grounding_every"],
        )

        # Goal generation from drives
        drive_goals = self.goals.generate_from_drives(
            errors, self.homeostasis.as_vector()
        )
        curiosity_goal = self.goals.generate_curiosity_goal(
            self.cog_map.get_frontier_direction(z_pooled.squeeze(0).detach()),
            float(self.curiosity.beta),
        )
        if curiosity_goal:
            drive_goals.append(curiosity_goal)
        self.goals.update_goals(drive_goals)

        # Hierarchical goal stack: tick the stack (check completion/failure),
        # then push new goals from the generator if the stack is empty.
        stack_event = self.goal_stack.tick(z_pooled.squeeze(0).detach())
        if stack_event == "completed":
            self.hooks.fire("goal_completed", {"goal": self.goal_stack.stack_names()})
        elif stack_event == "failed":
            self.hooks.fire("goal_failed", {"goal": self.goal_stack.stack_names()})
            self._replan_goals(errors)

        # If stack is empty, push the top flat goal as a new stack frame.
        if self.goal_stack.is_empty:
            top_flat = self.goals.top_goal()
            if top_flat is not None:
                self.goal_stack.push(GoalFrame(
                    name=top_flat.name,
                    target_latent=top_flat.target_latent,
                    urgency=top_flat.urgency,
                    source=top_flat.source_drive,
                    max_ticks=top_flat.horizon * 10,
                ))

        # Meta-cognition: should we deliberate?
        task_tensor = torch.tensor([task_id], dtype=torch.long).to(self.device) if task_id is not None else None
        with torch.no_grad():
            raw_action, log_prob, gate = self.policy.get_action(
                z_filtered if z_filtered.shape[1] > 0 else obs_token,
                task_id=task_tensor,
            )
            q1, q2 = self.policy.evaluate_q(
                z_filtered if z_filtered.shape[1] > 0 else obs_token,
                raw_action,
                task_id=task_tensor,
            )
        confidence, deliberating = self.meta.assess(
            z_conditioned, log_prob, q1, q2,
            wm_error=float(curiosity_reward.mean().item()),
        )

        # Inner speech — surface the current state into language and feed
        # the granite-encoded thought back into working memory. Fires on
        # deliberation, on a strong drive winner, or periodically.
        drive_winner_name = winning_drive.name if winning_drive is not None else None
        should_speak, speech_trigger = self.inner_speech.should_speak(
            tick=self._tick,
            deliberating=deliberating,
            drive_winner=drive_winner_name,
        )
        if should_speak:
            from thalamus.granite_embedder import get_embedder
            _emb = get_embedder()
            # Use encode_now so inner speech always encodes the current thought
            # text synchronously, bypassing the 1-tick async lag.
            class _SyncWrapper:
                def encode(self, text, **kw):
                    return _emb.encode_now(text)
            thought = self.inner_speech.speak(
                tick=self._tick,
                z_conscious=z_conditioned.squeeze(0).detach(),
                mood=mood_name,
                homeostasis_errors=errors,
                concept_grounder=self.concepts,
                embedder=_SyncWrapper(),
                working_memory=self.working_mem,
                trigger=speech_trigger,
            )
            if thought is not None:
                self.bus.publish(NeuralSignal(
                    "cerebrum", "*", "inner_speech",
                    thought.to_dict(), priority=0.6,
                ))
                self.hooks.fire("inner_speech", thought.to_dict())

        # Self-consistency check: does the current observation contradict
        # any core belief? Quick check every tick (cheap cosine test).
        # Uses the backbone-processed latent (z_pooled) rather than raw granite,
        # because granite embeddings live in too tight a cone to distinguish
        # semantically opposing content by cosine alone. The backbone learns
        # to spread things apart once trained.
        contradiction = self.consistency.quick_check(
            evidence=z_pooled.squeeze(0).detach(),
            beliefs=self.narrative._core_beliefs,
        )
        if contradiction is not None:
            contradiction.tick = self._tick
            resolution = self.consistency.resolve(contradiction, self.narrative)
            self.bus.publish(NeuralSignal(
                "cerebrum", "*", "contradiction_detected",
                {"belief": contradiction.belief_name,
                 "severity": contradiction.severity,
                 "resolution": resolution},
                priority=0.8 if resolution == "crisis" else 0.5,
            ))
            self.hooks.fire("contradiction", {
                "tick": self._tick,
                "belief": contradiction.belief_name,
                "severity": contradiction.severity,
                "resolution": resolution,
            })
            # Crisis → force deliberation on this tick
            if resolution == "crisis":
                deliberating = True

        # If low confidence, run tree-search planner for best action,
        # then refine with reasoning chain.
        if deliberating:
            stack_goal = self.goal_stack.current_goal()
            z_goal = stack_goal.target_latent.to(self.device).unsqueeze(0) if (
                stack_goal and stack_goal.target_latent is not None
            ) else None
            z_wm = self.working_mem.attend(z_conditioned)

            # Tree-search: sample K candidates, roll out through world model,
            # pick the best trajectory. Overrides the policy's default action.
            best_action, best_value = self.planner.search(
                z_current=z_pooled.squeeze(0).detach(),
                policy_action=raw_action,
            )
            raw_action = best_action.to(self.device)

            # Reasoning chain refines the latent for the re-encode pass.
            z_refined, _ = self.reasoning(z_conditioned, z_goal=z_goal, z_wm=z_wm.unsqueeze(0) if z_wm.dim() == 1 else z_wm)
            # Re-encode with refined latent
            raw_action, log_prob, gate = self.policy.get_action(
                z_refined.unsqueeze(1), task_id=task_tensor
            )

        self.bus.publish(NeuralSignal(
            "cerebrum", "cerebellum", "action_raw", raw_action, priority=0.7
        ))

        # Build top-down attention query for the *next* tick's thalamus pass.
        # Components: top goal latent, current working memory attended context,
        # narrative self-model. The cached query gets consumed by the
        # attention bottleneck on tick t+1.
        stack_goal = self.goal_stack.current_goal()
        goal_latent = (
            stack_goal.target_latent.to(self.device)
            if stack_goal and stack_goal.target_latent is not None
            else None
        )
        wm_ctx = self.working_mem.attend(z_conditioned)
        if wm_ctx.dim() > 1:
            wm_ctx = wm_ctx.squeeze(0)
        self_model = self.narrative.get_self_model(device=str(self.device))
        self.attn_query.build(
            goal_latent=goal_latent,
            wm_context=wm_ctx,
            self_model=self_model,
        )

        # ----------------------------------------------------------------
        # 6. AMYGDALA — fear veto (fast path, post-cerebrum)
        # ----------------------------------------------------------------
        _, vetoed = self.fear.assess(z_conditioned, raw_action)
        if vetoed:
            # Dampen the action toward zero rather than hard-blocking
            raw_action = raw_action * 0.1
            self.bus.publish(NeuralSignal(
                "amygdala", "cerebellum", "fear_veto",
                torch.tensor([1.0]), priority=1.0
            ))
            self.homeostasis.update({"safety": -0.1})

        # ----------------------------------------------------------------
        # 7. CEREBELLUM — action smoothing + skill check
        # ----------------------------------------------------------------
        # Check skill library first
        skill_match = self.skills.retrieve(z_pooled.squeeze(0).detach())
        if skill_match is not None:
            skill_name, skill_seq = skill_match
            # Use first step of the skill sequence as the action
            skill_action = skill_seq[0].unsqueeze(0).to(self.device)
            smoothed = self.smoother.smooth(skill_action)
        else:
            smoothed = self.smoother.smooth(raw_action)

        self.bus.publish(NeuralSignal(
            "cerebellum", "environment", "action_smooth", smoothed, priority=0.9
        ))

        # ----------------------------------------------------------------
        # 8. BRAINSTEM — health check
        # ----------------------------------------------------------------
        self.health.record("valence", float(valence.mean().item()))
        self.health.record("confidence", confidence)
        self.health.record("novelty", float(novelty))

        # ----------------------------------------------------------------
        # Periodic background processes
        # ----------------------------------------------------------------
        energy_fraction = float(self.homeostasis.as_vector()[1].item())

        # Circadian sleep/wake gating
        if not self.circadian.is_sleeping:
            if self.circadian.should_sleep(
                energy_fraction=energy_fraction,
                mean_prediction_error=pred_err_scalar,
            ):
                self.circadian.enter_sleep()
                self.hooks.fire("sleep_enter", {"tick": self._tick})
        else:
            novelty_spike = float(novelty) > 0.6
            if self.circadian.should_wake(energy_fraction=energy_fraction, novelty_spike=novelty_spike):
                self.circadian.wake_up()
                self.hooks.fire("sleep_exit", {"tick": self._tick})

        # During sleep: run active dreaming + consolidation more aggressively.
        # During wake: run on the normal fixed schedule.
        is_sleeping = self.circadian.is_sleeping
        dream_interval = max(10, self.cfg["dream_every"] // 3) if is_sleeping else self.cfg["dream_every"]
        consolidation_interval = max(50, self.cfg["consolidation_every"] // 3) if is_sleeping else self.cfg["consolidation_every"]

        # Active dreaming (replaces the no-op DreamCycle)
        if self._tick % dream_interval == 0 and self.energy.can_afford("dream_cycle"):
            self.energy.spend("dream_cycle")
            dream_result = self.active_dreamer.run(self.memory, batch_size=4)
            if dream_result.get("n_stored", 0) > 0:
                self.bus.publish(NeuralSignal(
                    "hippocampus", "*", "dream_complete",
                    dream_result, priority=0.3,
                ))

        # World model update (skip during sleep to save compute)
        if not is_sleeping and self._tick % self.cfg["world_model_update_every"] == 0:
            if self._last_z is not None and self._last_action is not None and self.energy.can_afford("world_model_rollout"):
                self.energy.spend("world_model_rollout")
                self.wm_trainer.update(
                    self._last_z.detach(),
                    self._last_action.detach(),
                    z_pooled.detach(),
                )

        # Memory consolidation (MemoryConsolidator now passes action=None correctly)
        if self._tick % consolidation_interval == 0 and self.energy.can_afford("memory_consolidation"):
            self.energy.spend("memory_consolidation")
            dream_batch = self.memory.get_dream_batch(16)
            if dream_batch is not None:
                consolidator = MemoryConsolidator(self.world_model)
                consolidator.consolidate(dream_batch.to(self.device))

        # Affective forecaster training (learn to predict future valence)
        if self._tick % self.cfg["affective_train_every"] == 0:
            dream_batch = self.memory.get_dream_batch(8)
            if dream_batch is not None and dream_batch.shape[1] > 1:
                dream_batch_dev = dream_batch.to(self.device)
                # Use valence of each state as the training target
                with torch.no_grad():
                    B_af, T_af, D_af = dream_batch_dev.shape
                    flat = dream_batch_dev.reshape(B_af * T_af, D_af)
                    valence_targets = self.emotions.get_valence(flat).reshape(B_af, T_af, 1)
                self.affect_trainer.update(dream_batch_dev, valence_targets)

        # EWC consolidation: snapshot policy importance periodically
        if self._tick % self.cfg["ewc_consolidate_every"] == 0 and self._tick > 0:
            self.ewc.consolidate()

        # LR scheduler step (every training tick)
        self.lr_scheduler.step()

        # Periodic persistence (autosave)
        if self.cryo.maybe_save(
            self._tick,
            policy=self.policy,
            episodic_memory=self.memory,
            emotional_core=self.emotions,
        ):
            self.hooks.fire("autosave", {
                "tick": self._tick,
                "save_count": self.cryo._save_count,
            })

        # ----------------------------------------------------------------
        # Store state for next tick
        # ----------------------------------------------------------------
        self._last_z = z_pooled.detach()
        self._last_action = smoothed.detach()
        self._last_obs_token = obs_token.detach()

        return smoothed

    # ------------------------------------------------------------------
    # Training update (call after receiving environment reward)
    # ------------------------------------------------------------------

    def train_step(
        self,
        reward: float,
        done: bool,
        next_observation: Optional[Any] = None,
        task_id: Optional[int] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Store the last transition in the replay buffer and run one SAC update.

        Args:
            reward:           Scalar reward from the environment.
            done:             Whether the episode ended.
            next_observation: Next observation (if available). If None,
                              uses the last encoded state.
            task_id:          Optional task index.

        Returns:
            Dict of training metrics, or None if buffer is too small.
        """
        if self._last_z is None or self._last_action is None:
            return None

        # Encode next observation if provided
        if next_observation is not None:
            with torch.no_grad():
                next_token = self.sensory.encode(next_observation, modality="text") if isinstance(next_observation, str) else next_observation
                next_token = next_token.to(self.device)
                z_next = self.backbone.forward_pass(next_token).mean(dim=1)
        else:
            z_next = self._last_z

        # Build transition tuple
        state = self._last_obs_token.squeeze(0) if self._last_obs_token is not None else self._last_z.unsqueeze(1)
        action = self._last_action.squeeze(0)

        # Augment reward with curiosity bonus
        curiosity_bonus = 0.0
        if self._last_z is not None:
            with torch.no_grad():
                z_pred = self.world_model(self._last_z, self._last_action)
                curiosity_bonus = float(
                    self.curiosity.compute_reward(z_pred, z_next).mean().item()
                ) * 0.1

        total_reward = reward + curiosity_bonus

        if task_id is not None:
            transition = (state, action, total_reward, z_next.squeeze(0), float(done), task_id)
        else:
            transition = (state, action, total_reward, z_next.squeeze(0), float(done))

        self.buffer.add(transition)

        # Confidence calibration: record whether the last action succeeded.
        # "Success" = positive reward. The meta-cognition module uses this
        # to fit Platt scaling so confidence scores become well-calibrated.
        last_confidence = (
            self._confidence_history[-1] if hasattr(self, '_confidence_history')
            and self._confidence_history else None
        )
        if last_confidence is None and self.meta._confidence_history:
            last_confidence = self.meta._confidence_history[-1]
        if last_confidence is not None:
            self.meta.record_outcome(
                predicted_confidence=last_confidence,
                actual_success=(reward > 0),
            )

        # Update homeostasis from reward signal
        self.emotions.update_homeostasis(
            action_impact=float(action.abs().mean().item()),
            environment_surprise=float(abs(total_reward)),
            task_success=max(0.0, reward),
        )
        self.homeostasis.update({
            "energy": -0.01,
            "engagement": 0.02 if reward > 0 else -0.01,
        })

        # Store episode in hippocampus if done
        if done and self._last_z is not None:
            valence = self.emotions.get_valence(self._last_z)
            self.memory.store_episode(
                state_sequence=self._last_z,
                valence_sequence=valence.detach(),
                empowerment_score=max(0.0, reward),
            )
            self.narrative.update_narrative(
                self._last_z.squeeze(0).detach().to('cpu'),
                outcome_valence=float(valence.mean().item()),
            )
            self.smoother.reset()
            self.working_mem.reset()

        # SAC update (with EWC penalty added to actor loss via trainer hook)
        if len(self.buffer.tree.data) > self.cfg["batch_size"]:
            # Inject EWC penalty into the trainer before the update step
            ewc_penalty = self.ewc.penalty()
            if ewc_penalty.item() > 0:
                self.trainer._ewc_penalty = ewc_penalty
            metrics = self.trainer.update_step(self.cfg["batch_size"])
            if metrics:
                self.health.record_dict(metrics)
            return metrics

        return None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Return a snapshot of all region states for monitoring."""
        return {
            "tick": self._tick,
            "device": self.device_str,
            "mood": self.emotions.current_mood()[0],
            "homeostasis": self.homeostasis.status(),
            "most_urgent_drive": self.homeostasis.most_urgent_drive(),
            "top_goal": self.goal_stack.current_goal().name if self.goal_stack.current_goal() else None,
            "goal_stack": self.goal_stack.status(),
            "working_memory": self.working_mem.status(),
            "episodic_memory_size": self.memory.size,
            "cognitive_map": self.cog_map.stats(),
            "temporal_levels_ready": self.temporal.ready_levels,
            "curiosity_beta": self.curiosity.beta,
            "energy": self.energy.status(),
            "meta_cognition": self.meta.status(),
            "inner_speech": self.inner_speech.status(),
            "self_consistency": self.consistency.status(),
            "narrative": self.narrative.status(),
            "causal_graph": self.causal.status(),
            "health": self.health.summary(),
            "cryostasis": self.cryo.status(),
            "circadian": self.circadian.status(),
            "ewc": self.ewc.status(),
            "active_dreamer": self.active_dreamer.status(),
            "planner": self.planner.status(),
            "emo_significance": getattr(self, "_last_emo_significance", 0.0),
        }

    # ------------------------------------------------------------------
    # Persistence — manual save / shutdown
    # ------------------------------------------------------------------

    def save(self) -> bool:
        """Force an immediate snapshot save. Returns True on success."""
        if not self._booted:
            return False
        return self.cryo.save(
            tick=self._tick,
            policy=self.policy,
            episodic_memory=self.memory,
            emotional_core=self.emotions,
        )

    def _replan_goals(self, drive_errors: Dict[str, float]) -> None:
        """
        Called when the goal stack fails. Generate fresh goals from current
        drives and push a new frame. This implements replan-on-failure.
        """
        new_goals = self.goals.generate_from_drives(
            drive_errors, self.homeostasis.as_vector()
        )
        if new_goals:
            best = new_goals[0]
            self.goal_stack.push(GoalFrame(
                name=f"replan_{best.name}",
                target_latent=best.target_latent,
                urgency=best.urgency,
                source="replan",
                max_ticks=best.horizon * 10,
            ))
            self._replan_count = getattr(self, '_replan_count', 0) + 1

    def shutdown(self) -> bool:
        """Save final state and emit a shutdown hook. Call before process exit."""
        ok = self.save()
        self.hooks.fire("shutdown", {"tick": self._tick, "saved": ok})
        # Shut down the async granite encoder thread cleanly.
        try:
            from thalamus.granite_embedder import get_embedder
            get_embedder().shutdown()
        except Exception:
            pass
        return ok

    def __repr__(self) -> str:
        state = "booted" if self._booted else "unbooted"
        return f"ChipBrain({state}, tick={self._tick}, device={self.device_str})"


__all__ = ["ChipBrain", "DEFAULT_CONFIG"]
