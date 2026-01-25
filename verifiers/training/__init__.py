"""
Reinforcement Learning training integration.

Provides:
- RLTrainer: Main training loop with trajectory collection
- RewardShaper: Convert rubric scores to RL rewards
- PrimeRLExporter: Export data for prime-rl training
"""

from .rl_trainer import (
    RLTrainer,
    TrainingConfig,
)

from .reward_shaper import (
    RewardShaper,
    RewardConfig,
)

from .prime_rl_adapter import (
    PrimeRLExporter,
    export_for_sft,
    export_for_rl,
)

__all__ = [
    # Trainer
    "RLTrainer",
    "TrainingConfig",

    # Rewards
    "RewardShaper",
    "RewardConfig",

    # Export
    "PrimeRLExporter",
    "export_for_sft",
    "export_for_rl",
]
