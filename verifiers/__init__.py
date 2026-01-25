"""
Verifiers: Agent Testing, Benchmarking & Training Framework

A standalone implementation following Prime Intellect's Verifiers architecture patterns
for testing, benchmarking, orchestrating, and training emergency detection agents.

Core Components:
- Environment: Defines test scenarios with datasets and rubrics
- Rubric: Evaluation functions that score agent outputs
- Dataset: Collections of test inputs with ground truth
- Trajectory: RL training trajectory for prime-rl integration

Compatible with:
- NVIDIA NIM Visual/Audio inference agents
- Integrated multi-modal emergency detection
- Emergency response routing agents
"""

from .base import (
    # Core Types
    Dataset,
    TestCase,
    RubricResult,

    # Base Classes
    Rubric,
    Environment,
    EnvironmentConfig,
    SingleTurnEnv,
    MultiTurnEnv,

    # RL Training Types
    TrajectoryStep,
    Trajectory,
)

__version__ = "0.1.0"
__all__ = [
    # Core Types
    "Dataset",
    "TestCase",
    "RubricResult",

    # Base Classes
    "Rubric",
    "Environment",
    "EnvironmentConfig",
    "SingleTurnEnv",
    "MultiTurnEnv",

    # RL Training Types
    "TrajectoryStep",
    "Trajectory",
]
