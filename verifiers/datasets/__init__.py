"""
Dataset utilities for environment testing.

Provides:
- Synthetic test case generation
- Benchmark dataset loading
- Dataset manipulation utilities
"""

from .synthetic_generator import (
    generate_integrated_dataset,
    generate_visual_dataset,
    generate_audio_dataset,
    SyntheticFrameGenerator,
    SyntheticAudioGenerator,
)

from .benchmark_loader import (
    load_benchmark_dataset,
    BenchmarkLoader,
)

__all__ = [
    # Generators
    "generate_integrated_dataset",
    "generate_visual_dataset",
    "generate_audio_dataset",
    "SyntheticFrameGenerator",
    "SyntheticAudioGenerator",

    # Loaders
    "load_benchmark_dataset",
    "BenchmarkLoader",
]
