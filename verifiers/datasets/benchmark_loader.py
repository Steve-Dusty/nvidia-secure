"""
Benchmark dataset loader for loading pre-defined test sets.

Supports loading from:
- TOML configuration files
- JSON/JSONL data files
- Existing video/audio files (if available)
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from verifiers.base import Dataset, TestCase

# Try to import toml (optional dependency)
try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import toml as tomllib  # Fallback to toml package
    except ImportError:
        tomllib = None


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark dataset."""
    name: str
    description: str
    version: str
    type: str  # "visual", "audio", "integrated"
    test_cases: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class BenchmarkLoader:
    """
    Load benchmark datasets from various sources.

    Supports:
    - TOML configuration files
    - JSON/JSONL data files
    - Directory of test files
    """

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize loader.

        Args:
            base_path: Base directory for benchmark files
        """
        self.base_path = Path(base_path) if base_path else Path(__file__).parent.parent.parent / "benchmarks"

    def load_from_toml(self, config_path: Union[str, Path]) -> Dataset:
        """
        Load benchmark from TOML configuration.

        Args:
            config_path: Path to TOML config file

        Returns:
            Dataset configured from TOML
        """
        if tomllib is None:
            raise ImportError("TOML support requires Python 3.11+ or 'toml' package")

        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = self.base_path / config_path

        with open(config_path, "rb") as f:
            if hasattr(tomllib, "load"):
                config = tomllib.load(f)
            else:
                # Fallback for older toml package
                config = tomllib.loads(f.read().decode())

        return self._build_dataset_from_config(config)

    def load_from_json(self, json_path: Union[str, Path]) -> Dataset:
        """
        Load benchmark from JSON file.

        Args:
            json_path: Path to JSON file

        Returns:
            Dataset from JSON
        """
        json_path = Path(json_path)
        if not json_path.is_absolute():
            json_path = self.base_path / json_path

        with open(json_path) as f:
            data = json.load(f)

        return self._build_dataset_from_json(data)

    def load_from_jsonl(self, jsonl_path: Union[str, Path]) -> Dataset:
        """
        Load benchmark from JSONL file (one JSON object per line).

        Args:
            jsonl_path: Path to JSONL file

        Returns:
            Dataset from JSONL
        """
        jsonl_path = Path(jsonl_path)
        if not jsonl_path.is_absolute():
            jsonl_path = self.base_path / jsonl_path

        test_cases = []
        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                if line.strip():
                    data = json.loads(line)
                    tc = self._build_test_case(data, f"tc_{i:04d}")
                    test_cases.append(tc)

        return Dataset(
            name=jsonl_path.stem,
            inputs=test_cases,
            metadata={"source": str(jsonl_path), "format": "jsonl"}
        )

    def load_from_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.json"
    ) -> Dataset:
        """
        Load benchmark from directory of test files.

        Args:
            directory: Directory containing test files
            pattern: Glob pattern for files to load

        Returns:
            Combined dataset from all files
        """
        directory = Path(directory)
        if not directory.is_absolute():
            directory = self.base_path / directory

        test_cases = []
        for file_path in sorted(directory.glob(pattern)):
            with open(file_path) as f:
                data = json.load(f)

            tc = self._build_test_case(data, file_path.stem)
            test_cases.append(tc)

        return Dataset(
            name=directory.name,
            inputs=test_cases,
            metadata={"source": str(directory), "pattern": pattern}
        )

    def _build_dataset_from_config(self, config: Dict) -> Dataset:
        """Build dataset from parsed TOML config."""
        benchmark = config.get("benchmark", {})

        test_cases = []
        for tc_config in config.get("test_cases", []):
            tc = self._build_test_case(tc_config, tc_config.get("id", f"tc_{len(test_cases)}"))
            test_cases.append(tc)

        return Dataset(
            name=benchmark.get("name", "benchmark"),
            inputs=test_cases,
            metadata={
                "version": benchmark.get("version", "1.0.0"),
                "description": benchmark.get("description", ""),
                **config.get("metadata", {})
            }
        )

    def _build_dataset_from_json(self, data: Dict) -> Dataset:
        """Build dataset from parsed JSON."""
        test_cases = []
        for i, tc_data in enumerate(data.get("test_cases", [])):
            tc = self._build_test_case(tc_data, tc_data.get("id", f"tc_{i:04d}"))
            test_cases.append(tc)

        return Dataset(
            name=data.get("name", "benchmark"),
            inputs=test_cases,
            metadata=data.get("metadata", {})
        )

    def _build_test_case(self, data: Dict, default_id: str) -> TestCase:
        """Build a TestCase from dictionary data."""
        tc_id = data.get("id", default_id)

        # Extract ground truth
        ground_truth = data.get("ground_truth", {})
        if not ground_truth:
            # Try to build from individual fields
            for field in ["emergency_type", "severity", "dispatch_recommended",
                          "fall_detected", "fight_detected", "distress_detected",
                          "action", "keywords_detected"]:
                if field in data:
                    ground_truth[field] = data[field]

        # Extract test data
        test_data = data.get("data", data.get("input", {}))

        # Handle frame/audio data references
        if "frame_path" in data:
            # Would load actual image here if needed
            test_data = {"frame_path": data["frame_path"]}
        if "audio_path" in data:
            test_data = {"audio_path": data["audio_path"]}

        return TestCase(
            id=tc_id,
            data=test_data,
            ground_truth=ground_truth,
            metadata=data.get("metadata", {})
        )


def load_benchmark_dataset(
    name: str,
    base_path: Optional[Path] = None
) -> Dataset:
    """
    Convenience function to load a benchmark by name.

    Automatically detects format based on file extension.

    Args:
        name: Benchmark name or path
        base_path: Optional base directory

    Returns:
        Loaded dataset
    """
    loader = BenchmarkLoader(base_path)

    # Try different file extensions
    for ext in [".toml", ".json", ".jsonl"]:
        try:
            path = Path(name)
            if path.suffix == ext:
                if ext == ".toml":
                    return loader.load_from_toml(path)
                elif ext == ".json":
                    return loader.load_from_json(path)
                elif ext == ".jsonl":
                    return loader.load_from_jsonl(path)
        except FileNotFoundError:
            continue

    # Try with extensions
    for ext in [".toml", ".json", ".jsonl"]:
        try:
            path = Path(f"{name}{ext}")
            if ext == ".toml":
                return loader.load_from_toml(path)
            elif ext == ".json":
                return loader.load_from_json(path)
            else:
                return loader.load_from_jsonl(path)
        except FileNotFoundError:
            continue

    # Try as directory
    try:
        return loader.load_from_directory(name)
    except (FileNotFoundError, NotADirectoryError):
        pass

    raise FileNotFoundError(f"Benchmark not found: {name}")


def list_available_benchmarks(base_path: Optional[Path] = None) -> List[str]:
    """
    List all available benchmark datasets.

    Args:
        base_path: Base directory to search

    Returns:
        List of benchmark names
    """
    base = Path(base_path) if base_path else Path(__file__).parent.parent.parent / "benchmarks"

    if not base.exists():
        return []

    benchmarks = []
    for ext in ["*.toml", "*.json", "*.jsonl"]:
        for f in base.glob(ext):
            benchmarks.append(f.stem)

    # Also list subdirectories
    for d in base.iterdir():
        if d.is_dir() and not d.name.startswith("."):
            benchmarks.append(d.name)

    return sorted(set(benchmarks))
