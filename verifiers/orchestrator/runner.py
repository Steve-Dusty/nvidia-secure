"""
Agent execution engine for running benchmarks.

Orchestrates agent execution across multiple environments,
collecting results and managing resources.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import asyncio
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from verifiers.base import Environment


@dataclass
class AgentConfig:
    """Configuration for an agent to evaluate."""
    name: str
    agent_class: Type
    init_kwargs: Dict[str, Any] = field(default_factory=dict)
    resource_limits: Dict[str, Any] = field(default_factory=dict)

    def create_agent(self) -> Any:
        """Instantiate the agent."""
        return self.agent_class(**self.init_kwargs)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    agent_name: str
    environment_name: str
    overall_score: float
    rubric_scores: Dict[str, float]
    latency_stats: Dict[str, float]
    passed: bool
    timestamp: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_name": self.agent_name,
            "environment_name": self.environment_name,
            "overall_score": self.overall_score,
            "rubric_scores": self.rubric_scores,
            "latency_stats": self.latency_stats,
            "passed": self.passed,
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": self.duration_seconds,
            "details": self.details,
        }


class AgentRunner:
    """
    Orchestrates agent execution across environments.

    Features:
    - Single and parallel benchmark execution
    - Resource management
    - Result aggregation
    - Export capabilities
    """

    def __init__(self, max_workers: int = 4):
        """
        Initialize runner.

        Args:
            max_workers: Maximum concurrent evaluations
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.results: List[BenchmarkResult] = []

    async def run_benchmark(
        self,
        agent_config: AgentConfig,
        environment: Environment
    ) -> BenchmarkResult:
        """
        Run a single agent through an environment.

        Args:
            agent_config: Configuration for agent to run
            environment: Environment to evaluate in

        Returns:
            BenchmarkResult with scores and metrics
        """
        start_time = datetime.now()

        # Instantiate agent
        agent = agent_config.create_agent()

        # Run environment
        env_results = await environment.run(agent)

        duration = (datetime.now() - start_time).total_seconds()

        # Build result
        result = BenchmarkResult(
            agent_name=agent_config.name,
            environment_name=environment.config.name if hasattr(environment, 'config') else "unknown",
            overall_score=env_results.get("overall_score", 0.0),
            rubric_scores=env_results.get("rubric_scores", {}),
            latency_stats=env_results.get("latency_stats", {}),
            passed=env_results.get("passed", False),
            duration_seconds=duration,
            details=env_results,
        )

        self.results.append(result)
        return result

    async def run_suite(
        self,
        agents: List[AgentConfig],
        environments: List[Environment]
    ) -> Dict[str, List[BenchmarkResult]]:
        """
        Run multiple agents across multiple environments.

        Args:
            agents: List of agent configurations
            environments: List of environments

        Returns:
            Dictionary mapping agent names to their results
        """
        all_results: Dict[str, List[BenchmarkResult]] = {}

        for agent_config in agents:
            agent_results = []
            for env in environments:
                result = await self.run_benchmark(agent_config, env)
                agent_results.append(result)
            all_results[agent_config.name] = agent_results

        return all_results

    async def run_parallel(
        self,
        agent_config: AgentConfig,
        environments: List[Environment]
    ) -> List[BenchmarkResult]:
        """
        Run an agent across environments in parallel.

        Args:
            agent_config: Agent to evaluate
            environments: Environments to run in parallel

        Returns:
            List of results from all environments
        """
        tasks = [
            self.run_benchmark(agent_config, env)
            for env in environments
        ]

        return await asyncio.gather(*tasks)

    async def compare_agents(
        self,
        agents: List[AgentConfig],
        environment: Environment
    ) -> Dict[str, Any]:
        """
        Compare multiple agents on the same environment.

        Args:
            agents: Agents to compare
            environment: Environment for comparison

        Returns:
            Comparison results with rankings
        """
        results = []
        for agent_config in agents:
            result = await self.run_benchmark(agent_config, environment)
            results.append(result)

        # Sort by score
        sorted_results = sorted(results, key=lambda r: r.overall_score, reverse=True)

        return {
            "environment": environment.config.name if hasattr(environment, 'config') else "unknown",
            "rankings": [
                {
                    "rank": i + 1,
                    "agent": r.agent_name,
                    "score": r.overall_score,
                    "passed": r.passed,
                }
                for i, r in enumerate(sorted_results)
            ],
            "best_agent": sorted_results[0].agent_name if sorted_results else None,
            "detailed_results": [r.to_dict() for r in sorted_results],
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmark runs."""
        if not self.results:
            return {"status": "no_results"}

        # Group by agent
        agent_results: Dict[str, List[BenchmarkResult]] = {}
        for result in self.results:
            if result.agent_name not in agent_results:
                agent_results[result.agent_name] = []
            agent_results[result.agent_name].append(result)

        # Calculate per-agent statistics
        agent_stats = {}
        for agent, results in agent_results.items():
            scores = [r.overall_score for r in results]
            agent_stats[agent] = {
                "num_benchmarks": len(results),
                "mean_score": sum(scores) / len(scores),
                "min_score": min(scores),
                "max_score": max(scores),
                "pass_rate": sum(1 for r in results if r.passed) / len(results),
            }

        return {
            "total_benchmarks": len(self.results),
            "agents_evaluated": len(agent_results),
            "agent_statistics": agent_stats,
        }

    def export_results(self, filepath: str, format: str = "json"):
        """
        Export benchmark results to file.

        Args:
            filepath: Output path
            format: "json" or "jsonl"
        """
        if format == "jsonl":
            with open(filepath, "w") as f:
                for result in self.results:
                    f.write(json.dumps(result.to_dict()) + "\n")
        else:
            data = {
                "results": [r.to_dict() for r in self.results],
                "summary": self.get_summary(),
                "exported_at": datetime.now().isoformat(),
            }
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

    def clear_results(self):
        """Clear stored results."""
        self.results = []


async def run_benchmark(
    agent_class: Type,
    environment: Environment,
    agent_kwargs: Optional[Dict] = None
) -> BenchmarkResult:
    """
    Convenience function for quick benchmarking.

    Args:
        agent_class: Agent class to instantiate
        environment: Environment to run
        agent_kwargs: Optional agent init kwargs

    Returns:
        Benchmark result
    """
    config = AgentConfig(
        name=agent_class.__name__,
        agent_class=agent_class,
        init_kwargs=agent_kwargs or {}
    )

    runner = AgentRunner()
    return await runner.run_benchmark(config, environment)
