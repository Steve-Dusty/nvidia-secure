"""
Base classes for the Verifiers framework.

Implements core abstractions following Prime Intellect's Verifiers architecture:
- Dataset: Collection of test inputs
- Rubric: Evaluation functions
- Environment: Test scenarios combining dataset + rubrics
- Trajectory: RL training trajectories
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Optional, Generic, TypeVar,
    Iterator, Union
)
from datetime import datetime
from enum import Enum
import asyncio
import numpy as np


# Type variables for generic typing
T = TypeVar('T')  # Input type
R = TypeVar('R')  # Result type


# =============================================================================
# Core Data Types
# =============================================================================

@dataclass
class TestCase(Generic[T]):
    """
    A single test case for environment evaluation.

    Attributes:
        id: Unique identifier for the test case
        data: Input data (frame, audio, event, etc.)
        ground_truth: Expected output for evaluation
        metadata: Additional context (scenario type, difficulty, etc.)
    """
    id: str
    data: T
    ground_truth: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dataset(Generic[T]):
    """
    Collection of test inputs for an environment.

    Provides iteration and indexing over test cases.
    Compatible with Verifiers' dataset loading patterns.
    """
    name: str
    inputs: List[TestCase[T]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[TestCase[T]]:
        return iter(self.inputs)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> TestCase[T]:
        return self.inputs[idx]

    def filter(self, predicate: Callable[[TestCase[T]], bool]) -> 'Dataset[T]':
        """Filter test cases by predicate."""
        filtered = [tc for tc in self.inputs if predicate(tc)]
        return Dataset(
            name=f"{self.name}_filtered",
            inputs=filtered,
            metadata={**self.metadata, "filtered": True}
        )

    def sample(self, n: int, seed: Optional[int] = None) -> 'Dataset[T]':
        """Random sample of n test cases."""
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self.inputs), size=min(n, len(self.inputs)), replace=False)
        sampled = [self.inputs[i] for i in indices]
        return Dataset(
            name=f"{self.name}_sample_{n}",
            inputs=sampled,
            metadata={**self.metadata, "sampled": n}
        )


@dataclass
class RubricResult:
    """
    Result from a rubric evaluation.

    Attributes:
        score: Normalized score from 0.0 to 1.0
        passed: Whether the evaluation passed threshold
        metrics: Detailed metrics from evaluation
        feedback: Human-readable feedback string
    """
    score: float  # 0.0 to 1.0
    passed: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    feedback: str = ""

    def __post_init__(self):
        # Ensure score is in valid range
        self.score = max(0.0, min(1.0, self.score))


# =============================================================================
# Rubric Base Class
# =============================================================================

class Rubric(ABC):
    """
    Base class for evaluation rubrics.

    Rubrics define how agent outputs are scored against ground truth.
    Implement the `evaluate` method to create custom rubrics.

    Attributes:
        name: Unique identifier for the rubric
        weight: Weight for combining multiple rubrics (default 1.0)
    """

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    @abstractmethod
    async def evaluate(
        self,
        prediction: Any,
        ground_truth: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> RubricResult:
        """
        Evaluate agent output against ground truth.

        Args:
            prediction: Agent's output/prediction
            ground_truth: Expected output from test case
            context: Additional context (turn number, state, etc.)

        Returns:
            RubricResult with score, pass/fail, and detailed metrics
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', weight={self.weight})"


# =============================================================================
# Environment Configuration
# =============================================================================

@dataclass
class EnvironmentConfig:
    """
    Configuration for an environment.

    Attributes:
        name: Environment identifier
        timeout_ms: Maximum time per step in milliseconds
        max_retries: Number of retry attempts on failure
        sandbox_enabled: Whether to run in sandboxed mode
        resource_limits: Memory/CPU limits for agents
    """
    name: str
    timeout_ms: int = 30000
    max_retries: int = 3
    sandbox_enabled: bool = False
    resource_limits: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.resource_limits:
            self.resource_limits = {
                "memory_mb": 1024,
                "cpu_percent": 100,
                "max_concurrent": 4
            }


# =============================================================================
# Environment Base Classes
# =============================================================================

class Environment(ABC, Generic[T, R]):
    """
    Base environment for agent testing and evaluation.

    Environments combine a dataset with rubrics to create
    complete testing scenarios. Implement the `step` method
    to define how agents interact with test cases.

    Attributes:
        dataset: Collection of test cases
        rubrics: List of evaluation rubrics
        config: Environment configuration
        results: Accumulated results from runs
    """

    def __init__(
        self,
        dataset: Dataset[T],
        rubrics: List[Rubric],
        config: Optional[EnvironmentConfig] = None
    ):
        self.dataset = dataset
        self.rubrics = rubrics
        self.config = config or EnvironmentConfig(name="default")
        self.results: List[Dict[str, Any]] = []

        # State tracking
        self._current_test_case: Optional[TestCase[T]] = None
        self._step_count = 0

    @abstractmethod
    async def step(self, agent: Any, input_data: T) -> R:
        """
        Execute one step with the agent.

        Args:
            agent: The agent being evaluated
            input_data: Input from the test case

        Returns:
            Agent's output/prediction
        """
        pass

    def reset(self, test_case: Optional[TestCase[T]] = None) -> Dict[str, Any]:
        """
        Reset environment state for a new episode.

        Args:
            test_case: Optional test case to set as current

        Returns:
            Initial state dictionary
        """
        self._current_test_case = test_case
        self._step_count = 0
        return {"test_case_id": test_case.id if test_case else None}

    async def evaluate(
        self,
        prediction: R,
        ground_truth: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, RubricResult]:
        """
        Run all rubrics on a prediction.

        Args:
            prediction: Agent's output
            ground_truth: Expected output
            context: Additional evaluation context

        Returns:
            Dictionary mapping rubric names to results
        """
        results = {}
        for rubric in self.rubrics:
            try:
                result = await rubric.evaluate(prediction, ground_truth, context)
                results[rubric.name] = result
            except Exception as e:
                results[rubric.name] = RubricResult(
                    score=0.0,
                    passed=False,
                    feedback=f"Rubric error: {str(e)}"
                )
        return results

    async def run(self, agent: Any) -> Dict[str, Any]:
        """
        Run the full environment benchmark.

        Iterates through all test cases, runs the agent,
        and evaluates with all rubrics.

        Args:
            agent: The agent to evaluate

        Returns:
            Aggregated benchmark results
        """
        all_results = []

        for test_case in self.dataset:
            self.reset(test_case)

            start_time = datetime.now()

            try:
                # Run agent with timeout
                prediction = await asyncio.wait_for(
                    self.step(agent, test_case.data),
                    timeout=self.config.timeout_ms / 1000
                )
            except asyncio.TimeoutError:
                prediction = None
            except Exception as e:
                prediction = None
                print(f"Agent error on {test_case.id}: {e}")

            latency_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Evaluate with rubrics
            rubric_results = await self.evaluate(
                prediction,
                test_case.ground_truth,
                {"latency_ms": latency_ms, "test_case_id": test_case.id}
            )

            all_results.append({
                "input_id": test_case.id,
                "prediction": prediction,
                "rubric_results": rubric_results,
                "latency_ms": latency_ms,
                "metadata": test_case.metadata
            })

        self.results = all_results
        return self._aggregate_results(all_results)

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate individual results into summary metrics.

        Args:
            results: List of per-test-case results

        Returns:
            Summary with overall scores and statistics
        """
        if not results:
            return {"overall_score": 0.0, "passed": False}

        # Collect rubric scores
        rubric_scores: Dict[str, List[float]] = {}
        rubric_passed: Dict[str, List[bool]] = {}
        latencies: List[float] = []

        for result in results:
            latencies.append(result["latency_ms"])
            for rubric_name, rubric_result in result["rubric_results"].items():
                if rubric_name not in rubric_scores:
                    rubric_scores[rubric_name] = []
                    rubric_passed[rubric_name] = []
                rubric_scores[rubric_name].append(rubric_result.score)
                rubric_passed[rubric_name].append(rubric_result.passed)

        # Calculate per-rubric averages
        rubric_averages = {
            name: np.mean(scores) for name, scores in rubric_scores.items()
        }
        rubric_pass_rates = {
            name: np.mean(passed) for name, passed in rubric_passed.items()
        }

        # Calculate weighted overall score
        total_weight = sum(r.weight for r in self.rubrics)
        weighted_score = sum(
            rubric_averages.get(r.name, 0) * r.weight
            for r in self.rubrics
        ) / total_weight if total_weight > 0 else 0

        return {
            "overall_score": weighted_score,
            "passed": weighted_score >= 0.7,
            "rubric_scores": rubric_averages,
            "rubric_pass_rates": rubric_pass_rates,
            "latency_stats": {
                "mean_ms": np.mean(latencies),
                "p50_ms": np.percentile(latencies, 50),
                "p95_ms": np.percentile(latencies, 95),
                "max_ms": np.max(latencies)
            },
            "total_test_cases": len(results),
            "individual_results": results
        }


class SingleTurnEnv(Environment[T, R]):
    """
    Environment for single-interaction tasks.

    Each test case is evaluated in a single step with no
    state carried between steps. Suitable for:
    - Single-frame classification
    - One-shot predictions
    - Stateless analysis
    """

    async def run(self, agent: Any) -> Dict[str, Any]:
        """Run single-turn evaluation on all test cases."""
        return await super().run(agent)


class MultiTurnEnv(Environment[T, R]):
    """
    Environment for multi-turn/streaming tasks.

    Supports stateful interactions across multiple steps.
    Suitable for:
    - Video stream analysis
    - Multi-frame tracking
    - Sequential decision making

    Attributes:
        max_turns: Maximum number of turns per episode
        state: Persistent state across turns
    """

    def __init__(
        self,
        *args,
        max_turns: int = 10,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.max_turns = max_turns
        self.state: Dict[str, Any] = {}
        self._current_turn = 0

    def reset(self, test_case: Optional[TestCase[T]] = None) -> Dict[str, Any]:
        """Reset environment state for new episode."""
        base_state = super().reset(test_case)
        self.state = {}
        self._current_turn = 0
        return {**base_state, "state": self.state}

    def update_state(self, key: str, value: Any):
        """Update persistent state."""
        self.state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get value from persistent state."""
        return self.state.get(key, default)


# =============================================================================
# RL Training Types
# =============================================================================

@dataclass
class TrajectoryStep:
    """
    Single step in an RL trajectory.

    Captures the full SARS tuple for training:
    - State: Environment observation
    - Action: Agent's prediction/action
    - Reward: Rubric-computed reward
    - Next State: Resulting state
    - Done: Episode termination flag
    """
    state: Dict[str, Any]
    action: Dict[str, Any]
    reward: float
    next_state: Optional[Dict[str, Any]] = None
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """
    Complete trajectory for RL training.

    Collects all steps from a single episode for use with
    prime-rl or other training frameworks.

    Attributes:
        environment_name: Source environment
        agent_name: Agent that generated trajectory
        steps: List of trajectory steps
        total_reward: Sum of all step rewards
        metadata: Additional trajectory info
    """
    environment_name: str
    agent_name: str
    steps: List[TrajectoryStep]
    total_reward: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.steps)

    def to_training_format(self) -> Dict[str, Any]:
        """
        Convert to prime-rl compatible format.

        Returns format expected by prime-rl trainer:
        - states: List of state observations
        - actions: List of agent actions
        - rewards: List of step rewards
        - dones: List of done flags
        """
        return {
            "env": self.environment_name,
            "agent": self.agent_name,
            "states": [s.state for s in self.steps],
            "actions": [s.action for s in self.steps],
            "rewards": [s.reward for s in self.steps],
            "dones": [s.done for s in self.steps],
            "total_reward": self.total_reward,
            "metadata": self.metadata
        }

    def compute_returns(self, gamma: float = 0.99) -> List[float]:
        """
        Compute discounted returns for each step.

        Args:
            gamma: Discount factor

        Returns:
            List of discounted returns
        """
        returns = []
        G = 0
        for step in reversed(self.steps):
            G = step.reward + gamma * G
            returns.insert(0, G)
        return returns

    def compute_advantages(
        self,
        gamma: float = 0.99,
        values: Optional[List[float]] = None
    ) -> List[float]:
        """
        Compute advantage estimates.

        Args:
            gamma: Discount factor
            values: Optional value estimates (uses rewards if not provided)

        Returns:
            List of advantage estimates
        """
        returns = self.compute_returns(gamma)
        if values is None:
            values = [s.reward for s in self.steps]
        return [r - v for r, v in zip(returns, values)]


# =============================================================================
# Utility Functions
# =============================================================================

def load_example_dataset(name: str) -> Dataset:
    """
    Load an example dataset by name.

    Placeholder for dataset loading - implement based on
    actual data sources.

    Args:
        name: Dataset identifier

    Returns:
        Loaded dataset
    """
    # Placeholder - actual implementation would load from files
    return Dataset(name=name, inputs=[])


async def run_agent_with_retry(
    agent: Any,
    step_fn: Callable,
    input_data: Any,
    max_retries: int = 3
) -> Any:
    """
    Run agent step with retry logic.

    Args:
        agent: Agent to run
        step_fn: Step function to call
        input_data: Input for the step
        max_retries: Maximum retry attempts

    Returns:
        Agent output or None on failure
    """
    for attempt in range(max_retries):
        try:
            return await step_fn(agent, input_data)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {e}")
                return None
            await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff
    return None
