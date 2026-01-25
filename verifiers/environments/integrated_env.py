"""
Integrated Multi-Modal Emergency Detection Environment.

This is the primary environment for testing agents that combine
visual and audio analysis for emergency detection.

Supports:
- Multi-turn evaluation with state tracking
- Visual (frames) + audio (chunks) input
- Comprehensive rubric evaluation
- Trajectory collection for RL training
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Generator
from datetime import datetime
import asyncio
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from verifiers.base import (
    MultiTurnEnv,
    Dataset,
    TestCase,
    EnvironmentConfig,
    Rubric,
    TrajectoryStep,
)
from verifiers.rubrics.visual_rubric import (
    FallDetectionRubric,
    FightDetectionRubric,
    SeverityClassificationRubric,
)
from verifiers.rubrics.audio_rubric import (
    DistressDetectionRubric,
    KeywordDetectionRubric,
)
from verifiers.rubrics.composite_rubric import (
    EmergencyClassificationRubric,
    MultiModalFusionRubric,
    DispatchDecisionRubric,
    AlertTimingRubric,
)
from verifiers.rubrics.latency_rubric import LatencyRubric


@dataclass
class MultiModalTestCase:
    """
    Test case with both visual and audio data for integrated testing.

    Attributes:
        id: Unique identifier
        frames: Sequence of video frames (numpy arrays)
        audio_chunks: Corresponding audio segments
        ground_truth: Expected outputs per turn
        scenario_type: Type of emergency scenario
        metadata: Additional test case info
    """
    id: str
    frames: List[np.ndarray]
    audio_chunks: List[Optional[np.ndarray]]
    ground_truth: Dict[str, Any]
    scenario_type: str  # "fall", "fight", "distress", "medical", "normal"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_turns(self) -> int:
        return len(self.frames)

    def get_turn_data(self, turn: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Get frame and audio for a specific turn."""
        frame = self.frames[turn] if turn < len(self.frames) else self.frames[-1]
        audio = self.audio_chunks[turn] if turn < len(self.audio_chunks) else None
        return frame, audio

    def get_turn_ground_truth(self, turn: int) -> Dict[str, Any]:
        """Get ground truth for a specific turn (may vary by turn)."""
        # Check if ground truth has per-turn data
        if "per_turn" in self.ground_truth:
            turn_gt = self.ground_truth["per_turn"].get(turn, {})
            # Merge with base ground truth
            return {**self.ground_truth, **turn_gt}
        return self.ground_truth


def load_environment(
    scenario: str = "integrated",
    num_test_cases: int = 10,
    max_turns: int = 10
) -> 'IntegratedEmergencyEnv':
    """
    Load integrated emergency detection environment.

    Verifiers-compatible entry point.

    Args:
        scenario: Scenario type or "all" for mixed
        num_test_cases: Number of test cases to generate
        max_turns: Maximum turns per episode

    Returns:
        Configured IntegratedEmergencyEnv
    """
    from verifiers.datasets.synthetic_generator import generate_integrated_dataset

    # Generate or load dataset
    dataset = generate_integrated_dataset(
        scenario=scenario,
        num_cases=num_test_cases,
        max_frames=max_turns
    )

    # Configure rubrics for integrated evaluation
    rubrics = [
        # Emergency detection rubrics (high weight)
        EmergencyClassificationRubric(name="emergency_type", weight=2.0),
        DispatchDecisionRubric(name="dispatch_decision", weight=2.0),

        # Multi-modal fusion rubric
        MultiModalFusionRubric(name="fusion_quality", weight=1.5),

        # Visual detection rubrics
        FallDetectionRubric(name="fall_detection", weight=1.5),
        FightDetectionRubric(name="fight_detection", weight=1.5),
        SeverityClassificationRubric(name="severity", weight=1.0),

        # Audio detection rubrics
        DistressDetectionRubric(name="distress_detection", weight=1.5),
        KeywordDetectionRubric(name="keyword_detection", weight=1.0),

        # Timing and performance
        AlertTimingRubric(name="alert_timing", weight=1.0),
        LatencyRubric(name="latency", weight=0.5, max_latency_ms=500),
    ]

    # Environment configuration
    config = EnvironmentConfig(
        name=f"integrated_emergency_{scenario}",
        timeout_ms=5000,
        max_retries=2,
        resource_limits={
            "memory_mb": 1024,
            "max_concurrent": 2,
        }
    )

    return IntegratedEmergencyEnv(
        dataset=dataset,
        rubrics=rubrics,
        config=config,
        max_turns=max_turns
    )


class IntegratedEmergencyEnv(MultiTurnEnv[MultiModalTestCase, Any]):
    """
    Multi-turn environment for integrated emergency detection.

    Processes sequences of frames and audio to test agents that
    combine visual and audio analysis.

    State Tracking:
    - current_turn: Current position in the sequence
    - alert_triggered: Whether dispatch has been recommended
    - detection_history: History of detections across turns
    """

    def __init__(
        self,
        dataset: Dataset[MultiModalTestCase],
        rubrics: List[Rubric],
        config: Optional[EnvironmentConfig] = None,
        max_turns: int = 10
    ):
        # Convert MultiModalTestCase to TestCase format
        test_cases = []
        for tc in dataset.inputs if hasattr(dataset, 'inputs') else dataset:
            if isinstance(tc, MultiModalTestCase):
                test_cases.append(TestCase(
                    id=tc.id,
                    data=tc,
                    ground_truth=tc.ground_truth,
                    metadata=tc.metadata
                ))
            else:
                test_cases.append(tc)

        converted_dataset = Dataset(
            name=dataset.name if hasattr(dataset, 'name') else "integrated",
            inputs=test_cases
        )

        super().__init__(converted_dataset, rubrics, config, max_turns=max_turns)

        # Additional state for integrated environment
        self.current_turn = 0
        self.alert_triggered = False
        self.alert_turn: Optional[int] = None
        self.detection_history: List[Dict[str, Any]] = []
        self._current_test_case: Optional[MultiModalTestCase] = None

    def reset(self, test_case: Optional[TestCase] = None) -> Dict[str, Any]:
        """Reset environment for new episode."""
        base_state = super().reset(test_case)

        self.current_turn = 0
        self.alert_triggered = False
        self.alert_turn = None
        self.detection_history = []

        if test_case and hasattr(test_case, 'data'):
            self._current_test_case = test_case.data

        return {
            **base_state,
            "turn": 0,
            "alert_triggered": False,
            "detection_history": [],
        }

    async def step(self, agent: Any, input_data: Any) -> Any:
        """
        Execute one step (process one frame/audio pair).

        Args:
            agent: The integrated detection agent
            input_data: TestCase containing MultiModalTestCase

        Returns:
            Agent's IntegratedAnalysisResult
        """
        # Extract the actual test case data
        if isinstance(input_data, TestCase):
            test_case = input_data.data
        elif isinstance(input_data, MultiModalTestCase):
            test_case = input_data
        else:
            test_case = self._current_test_case

        if test_case is None:
            raise ValueError("No test case available")

        # Get frame and audio for current turn
        frame, audio = test_case.get_turn_data(self.current_turn)

        # Call agent's analyze method
        # Compatible with NVIDIANIMIntegratedSystem.analyze_frame
        if hasattr(agent, 'analyze_frame'):
            result = await asyncio.to_thread(
                agent.analyze_frame,
                frame,
                audio
            )
        elif hasattr(agent, 'analyze'):
            result = await asyncio.to_thread(
                agent.analyze,
                frame,
                audio
            )
        else:
            # Assume agent is callable
            result = await asyncio.to_thread(
                agent,
                frame,
                audio
            )

        # Track state changes
        self._update_state(result)

        return result

    def _update_state(self, result: Any):
        """Update environment state based on agent result."""
        # Track alert
        if hasattr(result, 'dispatch_recommended') and result.dispatch_recommended:
            if not self.alert_triggered:
                self.alert_triggered = True
                self.alert_turn = self.current_turn
                self.state["alert_frame"] = self.current_turn

        # Track detection history
        detection = {
            "turn": self.current_turn,
            "emergency_type": getattr(result, 'emergency_type', None),
            "severity": getattr(result, 'overall_severity', None),
            "dispatch": getattr(result, 'dispatch_recommended', False),
        }
        self.detection_history.append(detection)
        self.state["detection_history"] = self.detection_history

        # Increment turn
        self.current_turn += 1
        self._step_count += 1

    async def run(self, agent: Any) -> Dict[str, Any]:
        """
        Run full multi-turn evaluation.

        Processes all test cases, collecting results and
        trajectories for potential training.
        """
        all_results = []

        for test_case in self.dataset:
            # Reset for new episode
            self.reset(test_case)

            tc_data = test_case.data if isinstance(test_case, TestCase) else test_case
            num_turns = tc_data.num_turns if hasattr(tc_data, 'num_turns') else self.max_turns

            turn_results = []
            episode_done = False

            while self.current_turn < min(num_turns, self.max_turns) and not episode_done:
                start_time = datetime.now()

                try:
                    # Run agent step
                    result = await asyncio.wait_for(
                        self.step(agent, test_case),
                        timeout=self.config.timeout_ms / 1000
                    )
                except asyncio.TimeoutError:
                    result = None
                except Exception as e:
                    print(f"Agent error on turn {self.current_turn}: {e}")
                    result = None

                latency_ms = (datetime.now() - start_time).total_seconds() * 1000

                # Get turn-specific ground truth
                turn_gt = tc_data.get_turn_ground_truth(self.current_turn - 1)

                # Evaluate with rubrics
                context = {
                    "turn": self.current_turn - 1,
                    "latency_ms": latency_ms,
                    "test_case_id": test_case.id,
                    "alert_triggered": self.alert_triggered,
                }
                rubric_results = await self.evaluate(result, turn_gt, context)

                turn_results.append({
                    "turn": self.current_turn - 1,
                    "result": result,
                    "rubric_results": rubric_results,
                    "latency_ms": latency_ms,
                })

                # Check for early termination
                if self._should_terminate(result, tc_data):
                    episode_done = True

            # Compile episode results
            episode_result = {
                "test_case_id": test_case.id,
                "scenario_type": tc_data.scenario_type if hasattr(tc_data, 'scenario_type') else "unknown",
                "turns": turn_results,
                "total_turns": len(turn_results),
                "alert_triggered": self.alert_triggered,
                "alert_turn": self.alert_turn,
                "final_state": dict(self.state),
                "ground_truth": test_case.ground_truth,
            }
            all_results.append(episode_result)

        self.results = all_results
        return self._aggregate_results(all_results)

    def _should_terminate(self, result: Any, test_case: MultiModalTestCase) -> bool:
        """Check if episode should terminate early."""
        # Terminate if dispatch is recommended and correct
        if self.alert_triggered:
            expected_dispatch = test_case.ground_truth.get("dispatch_recommended", False)
            if expected_dispatch:
                return True  # Correct dispatch - can stop

        return False

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate episode results into summary."""
        if not results:
            return {"overall_score": 0.0, "passed": False}

        # Collect metrics across all episodes
        rubric_scores: Dict[str, List[float]] = {}
        latencies: List[float] = []
        correct_alerts = 0
        total_alerts_expected = 0
        correct_emergency_types = 0

        for episode in results:
            for turn in episode["turns"]:
                latencies.append(turn["latency_ms"])
                for rubric_name, result in turn["rubric_results"].items():
                    if rubric_name not in rubric_scores:
                        rubric_scores[rubric_name] = []
                    rubric_scores[rubric_name].append(result.score)

            # Check alert correctness
            expected_dispatch = episode["ground_truth"].get("dispatch_recommended", False)
            total_alerts_expected += int(expected_dispatch)
            if episode["alert_triggered"] == expected_dispatch:
                correct_alerts += 1

            # Check emergency type
            expected_type = episode["ground_truth"].get("emergency_type", "none")
            if episode["turns"]:
                last_result = episode["turns"][-1].get("result")
                if last_result and hasattr(last_result, 'emergency_type'):
                    predicted_type = last_result.emergency_type.value if hasattr(
                        last_result.emergency_type, 'value'
                    ) else str(last_result.emergency_type)
                    if predicted_type.lower() == expected_type.lower():
                        correct_emergency_types += 1

        # Calculate averages
        rubric_averages = {
            name: np.mean(scores) for name, scores in rubric_scores.items()
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
            "latency_stats": {
                "mean_ms": np.mean(latencies) if latencies else 0,
                "p50_ms": np.percentile(latencies, 50) if latencies else 0,
                "p95_ms": np.percentile(latencies, 95) if latencies else 0,
                "max_ms": np.max(latencies) if latencies else 0,
            },
            "alert_accuracy": correct_alerts / len(results) if results else 0,
            "emergency_type_accuracy": correct_emergency_types / len(results) if results else 0,
            "total_episodes": len(results),
            "total_turns": sum(ep["total_turns"] for ep in results),
            "individual_results": results,
        }

    def collect_trajectory(
        self,
        agent: Any,
        test_case: TestCase,
        reward_fn: Optional[callable] = None
    ) -> Generator[TrajectoryStep, None, None]:
        """
        Generator that yields trajectory steps for RL training.

        Args:
            agent: Agent to evaluate
            test_case: Test case to run
            reward_fn: Optional custom reward function

        Yields:
            TrajectoryStep for each turn
        """
        self.reset(test_case)
        tc_data = test_case.data

        while self.current_turn < min(tc_data.num_turns, self.max_turns):
            # Capture state before step
            state = {
                "turn": self.current_turn,
                "alert_triggered": self.alert_triggered,
                "scenario_type": tc_data.scenario_type,
            }

            # Run step synchronously for generator compatibility
            import asyncio
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(self.step(agent, test_case))
            finally:
                loop.close()

            # Compute reward
            if reward_fn:
                reward = reward_fn(result, tc_data.ground_truth)
            else:
                # Default reward based on rubric scores
                reward = self._compute_default_reward(result, tc_data.ground_truth)

            # Check if done
            done = (
                self.current_turn >= tc_data.num_turns or
                self._should_terminate(result, tc_data)
            )

            yield TrajectoryStep(
                state=state,
                action=self._result_to_action(result),
                reward=reward,
                next_state={"turn": self.current_turn, "alert_triggered": self.alert_triggered},
                done=done,
                info={"test_case_id": test_case.id}
            )

    def _compute_default_reward(self, result: Any, ground_truth: Dict) -> float:
        """Compute default reward from result."""
        if result is None:
            return -1.0

        reward = 0.0

        # Reward for correct dispatch decision
        expected_dispatch = ground_truth.get("dispatch_recommended", False)
        predicted_dispatch = getattr(result, "dispatch_recommended", False)
        if expected_dispatch == predicted_dispatch:
            reward += 0.5
        elif expected_dispatch and not predicted_dispatch:
            reward -= 1.0  # Penalty for missed dispatch
        else:
            reward -= 0.25  # Smaller penalty for false alarm

        # Reward for correct emergency type
        expected_type = ground_truth.get("emergency_type", "none")
        if hasattr(result, "emergency_type"):
            predicted_type = result.emergency_type.value if hasattr(
                result.emergency_type, "value"
            ) else str(result.emergency_type)
            if predicted_type.lower() == expected_type.lower():
                reward += 0.3

        # Reward for confidence
        confidence = getattr(result, "confidence", 0)
        reward += confidence * 0.2

        return reward

    def _result_to_action(self, result: Any) -> Dict[str, Any]:
        """Convert result to action dictionary for trajectory."""
        if result is None:
            return {"error": "no_result"}

        action = {}
        for attr in ["emergency_type", "overall_severity", "dispatch_recommended",
                     "confidence", "recommended_response"]:
            if hasattr(result, attr):
                val = getattr(result, attr)
                if hasattr(val, "value"):
                    action[attr] = val.value
                elif hasattr(val, "name"):
                    action[attr] = val.name
                else:
                    action[attr] = val

        return action
