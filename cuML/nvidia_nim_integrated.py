#!/usr/bin/env python3
"""
NVIDIA NIM Integrated Emergency Detection System

Combines visual and audio inference using NVIDIA hosted models for:
- Real-time emergency detection
- Multi-modal analysis (video + audio)
- Predictive response recommendations

All inference runs on NVIDIA's cloud infrastructure (NIM).
"""

import os
import time
import threading
from queue import Queue
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Generator
from datetime import datetime
import numpy as np

from nvidia_nim_visual import (
    NVIDIANIMVisualInference,
    NVIDIANIMPoseEstimation,
    VisualAnalysisResult,
    ActionType,
    SeverityLevel as VisualSeverity,
    DetectedPerson
)
from nvidia_nim_audio import (
    NVIDIANIMAudioInference,
    AudioAnalysisResult,
    AudioEventType,
    AudioSeverity
)


class EmergencyType(Enum):
    """Types of detected emergencies"""
    NONE = "none"
    MEDICAL = "medical"
    VIOLENCE = "violence"
    FALL = "fall"
    DISTRESS = "distress"
    CROWD_INCIDENT = "crowd_incident"
    UNKNOWN = "unknown"


class OverallSeverity(Enum):
    """Combined severity levels"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class IntegratedAnalysisResult:
    """Combined result from visual + audio analysis"""
    timestamp: datetime
    frame_id: int

    # Visual results
    visual_result: Optional[VisualAnalysisResult] = None
    persons_detected: int = 0
    visual_actions: List[ActionType] = field(default_factory=list)

    # Audio results
    audio_result: Optional[AudioAnalysisResult] = None
    transcript: Optional[str] = None
    audio_events: List[AudioEventType] = field(default_factory=list)

    # Combined analysis
    emergency_type: EmergencyType = EmergencyType.NONE
    overall_severity: OverallSeverity = OverallSeverity.NONE
    confidence: float = 0.0

    # Detection flags
    fight_detected: bool = False
    fall_detected: bool = False
    help_detected: bool = False
    coughing_detected: bool = False
    medical_emergency: bool = False

    # Recommendations
    dispatch_recommended: bool = False
    recommended_response: str = ""
    response_priority: int = 0  # 1-5, 1 highest

    # Timing
    visual_inference_ms: float = 0.0
    audio_inference_ms: float = 0.0
    total_inference_ms: float = 0.0


class NVIDIANIMIntegratedSystem:
    """
    Integrated emergency detection using NVIDIA NIM.

    Combines:
    - Visual: Nemotron VL, Grounding DINO, SAM2 for action/person detection
    - Audio: Parakeet, Canary for speech/sound detection
    - Logic: Multi-modal fusion for emergency classification

    All models run on NVIDIA cloud infrastructure.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize integrated system"""
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")

        # Initialize analyzers
        self.visual_analyzer = NVIDIANIMVisualInference(self.api_key)
        self.audio_analyzer = NVIDIANIMAudioInference(self.api_key)
        self.pose_estimator = NVIDIANIMPoseEstimation(self.api_key)

        # State tracking
        self.frame_count = 0
        self.alert_history: List[IntegratedAnalysisResult] = []
        self.consecutive_alerts = 0

        # Callbacks
        self.alert_callback: Optional[Callable[[IntegratedAnalysisResult], None]] = None

    def set_alert_callback(self, callback: Callable[[IntegratedAnalysisResult], None]):
        """Set callback for emergency alerts"""
        self.alert_callback = callback

    def _combine_severities(self, visual_severity: VisualSeverity,
                            audio_severity: AudioSeverity) -> OverallSeverity:
        """Combine visual and audio severities"""
        v_val = visual_severity.value if visual_severity else 0
        a_val = audio_severity.value if audio_severity else 0

        # Take maximum, with boost if both are elevated
        max_severity = max(v_val, a_val)

        # Boost severity if both modalities detect issues
        if v_val >= 2 and a_val >= 2:
            max_severity = min(4, max_severity + 1)

        return OverallSeverity(max_severity)

    def _classify_emergency(self, visual: Optional[VisualAnalysisResult],
                            audio: Optional[AudioAnalysisResult]) -> EmergencyType:
        """Classify emergency type from combined analysis"""
        # Check for violence (fighting)
        fight_visual = visual and visual.fight_detected
        fight_audio = audio and audio.fighting_detected

        if fight_visual or fight_audio:
            return EmergencyType.VIOLENCE

        # Check for fall
        if visual and visual.fall_detected:
            return EmergencyType.FALL

        # Check for distress call
        if audio and audio.help_detected:
            return EmergencyType.DISTRESS

        # Check for medical (coughing, collapse)
        if audio and audio.coughing_detected:
            return EmergencyType.MEDICAL

        if visual:
            for person in visual.persons:
                if person.action in [ActionType.LYING_DOWN, ActionType.FALLING]:
                    return EmergencyType.MEDICAL

        # Check for crowd incident
        if visual and visual.crowd_density > 0.5:
            return EmergencyType.CROWD_INCIDENT

        return EmergencyType.NONE

    def _generate_response_recommendation(self, result: IntegratedAnalysisResult) -> tuple:
        """Generate response recommendation based on analysis"""
        if result.overall_severity == OverallSeverity.CRITICAL:
            if result.fight_detected:
                return (True, "Dispatch police and EMS immediately. Multiple units needed for violent altercation.", 1)
            elif result.help_detected:
                return (True, "Immediate dispatch required. Verbal distress call detected.", 1)
            elif result.fall_detected:
                return (True, "Medical emergency - person down. Dispatch EMS with trauma capability.", 1)
            else:
                return (True, "Critical incident detected. Dispatch emergency services.", 1)

        elif result.overall_severity == OverallSeverity.HIGH:
            if result.fight_detected:
                return (True, "Potential violence detected. Dispatch police for assessment.", 2)
            elif result.fall_detected:
                return (True, "Person fallen. Dispatch EMS for medical evaluation.", 2)
            elif result.medical_emergency:
                return (True, "Medical emergency indicators. Dispatch EMS.", 2)
            else:
                return (True, "High-priority incident. Recommend dispatch for assessment.", 2)

        elif result.overall_severity == OverallSeverity.MEDIUM:
            if result.coughing_detected:
                return (False, "Possible medical issue (coughing). Monitor and offer assistance.", 3)
            else:
                return (False, "Elevated activity detected. Continue monitoring.", 3)

        elif result.overall_severity == OverallSeverity.LOW:
            return (False, "Minor activity detected. No action required.", 4)

        return (False, "Normal activity. No action required.", 5)

    def analyze_frame(self,
                      image: Optional[np.ndarray] = None,
                      audio: Optional[np.ndarray] = None,
                      sample_rate: int = 16000) -> IntegratedAnalysisResult:
        """
        Analyze a single frame with optional audio.

        Args:
            image: Video frame as numpy array (H, W, C)
            audio: Audio samples as numpy array
            sample_rate: Audio sample rate (default 16kHz)

        Returns:
            IntegratedAnalysisResult with combined analysis
        """
        start_time = time.time()
        self.frame_count += 1

        visual_result = None
        audio_result = None

        # Visual analysis
        visual_start = time.time()
        if image is not None:
            visual_result = self.visual_analyzer.analyze_frame(image)
        visual_time = (time.time() - visual_start) * 1000

        # Audio analysis
        audio_start = time.time()
        if audio is not None:
            audio_result = self.audio_analyzer.analyze_audio(audio, sample_rate)
        audio_time = (time.time() - audio_start) * 1000

        # Combine results
        fight_detected = (
            (visual_result and visual_result.fight_detected) or
            (audio_result and audio_result.fighting_detected)
        )
        fall_detected = visual_result and visual_result.fall_detected
        help_detected = audio_result and audio_result.help_detected
        coughing_detected = audio_result and audio_result.coughing_detected

        # Medical emergency detection
        medical_emergency = (
            coughing_detected or
            fall_detected or
            (visual_result and any(
                p.action in [ActionType.LYING_DOWN, ActionType.FALLING]
                for p in visual_result.persons
            ))
        )

        # Combine severities
        visual_severity = visual_result.scene_severity if visual_result else VisualSeverity.NONE
        audio_severity = audio_result.severity if audio_result else AudioSeverity.NONE
        overall_severity = self._combine_severities(visual_severity, audio_severity)

        # Classify emergency
        emergency_type = self._classify_emergency(visual_result, audio_result)

        # Calculate confidence
        confidences = []
        if visual_result:
            confidences.append(max((p.confidence for p in visual_result.persons), default=0))
        if audio_result:
            confidences.append(max((e.confidence for e in audio_result.events), default=0))
        confidence = max(confidences) if confidences else 0.0

        # Create result
        result = IntegratedAnalysisResult(
            timestamp=datetime.now(),
            frame_id=self.frame_count,
            visual_result=visual_result,
            persons_detected=len(visual_result.persons) if visual_result else 0,
            visual_actions=[p.action for p in visual_result.persons] if visual_result else [],
            audio_result=audio_result,
            transcript=audio_result.transcript if audio_result else None,
            audio_events=[e.event_type for e in audio_result.events] if audio_result else [],
            emergency_type=emergency_type,
            overall_severity=overall_severity,
            confidence=confidence,
            fight_detected=fight_detected,
            fall_detected=fall_detected,
            help_detected=help_detected,
            coughing_detected=coughing_detected,
            medical_emergency=medical_emergency,
            visual_inference_ms=visual_time,
            audio_inference_ms=audio_time,
            total_inference_ms=(time.time() - start_time) * 1000
        )

        # Generate response recommendation
        dispatch, recommendation, priority = self._generate_response_recommendation(result)
        result.dispatch_recommended = dispatch
        result.recommended_response = recommendation
        result.response_priority = priority

        # Track alerts
        if overall_severity.value >= OverallSeverity.HIGH.value:
            self.consecutive_alerts += 1
            self.alert_history.append(result)

            # Trigger callback if set
            if self.alert_callback:
                self.alert_callback(result)
        else:
            self.consecutive_alerts = 0

        return result

    def analyze_stream(self,
                       frame_generator: Generator,
                       audio_generator: Optional[Generator] = None,
                       skip_frames: int = 5,
                       callback: Optional[Callable] = None):
        """
        Analyze video/audio stream.

        Args:
            frame_generator: Yields (frame_id, image) tuples
            audio_generator: Optional, yields (timestamp, audio_chunk) tuples
            skip_frames: Process every Nth frame
            callback: Called with each IntegratedAnalysisResult
        """
        audio_queue = Queue(maxsize=10)
        current_audio = None

        # Audio collection thread if audio provided
        def collect_audio():
            nonlocal current_audio
            if audio_generator:
                for ts, audio_chunk in audio_generator:
                    audio_queue.put(audio_chunk)
                    current_audio = audio_chunk

        if audio_generator:
            audio_thread = threading.Thread(target=collect_audio, daemon=True)
            audio_thread.start()

        # Process video frames
        for frame_id, image in frame_generator:
            if frame_id % skip_frames != 0:
                continue

            # Get latest audio if available
            audio = None
            if not audio_queue.empty():
                audio = audio_queue.get()
            elif current_audio is not None:
                audio = current_audio

            # Analyze
            result = self.analyze_frame(image, audio)

            if callback:
                callback(result)

            yield result

    def get_alert_summary(self) -> Dict:
        """Get summary of recent alerts"""
        if not self.alert_history:
            return {"total_alerts": 0, "by_type": {}, "by_severity": {}}

        by_type = {}
        by_severity = {}

        for alert in self.alert_history[-100:]:  # Last 100 alerts
            # Count by type
            etype = alert.emergency_type.value
            by_type[etype] = by_type.get(etype, 0) + 1

            # Count by severity
            sev = alert.overall_severity.name
            by_severity[sev] = by_severity.get(sev, 0) + 1

        return {
            "total_alerts": len(self.alert_history),
            "recent_alerts": len(self.alert_history[-100:]),
            "by_type": by_type,
            "by_severity": by_severity,
            "consecutive_alerts": self.consecutive_alerts
        }


def create_integrated_system(api_key: Optional[str] = None) -> NVIDIANIMIntegratedSystem:
    """Create configured integrated system instance"""
    return NVIDIANIMIntegratedSystem(api_key)


# Example alert handler for integration with dispatch
def example_alert_handler(result: IntegratedAnalysisResult):
    """Example alert handler - integrate with your dispatch system"""
    print(f"\n{'='*60}")
    print(f"EMERGENCY ALERT - {result.timestamp}")
    print(f"{'='*60}")
    print(f"Type: {result.emergency_type.value}")
    print(f"Severity: {result.overall_severity.name}")
    print(f"Confidence: {result.confidence:.1%}")
    print()
    print(f"Detections:")
    print(f"  - Fight: {result.fight_detected}")
    print(f"  - Fall: {result.fall_detected}")
    print(f"  - Help call: {result.help_detected}")
    print(f"  - Medical: {result.medical_emergency}")
    if result.transcript:
        print(f"  - Transcript: '{result.transcript}'")
    print()
    print(f"RECOMMENDATION: {result.recommended_response}")
    print(f"Dispatch Required: {result.dispatch_recommended}")
    print(f"Priority: {result.response_priority}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys

    print("NVIDIA NIM Integrated Emergency Detection System")
    print("=" * 60)
    print()
    print("Visual Models (NVIDIA NIM):")
    print("  - microsoft/florence-2: Scene understanding & action detection")
    print("  - nvidia/grounding-dino: Person detection")
    print("  - meta/sam2-hiera-large: Precise segmentation")
    print()
    print("Audio Models (NVIDIA NIM):")
    print("  - nvidia/parakeet-ctc-1.1b: Speech recognition")
    print("  - nvidia/canary-1b: Multilingual ASR")
    print("  - nvidia/audio-embedding: Sound classification")
    print()
    print("Detection Capabilities:")
    print("  Visual: running, standing, walking, fighting, falling")
    print("  Audio: help calls, coughing, screams, fighting sounds, chaos")
    print()

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("ERROR: Set NVIDIA_API_KEY environment variable")
        sys.exit(1)

    print("API key configured - system ready")
    print()

    # Create system with example alert handler
    system = create_integrated_system(api_key)
    system.set_alert_callback(example_alert_handler)

    print("To use:")
    print("  result = system.analyze_frame(image, audio)")
    print("  # or")
    print("  for result in system.analyze_stream(frames, audio):")
    print("      process(result)")
