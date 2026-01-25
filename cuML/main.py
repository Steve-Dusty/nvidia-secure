#!/usr/bin/env python3
"""
NVIDIA NIM Emergency Detection System - Main Entry Point

Unified system using NVIDIA hosted models for:
1. Real-time visual detection (fight, fall, running, walking, standing)
2. Audio detection (help calls, coughing, fighting sounds, chaos)
3. Injury severity assessment with predictive response
4. Emergency dispatch integration

All inference runs on NVIDIA's cloud infrastructure (NIM).

Usage:
    python main.py                    # Interactive mode
    python main.py --webcam           # Webcam monitoring
    python main.py --video path.mp4   # Process video file
    python main.py --audio            # Include audio analysis
"""

import argparse
import os
import sys
import cv2
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

# NVIDIA NIM modules
from nvidia_nim_visual import (
    NVIDIANIMVisualInference,
    NVIDIANIMPoseEstimation,
    VisualAnalysisResult,
    ActionType,
    SeverityLevel as VisualSeverity
)

from nvidia_nim_audio import (
    NVIDIANIMAudioInference,
    AudioAnalysisResult,
    AudioEventType,
    AudioSeverity
)

from nvidia_nim_integrated import (
    NVIDIANIMIntegratedSystem,
    IntegratedAnalysisResult,
    EmergencyType,
    OverallSeverity,
    example_alert_handler
)

# Optional emergency dispatch (Llama-based)
try:
    sys.path.insert(0, '../response-output')
    from emergency_response_agent import EmergencyResponseAgent, EmergencyEvent, Location
    DISPATCH_AVAILABLE = True
except ImportError:
    DISPATCH_AVAILABLE = False


# ============================================================================
# CAMERA REGISTRY
# ============================================================================

@dataclass
class CameraConfig:
    camera_id: str
    latitude: float
    longitude: float
    address: str


CAMERAS = {
    "CAM-001": CameraConfig("CAM-001", 37.7838, -122.4167, "455 Golden Gate Ave, Tenderloin"),
    "CAM-002": CameraConfig("CAM-002", 37.7749, -122.4194, "Market St & 5th St, SOMA"),
    "CAM-003": CameraConfig("CAM-003", 37.7652, -122.4194, "16th St & Mission St, Mission"),
}


# ============================================================================
# NVIDIA NIM EMERGENCY DETECTION SYSTEM
# ============================================================================

class EmergencyDetectionSystem:
    """
    Unified emergency detection using NVIDIA NIM hosted models.

    Visual Pipeline (NVIDIA NIM):
        Camera → Florence-2/DINO (action detection) → Severity Assessment → Dispatch

    Audio Pipeline (NVIDIA NIM):
        Microphone → Parakeet ASR (speech) → Audio Classification → Alert

    Models:
        Visual:
            - microsoft/florence-2: Scene understanding, action detection
            - nvidia/grounding-dino: Person detection and tracking
            - meta/sam2-hiera-large: Precise segmentation

        Audio:
            - nvidia/parakeet-ctc-1.1b: Speech recognition (ASR)
            - nvidia/canary-1b: Multilingual ASR
            - nvidia/audio-embedding: Sound event classification

    Note: Llama model (for dispatch reasoning) is NOT replaced - it remains NGC fine-tuned.
    """

    def __init__(self, camera_id: str = "CAM-001",
                 enable_dispatch: bool = True,
                 enable_audio: bool = False,
                 api_key: Optional[str] = None):

        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY environment variable required")

        self.camera = CAMERAS.get(camera_id, CAMERAS["CAM-001"])

        print(f"\n{'='*60}")
        print(f"NVIDIA NIM EMERGENCY DETECTION SYSTEM")
        print(f"{'='*60}")
        print(f"Camera: {self.camera.camera_id}")
        print(f"Location: {self.camera.address}")
        print(f"Audio: {'ENABLED' if enable_audio else 'DISABLED'}")
        print()

        # Initialize NVIDIA NIM analyzers
        print("Initializing NVIDIA NIM models...")
        self.integrated_system = NVIDIANIMIntegratedSystem(self.api_key)

        # Set alert callback
        self.integrated_system.set_alert_callback(self._handle_alert)

        print("  Visual: microsoft/florence-2, nvidia/grounding-dino")
        print("  Audio:  nvidia/parakeet-ctc-1.1b, nvidia/audio-embedding")

        # Optional Llama-based dispatch (NOT replaced - uses NGC fine-tuned Llama)
        self.dispatch_enabled = enable_dispatch and DISPATCH_AVAILABLE
        if self.dispatch_enabled:
            self.emergency_agent = EmergencyResponseAgent()
            print("  Dispatch: ENABLED (NGC Llama)")
        else:
            self.emergency_agent = None
            print("  Dispatch: DISABLED")

        # Audio support
        self.audio_enabled = enable_audio
        self.audio_buffer = []

        # State
        self.stats = {'frames': 0, 'events': 0, 'alerts': 0}
        self.alert_history = []

        print(f"\n{'='*60}")
        print("System ready!")
        print(f"{'='*60}\n")

    def _handle_alert(self, result: IntegratedAnalysisResult):
        """Handle emergency alert from integrated system"""
        self.stats['alerts'] += 1
        self.alert_history.append(result)

        # Print alert
        print(f"\n{'!'*60}")
        print(f"EMERGENCY ALERT #{self.stats['alerts']}")
        print(f"{'!'*60}")
        print(f"Type: {result.emergency_type.value.upper()}")
        print(f"Severity: {result.overall_severity.name}")
        print(f"Confidence: {result.confidence:.1%}")
        print()
        print(f"Detections:")
        if result.fight_detected:
            print(f"  - FIGHT DETECTED")
        if result.fall_detected:
            print(f"  - FALL DETECTED")
        if result.help_detected:
            print(f"  - HELP CALL: '{result.transcript}'")
        if result.coughing_detected:
            print(f"  - COUGHING DETECTED")
        if result.medical_emergency:
            print(f"  - MEDICAL EMERGENCY")
        print()
        print(f"RECOMMENDATION: {result.recommended_response}")
        print(f"Priority: {result.response_priority}")
        print(f"{'!'*60}\n")

        # Dispatch if enabled and critical
        if self.dispatch_enabled and result.overall_severity.value >= OverallSeverity.HIGH.value:
            self._dispatch(result)

    def _dispatch(self, result: IntegratedAnalysisResult):
        """Dispatch emergency response using Llama agent"""
        if not self.emergency_agent:
            return

        # Build symptoms from detections
        symptoms = []
        if result.fight_detected:
            symptoms.append("physical_altercation")
        if result.fall_detected:
            symptoms.append("person_fallen")
        if result.help_detected:
            symptoms.append("distress_call")
        if result.coughing_detected:
            symptoms.append("respiratory_distress")
        if result.medical_emergency:
            symptoms.append("medical_emergency")

        # Add visual actions
        for action in result.visual_actions:
            symptoms.append(action.value)

        event = EmergencyEvent(
            event_id=f"{self.camera.camera_id}-{self.stats['alerts']}",
            timestamp=datetime.now(),
            location=Location(
                self.camera.latitude,
                self.camera.longitude,
                self.camera.address
            ),
            symptoms=symptoms,
            patient_age=None,
            patient_condition="critical" if result.overall_severity == OverallSeverity.CRITICAL else "unknown",
            witnessed_overdose=False,
            naloxone_available=False,
            caller_relation="automated_camera",
            additional_context=f"NIM Detection - Type: {result.emergency_type.value}, "
                             f"Persons: {result.persons_detected}, "
                             f"Transcript: {result.transcript or 'N/A'}"
        )

        try:
            print(f"\nDispatching via Llama agent...")
            self.emergency_agent.process_emergency(event)
        except Exception as e:
            print(f"Dispatch error: {e}")

    def process_frame(self, frame: np.ndarray,
                      audio: Optional[np.ndarray] = None) -> IntegratedAnalysisResult:
        """Process single frame through NVIDIA NIM pipeline"""
        self.stats['frames'] += 1

        # Run integrated analysis
        result = self.integrated_system.analyze_frame(frame, audio)

        # Track events
        if result.persons_detected > 0 or result.audio_events:
            self.stats['events'] += 1

        return result

    def run(self, video_source=0, display: bool = True, save_log: bool = True):
        """Run real-time monitoring using NVIDIA NIM"""
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_source}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Monitoring: {video_source}")
        print(f"Resolution: {width}x{height} @ {fps:.1f} FPS")
        print("Press 'q' to quit\n")

        all_results = []
        skip_frames = 5  # Process every 5th frame for API efficiency

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.stats['frames'] += 1

            # Process every Nth frame
            if self.stats['frames'] % skip_frames == 0:
                try:
                    result = self.process_frame(frame)
                    all_results.append({
                        'frame': self.stats['frames'],
                        'timestamp': result.timestamp.isoformat(),
                        'severity': result.overall_severity.name,
                        'emergency_type': result.emergency_type.value,
                        'persons': result.persons_detected,
                        'actions': [a.value for a in result.visual_actions],
                        'inference_ms': result.total_inference_ms
                    })

                    # Status update
                    if self.stats['frames'] % 50 == 0:
                        print(f"Frame {self.stats['frames']}: "
                              f"Persons={result.persons_detected}, "
                              f"Severity={result.overall_severity.name}, "
                              f"Latency={result.total_inference_ms:.0f}ms")

                except Exception as e:
                    print(f"Processing error at frame {self.stats['frames']}: {e}")

            # Display
            if display:
                annotated = self._annotate(frame)
                cv2.imshow('NVIDIA NIM Emergency Detection', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if display:
            cv2.destroyAllWindows()

        # Save log
        if save_log and all_results:
            log_file = f"nim_detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(log_file, 'w') as f:
                json.dump({
                    'session': {
                        'camera': self.camera.camera_id,
                        'location': self.camera.address,
                        'start_time': all_results[0]['timestamp'] if all_results else None,
                        'end_time': all_results[-1]['timestamp'] if all_results else None,
                    },
                    'stats': self.stats,
                    'alert_summary': self.integrated_system.get_alert_summary(),
                    'frames': all_results
                }, f, indent=2)
            print(f"\nLog saved: {log_file}")

        # Summary
        print(f"\n{'='*60}")
        print("SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"  Frames processed: {self.stats['frames']}")
        print(f"  Events detected: {self.stats['events']}")
        print(f"  Alerts triggered: {self.stats['alerts']}")
        print(f"  Alert breakdown: {self.integrated_system.get_alert_summary()}")

        return all_results

    def _annotate(self, frame: np.ndarray) -> np.ndarray:
        """Annotate frame with status"""
        annotated = frame.copy()

        # Header
        cv2.rectangle(annotated, (0, 0), (300, 100), (0, 0, 0), -1)
        cv2.putText(annotated, "NVIDIA NIM Detection", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated, f"Camera: {self.camera.camera_id}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated, f"Frame: {self.stats['frames']}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Alert indicator
        alert_color = (0, 0, 255) if self.stats['alerts'] else (255, 255, 255)
        cv2.putText(annotated, f"Alerts: {self.stats['alerts']}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, alert_color, 1)

        return annotated


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='NVIDIA NIM Emergency Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models used (all NVIDIA NIM hosted):
  Visual:
    - microsoft/florence-2: Scene understanding, action detection
    - nvidia/grounding-dino: Person detection
    - meta/sam2-hiera-large: Segmentation

  Audio:
    - nvidia/parakeet-ctc-1.1b: Speech recognition
    - nvidia/canary-1b: Multilingual ASR
    - nvidia/audio-embedding: Sound classification

  Dispatch (unchanged):
    - NGC fine-tuned Llama: Emergency response reasoning

Environment:
  NVIDIA_API_KEY: Required for NIM API access
        """
    )
    parser.add_argument('--webcam', action='store_true', help='Use webcam')
    parser.add_argument('--video', type=str, help='Video file path')
    parser.add_argument('--camera', type=str, default='CAM-001', help='Camera ID')
    parser.add_argument('--audio', action='store_true', help='Enable audio analysis')
    parser.add_argument('--no-display', action='store_true', help='Disable display')
    parser.add_argument('--no-dispatch', action='store_true', help='Disable dispatch')
    args = parser.parse_args()

    # Check API key
    if not os.environ.get("NVIDIA_API_KEY"):
        print("ERROR: NVIDIA_API_KEY environment variable not set")
        print("Get your key from: https://build.nvidia.com/")
        sys.exit(1)

    try:
        system = EmergencyDetectionSystem(
            camera_id=args.camera,
            enable_dispatch=not args.no_dispatch,
            enable_audio=args.audio
        )

        if args.webcam:
            system.run(video_source=0, display=not args.no_display)
        elif args.video:
            system.run(video_source=args.video, display=not args.no_display)
        else:
            # Interactive mode
            print("\nSelect mode:")
            print("1. Webcam")
            print("2. Video file")
            print("3. Exit")

            choice = input("\nChoice (1-3): ").strip()

            if choice == "1":
                system.run(video_source=0, display=True)
            elif choice == "2":
                path = input("Video path: ").strip()
                system.run(video_source=path, display=True)
            else:
                print("Exiting...")

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
