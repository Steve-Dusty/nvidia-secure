#!/usr/bin/env python3
"""
NVIDIA NIM Inference - ARM Architecture (DGX Spark)

Optimized inference runner for NVIDIA DGX Spark with ARM (Grace) CPU.
All model inference runs on NVIDIA NIM cloud APIs.

Hardware Target:
    - NVIDIA DGX Spark
    - ARM64 (aarch64) architecture
    - Grace CPU + NVIDIA GPU

Models (NVIDIA NIM - Cloud Hosted):
    Visual:
        - microsoft/florence-2: Scene understanding, action detection
        - nvidia/grounding-dino: Person detection
        - nvidia/bodypose-estimation: 17-keypoint pose

    Audio:
        - nvidia/parakeet-ctc-1.1b: Speech recognition (ASR)
        - nvidia/canary-1b: Multilingual ASR
        - nvidia/audio-embedding: Sound classification

Usage:
    python nim_inference_arm.py --webcam
    python nim_inference_arm.py --video path/to/video.mp4
    python nim_inference_arm.py --stream rtsp://camera/stream
"""

import argparse
import os
import sys
import platform
import time
import json
from datetime import datetime
from typing import Optional, Dict, List

import cv2
import numpy as np

# Import NIM modules
from nvidia_nim_visual import (
    NVIDIANIMVisualInference,
    NVIDIANIMPoseEstimation,
    VisualAnalysisResult,
    ActionType,
    SeverityLevel
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
    OverallSeverity
)


class ARMInferenceRunner:
    """
    Inference runner optimized for ARM architecture (DGX Spark).

    Features:
    - Efficient frame batching for API calls
    - ARM-optimized OpenCV operations
    - Async-ready for high throughput
    - Memory-efficient processing
    """

    def __init__(self, enable_audio: bool = False):
        self.api_key = os.environ.get("NVIDIA_API_KEY", "")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY environment variable required")

        # Print system info
        self._print_system_info()

        # Initialize NIM system
        print("\nInitializing NVIDIA NIM models...")
        self.system = NVIDIANIMIntegratedSystem(self.api_key)
        self.system.set_alert_callback(self._on_alert)

        self.enable_audio = enable_audio
        self.stats = {
            "frames_processed": 0,
            "alerts_triggered": 0,
            "start_time": None,
            "total_inference_ms": 0
        }

        print("System ready!")

    def _print_system_info(self):
        """Print system architecture information"""
        print("\n" + "=" * 60)
        print("NVIDIA NIM Inference - ARM Architecture")
        print("=" * 60)
        print(f"Platform:     {platform.system()} {platform.release()}")
        print(f"Architecture: {platform.machine()}")
        print(f"Processor:    {platform.processor() or 'ARM64'}")
        print(f"Python:       {platform.python_version()}")

        # Check if running on ARM
        arch = platform.machine().lower()
        if arch in ["aarch64", "arm64"]:
            print(f"ARM Status:   Optimized for DGX Spark")
        else:
            print(f"ARM Status:   Running in compatibility mode")

        print("=" * 60)

    def _on_alert(self, result: IntegratedAnalysisResult):
        """Handle emergency alerts"""
        self.stats["alerts_triggered"] += 1

        print(f"\n{'!'*60}")
        print(f"ALERT #{self.stats['alerts_triggered']}")
        print(f"{'!'*60}")
        print(f"Type:       {result.emergency_type.value}")
        print(f"Severity:   {result.overall_severity.name}")
        print(f"Confidence: {result.confidence:.1%}")

        if result.fight_detected:
            print("  -> FIGHT DETECTED")
        if result.fall_detected:
            print("  -> FALL DETECTED")
        if result.help_detected:
            print(f"  -> HELP CALL: '{result.transcript}'")
        if result.coughing_detected:
            print("  -> COUGHING DETECTED")

        print(f"\nAction: {result.recommended_response}")
        print(f"{'!'*60}\n")

    def process_video(self, source, display: bool = True, skip_frames: int = 5):
        """
        Process video source with NIM inference.

        Args:
            source: Video source (0 for webcam, path for file, URL for stream)
            display: Show video window
            skip_frames: Process every Nth frame
        """
        # Open video source
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\nVideo source: {source}")
        print(f"Resolution:   {width}x{height} @ {fps:.1f} FPS")
        print(f"Skip frames:  {skip_frames} (processing every {skip_frames}th frame)")
        print(f"Display:      {'Yes' if display else 'No (headless)'}")
        print("\nPress 'q' to quit\n")

        self.stats["start_time"] = time.time()
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Skip frames for efficiency
                if frame_count % skip_frames != 0:
                    if display:
                        cv2.imshow("NIM Inference (ARM)", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    continue

                # Run inference
                start_time = time.time()
                result = self.system.analyze_frame(frame, None)
                inference_time = (time.time() - start_time) * 1000

                self.stats["frames_processed"] += 1
                self.stats["total_inference_ms"] += inference_time

                # Log progress
                if self.stats["frames_processed"] % 10 == 0:
                    avg_inference = self.stats["total_inference_ms"] / self.stats["frames_processed"]
                    print(f"Frame {frame_count}: "
                          f"Persons={result.persons_detected}, "
                          f"Severity={result.overall_severity.name}, "
                          f"Latency={inference_time:.0f}ms (avg: {avg_inference:.0f}ms)")

                # Display
                if display:
                    annotated = self._annotate_frame(frame, result)
                    cv2.imshow("NIM Inference (ARM)", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()

        self._print_summary()

    def _annotate_frame(self, frame: np.ndarray, result: IntegratedAnalysisResult) -> np.ndarray:
        """Annotate frame with detection results"""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Header background
        cv2.rectangle(annotated, (0, 0), (350, 120), (0, 0, 0), -1)

        # Title
        cv2.putText(annotated, "NVIDIA NIM - DGX Spark ARM",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Stats
        cv2.putText(annotated, f"Persons: {result.persons_detected}",
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated, f"Severity: {result.overall_severity.name}",
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated, f"Latency: {result.total_inference_ms:.0f}ms",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Alert indicator
        if result.overall_severity.value >= OverallSeverity.HIGH.value:
            cv2.putText(annotated, "ALERT!",
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Detection flags
        flags = []
        if result.fight_detected:
            flags.append("FIGHT")
        if result.fall_detected:
            flags.append("FALL")
        if result.help_detected:
            flags.append("HELP")

        if flags:
            cv2.putText(annotated, " | ".join(flags),
                       (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return annotated

    def _print_summary(self):
        """Print session summary"""
        elapsed = time.time() - self.stats["start_time"]
        fps = self.stats["frames_processed"] / elapsed if elapsed > 0 else 0
        avg_latency = (self.stats["total_inference_ms"] / self.stats["frames_processed"]
                      if self.stats["frames_processed"] > 0 else 0)

        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        print(f"Duration:          {elapsed:.1f} seconds")
        print(f"Frames processed:  {self.stats['frames_processed']}")
        print(f"Effective FPS:     {fps:.1f}")
        print(f"Avg latency:       {avg_latency:.0f}ms")
        print(f"Alerts triggered:  {self.stats['alerts_triggered']}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="NVIDIA NIM Inference - ARM Architecture (DGX Spark)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models (NVIDIA NIM - Cloud Hosted):
  Visual:
    - microsoft/florence-2      Action detection
    - nvidia/grounding-dino     Person detection
    - nvidia/bodypose-estimation Pose analysis

  Audio:
    - nvidia/parakeet-ctc-1.1b  Speech recognition
    - nvidia/audio-embedding    Sound classification

Hardware Target:
  - NVIDIA DGX Spark
  - ARM64 (aarch64) architecture
  - Grace CPU + NVIDIA GPU

Environment:
  NVIDIA_API_KEY: Required for NIM API access
        """
    )

    parser.add_argument("--webcam", action="store_true",
                       help="Use webcam (device 0)")
    parser.add_argument("--video", type=str,
                       help="Video file path")
    parser.add_argument("--stream", type=str,
                       help="RTSP/HTTP stream URL")
    parser.add_argument("--audio", action="store_true",
                       help="Enable audio analysis")
    parser.add_argument("--no-display", action="store_true",
                       help="Headless mode (no GUI)")
    parser.add_argument("--skip-frames", type=int, default=5,
                       help="Process every Nth frame (default: 5)")

    args = parser.parse_args()

    # Check API key
    if not os.environ.get("NVIDIA_API_KEY"):
        print("ERROR: NVIDIA_API_KEY environment variable not set")
        print("Get your key from: https://build.nvidia.com/")
        sys.exit(1)

    # Determine source
    if args.webcam:
        source = 0
    elif args.video:
        source = args.video
    elif args.stream:
        source = args.stream
    else:
        # Interactive mode
        print("\nSelect input source:")
        print("1. Webcam")
        print("2. Video file")
        print("3. Stream URL")
        print("4. Exit")

        choice = input("\nChoice (1-4): ").strip()

        if choice == "1":
            source = 0
        elif choice == "2":
            source = input("Video path: ").strip()
        elif choice == "3":
            source = input("Stream URL: ").strip()
        else:
            print("Exiting...")
            sys.exit(0)

    # Run inference
    try:
        runner = ARMInferenceRunner(enable_audio=args.audio)
        runner.process_video(
            source=source,
            display=not args.no_display,
            skip_frames=args.skip_frames
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
