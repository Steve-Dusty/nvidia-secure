#!/usr/bin/env python3
"""
Integrated Emergency Response System

Combines visual event detection with emergency response agent:
1. Visual analysis (fight, fall, medical emergency detection)
2. Real-time emergency classification
3. Automatic dispatch via VAPI (311/911)
4. Nearest facility routing

Full pipeline: Camera → cuML Detection → Llama Analysis → Emergency Dispatch
"""

import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass

from visual_event_detector import (
    VisualEventDetector,
    DetectionResult,
    EventType
)

from emergency_response_agent import (
    EmergencyResponseAgent,
    EmergencyEvent,
    Location
)


# ============================================================================
# INTEGRATION CONFIGURATION
# ============================================================================

# Map visual events to emergency symptoms
EVENT_TO_SYMPTOMS = {
    EventType.FIGHT: ["assault", "violent altercation", "physical injury"],
    EventType.FALL: ["fall", "possible head injury", "unable to stand"],
    EventType.MEDICAL_EMERGENCY: ["collapse", "unresponsive", "medical distress"],
    EventType.CROWD_DISTURBANCE: ["crowd panic", "stampede risk", "public safety"],
    EventType.WEAPON_DETECTED: ["weapon present", "armed individual", "immediate threat"],
    EventType.FIRE_SMOKE: ["fire", "smoke inhalation", "evacuation needed"],
}

# Map visual events to patient conditions
EVENT_TO_CONDITION = {
    EventType.FIGHT: "conscious",  # Usually conscious during fight
    EventType.FALL: "altered",     # May be disoriented
    EventType.MEDICAL_EMERGENCY: "unconscious",  # Often unconscious
    EventType.CROWD_DISTURBANCE: "conscious",
    EventType.WEAPON_DETECTED: "conscious",
    EventType.FIRE_SMOKE: "conscious",
}


# ============================================================================
# CAMERA METADATA (for location tracking)
# ============================================================================

@dataclass
class CameraInfo:
    """Metadata for surveillance camera"""
    camera_id: str
    location: Location
    coverage_area: str
    viewing_angle: float
    is_active: bool


# Sample camera registry for San Francisco
CAMERA_REGISTRY = {
    "CAM-001": CameraInfo(
        camera_id="CAM-001",
        location=Location(
            latitude=37.7838,
            longitude=-122.4167,
            address="455 Golden Gate Ave, Tenderloin"
        ),
        coverage_area="Tenderloin (Golden Gate Ave corridor)",
        viewing_angle=120.0,
        is_active=True
    ),
    "CAM-002": CameraInfo(
        camera_id="CAM-002",
        location=Location(
            latitude=37.7749,
            longitude=-122.4194,
            address="Market St & 5th St, SOMA"
        ),
        coverage_area="SOMA (Market St & 5th)",
        viewing_angle=90.0,
        is_active=True
    ),
    "CAM-003": CameraInfo(
        camera_id="CAM-003",
        location=Location(
            latitude=37.7652,
            longitude=-122.4194,
            address="16th St & Mission St, Mission District"
        ),
        coverage_area="Mission District (16th & Mission)",
        viewing_angle=110.0,
        is_active=True
    ),
}


# ============================================================================
# INTEGRATED EMERGENCY SYSTEM
# ============================================================================

class IntegratedEmergencySystem:
    """
    Orchestrates visual detection and emergency response

    Pipeline:
    1. Camera feed → Visual detection (cuML)
    2. Event detected → Create emergency event
    3. Emergency event → Llama analysis
    4. Llama recommendation → VAPI dispatch
    """

    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.camera_info = CAMERA_REGISTRY.get(camera_id)

        if not self.camera_info:
            raise ValueError(f"Unknown camera ID: {camera_id}")

        # Initialize subsystems
        print(f"\n🎥 Initializing Integrated Emergency System for {camera_id}...")
        print(f"   Location: {self.camera_info.location.address}")

        self.visual_detector = VisualEventDetector()
        self.emergency_agent = EmergencyResponseAgent()

        # Event tracking
        self.active_events = {}  # event_id → EmergencyEvent
        self.detection_history = []

        print("✅ System initialized and ready")

    def process_camera_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Process single camera frame

        Returns:
            List of emergency responses triggered
        """
        # Step 1: Visual detection
        visual_detections = self.visual_detector.process_frame(frame)

        responses = []

        for detection in visual_detections:
            # Step 2: Check if emergency-level event
            if self._should_trigger_emergency(detection):

                # Step 3: Create emergency event from visual detection
                emergency_event = self._create_emergency_from_detection(detection)

                # Step 4: Process through emergency response agent
                recommendation = self.emergency_agent.process_emergency(emergency_event)

                # Step 5: Log response
                response_record = {
                    'camera_id': self.camera_id,
                    'visual_detection': detection,
                    'emergency_event': emergency_event,
                    'recommendation': recommendation,
                    'timestamp': datetime.now().isoformat()
                }

                responses.append(response_record)

                # Track active event
                self.active_events[emergency_event.event_id] = emergency_event

                # Log to history
                self.detection_history.append(response_record)

        return responses

    def _should_trigger_emergency(self, detection: DetectionResult) -> bool:
        """
        Determine if visual detection warrants emergency response

        Criteria:
        - Event type is critical (fight, fall, medical, weapon)
        - Confidence above threshold
        - Alert level is HIGH or CRITICAL
        """
        critical_events = {
            EventType.FIGHT,
            EventType.FALL,
            EventType.MEDICAL_EMERGENCY,
            EventType.WEAPON_DETECTED,
            EventType.FIRE_SMOKE
        }

        is_critical_event = detection.event_type in critical_events
        is_high_confidence = detection.confidence >= 0.70
        is_high_alert = detection.alert_level in ["HIGH", "CRITICAL"]

        return is_critical_event and is_high_confidence and is_high_alert

    def _create_emergency_from_detection(self, detection: DetectionResult) -> EmergencyEvent:
        """Convert visual detection to emergency event"""

        # Generate unique event ID
        event_id = f"{self.camera_id}-{detection.event_type.value.upper()}-{detection.frame_number}"

        # Map event type to symptoms
        symptoms = EVENT_TO_SYMPTOMS.get(detection.event_type, ["unknown emergency"])

        # Determine patient condition
        patient_condition = EVENT_TO_CONDITION.get(detection.event_type, "unknown")

        # Build context from visual detection
        additional_context = (
            f"Detected via camera {self.camera_id} at {self.camera_info.coverage_area}. "
            f"Visual confidence: {detection.confidence:.2f}. "
            f"Number of people visible: {detection.num_people}. "
            f"Motion intensity: {detection.motion_intensity:.2f}. "
        )

        # Add bounding box info
        if detection.bounding_boxes:
            additional_context += f"Detected in {len(detection.bounding_boxes)} locations within frame. "

        # Special handling for specific events
        if detection.event_type == EventType.FIGHT:
            additional_context += "Multiple individuals engaged in physical altercation. "
        elif detection.event_type == EventType.FALL:
            additional_context += "Individual observed falling to ground. "
        elif detection.event_type == EventType.MEDICAL_EMERGENCY:
            additional_context += "Individual collapsed or showing signs of medical distress. "

        # Create emergency event
        emergency_event = EmergencyEvent(
            event_id=event_id,
            timestamp=detection.timestamp,
            location=self.camera_info.location,
            symptoms=symptoms,
            patient_age=None,  # Not determinable from video
            patient_condition=patient_condition,
            witnessed_overdose=(detection.event_type == EventType.MEDICAL_EMERGENCY),
            naloxone_available=False,  # Assume not available at scene
            caller_relation="automated_camera_system",
            additional_context=additional_context
        )

        return emergency_event

    def monitor_camera_stream(self, video_source: int = 0,
                              display: bool = True,
                              save_log: bool = True):
        """
        Monitor live camera stream

        Args:
            video_source: Camera index (0 for default) or video file path
            display: Show annotated video
            save_log: Save event log to file
        """
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            raise ValueError(f"Cannot open camera/video: {video_source}")

        print(f"\n🎥 Monitoring camera stream: {self.camera_id}")
        print(f"   Location: {self.camera_info.location.address}")
        print("   Press 'q' to quit\n")

        frame_count = 0
        all_responses = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process frame (every Nth frame to reduce load)
            if frame_count % 5 == 0:  # Process every 5th frame
                responses = self.process_camera_frame(frame)

                # Handle responses
                for response in responses:
                    all_responses.append(response)

                    # Print alert
                    detection = response['visual_detection']
                    recommendation = response['recommendation']

                    print(f"\n{'='*70}")
                    print(f"🚨 EMERGENCY DETECTED - Frame {frame_count}")
                    print(f"{'='*70}")
                    print(f"Event: {detection.event_type.value.upper()}")
                    print(f"Confidence: {detection.confidence:.2f}")
                    print(f"People: {detection.num_people}")
                    print(f"Urgency: {recommendation.urgency_score}/10 ({recommendation.urgency_level})")
                    print(f"Action: {recommendation.call_number}")
                    print(f"{'='*70}\n")

            # Annotate frame
            annotated = self._annotate_integrated_frame(frame, frame_count)

            # Display
            if display:
                cv2.imshow(f'Integrated Emergency System - {self.camera_id}', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if display:
            cv2.destroyAllWindows()

        # Save log
        if save_log and all_responses:
            self._save_event_log(all_responses)

        print(f"\n✅ Monitoring complete")
        print(f"   Total frames processed: {frame_count}")
        print(f"   Emergency events detected: {len(all_responses)}")

        return all_responses

    def _annotate_integrated_frame(self, frame: np.ndarray, frame_count: int) -> np.ndarray:
        """Annotate frame with system status"""
        annotated = frame.copy()

        # System info overlay
        cv2.putText(annotated, f"Camera: {self.camera_id}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(annotated, f"Location: {self.camera_info.coverage_area}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.putText(annotated, f"Frame: {frame_count}", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Active events
        if self.active_events:
            cv2.putText(annotated, f"Active Events: {len(self.active_events)}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # System status
        status_text = "MONITORING"
        status_color = (0, 255, 0)

        if len(self.active_events) > 0:
            status_text = "ALERT"
            status_color = (0, 0, 255)

        cv2.putText(annotated, f"Status: {status_text}", (annotated.shape[1] - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        return annotated

    def _save_event_log(self, responses: List[Dict]):
        """Save event log to JSON file"""
        import json

        log_file = f"event_log_{self.camera_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert to serializable format
        log_data = []
        for response in responses:
            log_entry = {
                'camera_id': response['camera_id'],
                'timestamp': response['timestamp'],
                'event_type': response['visual_detection'].event_type.value,
                'confidence': response['visual_detection'].confidence,
                'urgency_score': response['recommendation'].urgency_score,
                'urgency_level': response['recommendation'].urgency_level,
                'call_number': response['recommendation'].call_number,
                'num_people': response['visual_detection'].num_people,
                'location': {
                    'latitude': self.camera_info.location.latitude,
                    'longitude': self.camera_info.location.longitude,
                    'address': self.camera_info.location.address
                }
            }
            log_data.append(log_entry)

        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"\n💾 Event log saved: {log_file}")


# ============================================================================
# MULTI-CAMERA MONITORING
# ============================================================================

class MultiCameraMonitor:
    """Monitor multiple cameras simultaneously"""

    def __init__(self, camera_ids: List[str]):
        self.systems = {
            cam_id: IntegratedEmergencySystem(cam_id)
            for cam_id in camera_ids
        }

        print(f"\n📹 Multi-camera monitor initialized")
        print(f"   Cameras: {len(self.systems)}")

    def monitor_all(self, video_sources: Dict[str, int]):
        """
        Monitor all cameras (simplified - single-threaded demo)

        In production, use multiprocessing/threading
        """
        # For demo, just monitor first camera
        first_cam = list(self.systems.keys())[0]
        first_source = video_sources.get(first_cam, 0)

        print(f"\n⚠️ Demo mode: Monitoring only {first_cam}")
        print("   (Multi-camera requires threading/multiprocessing)")

        self.systems[first_cam].monitor_camera_stream(first_source)


# ============================================================================
# DEMO / TESTING
# ============================================================================

def demo_single_camera():
    """Demo: Single camera monitoring"""
    print("\n" + "="*70)
    print("🎥 DEMO: SINGLE CAMERA MONITORING")
    print("="*70)

    # Create system for Tenderloin camera
    system = IntegratedEmergencySystem("CAM-001")

    # Monitor webcam (or provide video file path)
    responses = system.monitor_camera_stream(
        video_source=0,  # Webcam
        display=True,
        save_log=True
    )

    print(f"\n📊 Session Summary:")
    print(f"   Total emergencies: {len(responses)}")

    event_types = {}
    for r in responses:
        event_type = r['visual_detection'].event_type.value
        event_types[event_type] = event_types.get(event_type, 0) + 1

    for event_type, count in event_types.items():
        print(f"   {event_type.upper()}: {count}")


def demo_multi_camera():
    """Demo: Multi-camera monitoring"""
    print("\n" + "="*70)
    print("📹 DEMO: MULTI-CAMERA MONITORING")
    print("="*70)

    # Create multi-camera monitor
    monitor = MultiCameraMonitor(["CAM-001", "CAM-002", "CAM-003"])

    # Video sources for each camera
    sources = {
        "CAM-001": 0,  # Webcam
        "CAM-002": "video1.mp4",  # Video file
        "CAM-003": "video2.mp4",  # Video file
    }

    monitor.monitor_all(sources)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("🚨 INTEGRATED EMERGENCY RESPONSE SYSTEM")
    print("   Visual Detection (cuML) + Emergency Dispatch (Llama + VAPI)")
    print("="*70)

    print("\nSelect mode:")
    print("1. Single camera monitoring (live demo)")
    print("2. Multi-camera monitoring")
    print("3. Exit")

    choice = input("\nEnter choice (1-3): ")

    if choice == "1":
        demo_single_camera()
    elif choice == "2":
        demo_multi_camera()
    else:
        print("Exiting...")


if __name__ == "__main__":
    main()
