#!/usr/bin/env python3
"""
NVIDIA NIM Visual Inference Module

Uses NVIDIA hosted models (NIM) for visual detection:
- Person detection and tracking
- Action recognition: running, standing, walking, fighting, falling
- Pose estimation for movement analysis

All inference runs on NVIDIA's cloud infrastructure via API.
"""

import os
import base64
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import requests
import numpy as np

# NVIDIA NIM API Configuration
NVIDIA_API_BASE = "https://ai.api.nvidia.com/v1"
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")


class ActionType(Enum):
    """Detected human actions"""
    STANDING = "standing"
    WALKING = "walking"
    RUNNING = "running"
    FIGHTING = "fighting"
    FALLING = "falling"
    LYING_DOWN = "lying_down"
    UNKNOWN = "unknown"


class SeverityLevel(Enum):
    """Emergency severity levels"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class DetectedPerson:
    """Represents a detected person with pose and action"""
    person_id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 (normalized)
    confidence: float
    action: ActionType
    action_confidence: float
    pose_keypoints: Optional[Dict[str, Tuple[float, float, float]]] = None
    velocity: Optional[Tuple[float, float]] = None
    is_on_ground: bool = False


@dataclass
class VisualAnalysisResult:
    """Complete result from visual analysis"""
    timestamp: datetime
    frame_id: int
    persons: List[DetectedPerson]
    scene_action: ActionType
    scene_severity: SeverityLevel
    fight_detected: bool = False
    fall_detected: bool = False
    crowd_density: float = 0.0
    inference_time_ms: float = 0.0
    raw_responses: Dict = field(default_factory=dict)


class NVIDIANIMVisualInference:
    """
    Visual inference using NVIDIA NIM hosted models.

    Models used:
    - microsoft/florence-2: Vision-language model for scene understanding and action detection
    - nvidia/grounding-dino: Object detection with natural language grounding
    - meta/sam2-hiera-large: Segment Anything for precise person segmentation
    """

    # NVIDIA NIM Model Endpoints
    MODELS = {
        "scene_understanding": "microsoft/florence-2",
        "object_detection": "nvidia/grounding-dino",
        "segmentation": "meta/sam2-hiera-large",
        "action_recognition": "microsoft/florence-2",  # Uses VQA capability
    }

    # Action detection prompts for Florence-2
    ACTION_PROMPTS = {
        "detect_actions": "Describe what each person in this image is doing. List actions: standing, walking, running, fighting, falling, lying down.",
        "detect_fight": "Is there a fight or physical altercation happening in this image? Answer yes or no and describe.",
        "detect_fall": "Has anyone fallen or is lying on the ground in this image? Answer yes or no and describe.",
        "detect_emergency": "Is there a medical emergency, injury, or dangerous situation in this image? Describe any concerns.",
    }

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with NVIDIA API key"""
        self.api_key = api_key or NVIDIA_API_KEY
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY environment variable required")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        # Person tracking state
        self.tracked_persons: Dict[int, List[DetectedPerson]] = {}
        self.next_person_id = 0
        self.frame_count = 0

    def _encode_image(self, image: np.ndarray) -> str:
        """Encode image to base64 for API"""
        import cv2
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')

    def _call_nim_api(self, model: str, payload: Dict, timeout: int = 30) -> Dict:
        """Call NVIDIA NIM API endpoint"""
        url = f"{NVIDIA_API_BASE}/vlm/{model}"

        try:
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"NIM API error for {model}: {e}")
            return {"error": str(e)}

    def detect_persons_grounding_dino(self, image: np.ndarray) -> List[Dict]:
        """
        Detect persons using NVIDIA Grounding DINO.

        Model: nvidia/grounding-dino
        - Zero-shot object detection with text prompts
        - Returns bounding boxes for detected persons
        """
        image_b64 = self._encode_image(image)

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "person. human. people."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                    ]
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.1,
            "top_p": 0.9
        }

        result = self._call_nim_api("nvidia/grounding-dino", payload)

        # Parse detection results
        detections = []
        if "choices" in result:
            # Parse bounding boxes from response
            response_text = result["choices"][0].get("message", {}).get("content", "")
            detections = self._parse_detection_response(response_text)

        return detections

    def analyze_actions_florence(self, image: np.ndarray) -> Dict:
        """
        Analyze human actions using Microsoft Florence-2.

        Model: microsoft/florence-2
        - Vision-language model with strong action understanding
        - Can describe what people are doing in detail
        """
        image_b64 = self._encode_image(image)

        results = {}

        for prompt_key, prompt_text in self.ACTION_PROMPTS.items():
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                        ]
                    }
                ],
                "max_tokens": 512,
                "temperature": 0.2,
                "top_p": 0.9
            }

            response = self._call_nim_api("microsoft/florence-2", payload)

            if "choices" in response:
                results[prompt_key] = response["choices"][0].get("message", {}).get("content", "")

        return results

    def _parse_detection_response(self, response: str) -> List[Dict]:
        """Parse detection response into structured format"""
        detections = []
        # Response parsing logic - Florence/DINO return structured coordinates
        # Format varies by model, this handles common patterns
        try:
            if "[" in response and "]" in response:
                # Try to extract JSON-like bbox data
                import re
                bbox_pattern = r'\[(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\]'
                matches = re.findall(bbox_pattern, response)
                for i, match in enumerate(matches):
                    detections.append({
                        "id": i,
                        "bbox": [float(x) for x in match],
                        "confidence": 0.85,
                        "label": "person"
                    })
        except Exception:
            pass
        return detections

    def _classify_action_from_text(self, action_text: str) -> Tuple[ActionType, float]:
        """Classify action type from Florence-2 text response"""
        text_lower = action_text.lower()

        # Priority order for classification
        action_keywords = {
            ActionType.FIGHTING: ["fight", "fighting", "punch", "hit", "attack", "altercation", "violent", "assault"],
            ActionType.FALLING: ["fall", "falling", "fell", "collapse", "collapsing", "trip", "tumble"],
            ActionType.LYING_DOWN: ["lying", "ground", "floor", "prone", "collapsed", "unconscious"],
            ActionType.RUNNING: ["run", "running", "sprint", "rushing", "fleeing", "chasing"],
            ActionType.WALKING: ["walk", "walking", "stroll", "moving", "pace"],
            ActionType.STANDING: ["stand", "standing", "still", "stationary", "waiting"],
        }

        for action, keywords in action_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Higher confidence for more specific matches
                    confidence = 0.9 if len(keyword) > 4 else 0.75
                    return action, confidence

        return ActionType.UNKNOWN, 0.5

    def _determine_scene_severity(self, persons: List[DetectedPerson],
                                   action_analysis: Dict) -> Tuple[SeverityLevel, bool, bool]:
        """Determine overall scene severity"""
        fight_detected = False
        fall_detected = False

        # Check action analysis for fights/falls
        fight_response = action_analysis.get("detect_fight", "").lower()
        fall_response = action_analysis.get("detect_fall", "").lower()
        emergency_response = action_analysis.get("detect_emergency", "").lower()

        if "yes" in fight_response or "fight" in fight_response:
            fight_detected = True

        if "yes" in fall_response or "fall" in fall_response or "lying" in fall_response:
            fall_detected = True

        # Check individual person actions
        for person in persons:
            if person.action == ActionType.FIGHTING:
                fight_detected = True
            if person.action in [ActionType.FALLING, ActionType.LYING_DOWN]:
                fall_detected = True

        # Determine severity
        if fight_detected and fall_detected:
            severity = SeverityLevel.CRITICAL
        elif fight_detected:
            severity = SeverityLevel.HIGH
        elif fall_detected:
            severity = SeverityLevel.HIGH
        elif any("emergency" in emergency_response or "injury" in emergency_response
                 or "danger" in emergency_response for _ in [1]):
            severity = SeverityLevel.MEDIUM
        elif len(persons) > 5:  # Crowd detected
            severity = SeverityLevel.LOW
        else:
            severity = SeverityLevel.NONE

        return severity, fight_detected, fall_detected

    def analyze_frame(self, image: np.ndarray) -> VisualAnalysisResult:
        """
        Complete visual analysis of a single frame.

        Uses:
        1. Grounding DINO for person detection
        2. Florence-2 for action recognition
        3. Custom logic for severity assessment
        """
        start_time = time.time()
        self.frame_count += 1

        # Step 1: Detect persons
        detections = self.detect_persons_grounding_dino(image)

        # Step 2: Analyze actions
        action_analysis = self.analyze_actions_florence(image)

        # Step 3: Build person list with actions
        persons = []
        actions_text = action_analysis.get("detect_actions", "")
        primary_action, primary_confidence = self._classify_action_from_text(actions_text)

        for det in detections:
            person = DetectedPerson(
                person_id=self.next_person_id,
                bbox=tuple(det["bbox"]),
                confidence=det["confidence"],
                action=primary_action,
                action_confidence=primary_confidence,
                is_on_ground=primary_action in [ActionType.FALLING, ActionType.LYING_DOWN]
            )
            persons.append(person)
            self.next_person_id += 1

        # Step 4: Determine severity
        severity, fight_detected, fall_detected = self._determine_scene_severity(
            persons, action_analysis
        )

        # Calculate crowd density (persons per frame area)
        h, w = image.shape[:2]
        crowd_density = len(persons) / (w * h / 1000000)  # Persons per megapixel

        inference_time = (time.time() - start_time) * 1000

        return VisualAnalysisResult(
            timestamp=datetime.now(),
            frame_id=self.frame_count,
            persons=persons,
            scene_action=primary_action,
            scene_severity=severity,
            fight_detected=fight_detected,
            fall_detected=fall_detected,
            crowd_density=crowd_density,
            inference_time_ms=inference_time,
            raw_responses=action_analysis
        )

    def analyze_video_stream(self, frame_generator, callback=None, skip_frames: int = 5):
        """
        Analyze video stream with frame skipping for efficiency.

        Args:
            frame_generator: Iterator yielding (frame_id, frame) tuples
            callback: Optional function called with each VisualAnalysisResult
            skip_frames: Process every Nth frame (default: 5)
        """
        for frame_id, frame in frame_generator:
            if frame_id % skip_frames != 0:
                continue

            result = self.analyze_frame(frame)

            if callback:
                callback(result)

            # Alert on high severity
            if result.scene_severity.value >= SeverityLevel.HIGH.value:
                print(f"[ALERT] Frame {frame_id}: {result.scene_severity.name}")
                if result.fight_detected:
                    print(f"  - FIGHT DETECTED")
                if result.fall_detected:
                    print(f"  - FALL DETECTED")

            yield result


class NVIDIANIMPoseEstimation:
    """
    Pose estimation for detailed movement analysis.

    Model: nvidia/bodypose-estimation (NIM)
    - Detects 17 body keypoints per person
    - Used for precise fall detection and action classification
    """

    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or NVIDIA_API_KEY
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def estimate_pose(self, image: np.ndarray) -> List[Dict]:
        """
        Estimate poses for all persons in image.

        Returns list of pose dicts with keypoints and confidence scores.
        """
        import cv2
        _, buffer = cv2.imencode('.jpg', image)
        image_b64 = base64.b64encode(buffer).decode('utf-8')

        payload = {
            "input": [{"type": "image_url", "url": f"data:image/jpeg;base64,{image_b64}"}],
            "parameters": {"max_persons": 10}
        }

        try:
            response = requests.post(
                f"{NVIDIA_API_BASE}/cv/nvidia/bodypose-estimation",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("poses", [])
        except Exception as e:
            print(f"Pose estimation error: {e}")
            return []

    def is_person_fallen(self, keypoints: Dict) -> Tuple[bool, float]:
        """
        Determine if person has fallen based on keypoint positions.

        Checks:
        - Hip-to-shoulder angle (horizontal = fallen)
        - Head below hip level
        - Overall body orientation
        """
        try:
            # Get key points
            left_shoulder = keypoints.get("left_shoulder", [0, 0, 0])
            right_shoulder = keypoints.get("right_shoulder", [0, 0, 0])
            left_hip = keypoints.get("left_hip", [0, 0, 0])
            right_hip = keypoints.get("right_hip", [0, 0, 0])
            nose = keypoints.get("nose", [0, 0, 0])

            # Calculate body angle
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_center_y = (left_hip[1] + right_hip[1]) / 2

            # Body is horizontal if shoulders and hips at similar Y
            vertical_diff = abs(shoulder_center_y - hip_center_y)

            # Check if head is below hips (inverted)
            head_below_hips = nose[1] > hip_center_y

            # Calculate confidence
            if vertical_diff < 50 and head_below_hips:
                return True, 0.9
            elif vertical_diff < 100:
                return True, 0.7
            elif head_below_hips:
                return True, 0.6

            return False, 0.1

        except Exception:
            return False, 0.0

    def classify_movement(self, current_keypoints: Dict,
                          previous_keypoints: Optional[Dict] = None) -> Tuple[ActionType, float]:
        """
        Classify movement type based on keypoint changes.

        Standing: Minimal movement, vertical posture
        Walking: Moderate leg movement, steady pace
        Running: High leg movement, rapid changes
        """
        if not previous_keypoints:
            # Can't determine movement without previous frame
            return ActionType.STANDING, 0.5

        try:
            # Calculate movement magnitude
            total_movement = 0
            leg_movement = 0
            arm_movement = 0

            for kp_name in self.KEYPOINT_NAMES:
                curr = current_keypoints.get(kp_name, [0, 0, 0])
                prev = previous_keypoints.get(kp_name, [0, 0, 0])

                movement = np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
                total_movement += movement

                if "knee" in kp_name or "ankle" in kp_name or "hip" in kp_name:
                    leg_movement += movement
                if "elbow" in kp_name or "wrist" in kp_name or "shoulder" in kp_name:
                    arm_movement += movement

            # Classify based on movement thresholds
            if total_movement < 20:
                return ActionType.STANDING, 0.85
            elif total_movement < 80 and leg_movement < 40:
                return ActionType.WALKING, 0.75
            elif total_movement >= 80 or leg_movement >= 40:
                return ActionType.RUNNING, 0.80

            return ActionType.UNKNOWN, 0.5

        except Exception:
            return ActionType.UNKNOWN, 0.3


# Convenience function for simple usage
def create_visual_analyzer(api_key: Optional[str] = None) -> NVIDIANIMVisualInference:
    """Create configured visual analyzer instance"""
    return NVIDIANIMVisualInference(api_key)


if __name__ == "__main__":
    # Test with sample image
    import cv2

    print("NVIDIA NIM Visual Inference Module")
    print("=" * 50)
    print(f"Models configured:")
    for name, model in NVIDIANIMVisualInference.MODELS.items():
        print(f"  {name}: {model}")
    print()

    if not NVIDIA_API_KEY:
        print("Set NVIDIA_API_KEY environment variable to test")
    else:
        print("API key configured - ready for inference")
