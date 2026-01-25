#!/usr/bin/env python3
"""
Visual Event Detector - NVIDIA RAPIDS cuML Integration

Real-time video analysis for emergency event detection:
- Fight/Violence Detection
- Fall/Medical Emergency Detection
- Crowd Disturbance Detection
- Suspicious Activity Detection

Uses NVIDIA RAPIDS cuML for GPU-accelerated inference
"""

import os
import cv2
import numpy as np
import cupy as cp
import cudf
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# RAPIDS cuML imports
from cuml.ensemble import RandomForestClassifier
from cuml.svm import SVC
from cuml.neighbors import KNeighborsClassifier
from cuml.preprocessing import StandardScaler

# Optional: Deep learning for feature extraction
try:
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - using traditional CV features")


# ============================================================================
# EVENT TYPES & CONFIGURATION
# ============================================================================

class EventType(Enum):
    """Types of visual events to detect"""
    FIGHT = "fight"
    FALL = "fall"
    MEDICAL_EMERGENCY = "medical_emergency"
    CROWD_DISTURBANCE = "crowd_disturbance"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    WEAPON_DETECTED = "weapon_detected"
    FIRE_SMOKE = "fire_smoke"
    NORMAL = "normal"


@dataclass
class DetectionResult:
    """Result from visual event detection"""
    event_type: EventType
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    frame_number: int
    bounding_boxes: List[List[int]]  # [[x1,y1,x2,y2], ...]
    num_people: int
    motion_intensity: float
    alert_level: str  # "CRITICAL", "HIGH", "MODERATE", "LOW"
    metadata: Dict


# Model confidence thresholds
CONFIDENCE_THRESHOLDS = {
    EventType.FIGHT: 0.75,
    EventType.FALL: 0.80,
    EventType.MEDICAL_EMERGENCY: 0.70,
    EventType.CROWD_DISTURBANCE: 0.65,
    EventType.SUSPICIOUS_ACTIVITY: 0.60,
    EventType.WEAPON_DETECTED: 0.85,
    EventType.FIRE_SMOKE: 0.80,
}


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

class VisualFeatureExtractor:
    """Extract features from video frames for cuML models"""

    def __init__(self, use_deep_features: bool = True):
        self.use_deep_features = use_deep_features and TORCH_AVAILABLE

        if self.use_deep_features:
            # Use pretrained ResNet for feature extraction
            self.resnet = models.resnet50(pretrained=True)
            self.resnet.eval()
            self.resnet = self.resnet.cuda() if torch.cuda.is_available() else self.resnet

            # Remove final classification layer
            self.feature_extractor = torch.nn.Sequential(
                *list(self.resnet.children())[:-1]
            )

            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

    def extract_traditional_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract traditional computer vision features

        Features:
        1. Motion intensity (optical flow magnitude)
        2. Edge density (Canny edges)
        3. Color histograms (HSV)
        4. Texture features (LBP)
        5. Spatial moments
        6. Contour statistics
        """
        features = []

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Edge density (violence often has sharp edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)

        # 2. Color histogram (HSV space)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])

        # Normalize and flatten
        h_hist = h_hist.flatten() / (h_hist.sum() + 1e-7)
        s_hist = s_hist.flatten() / (s_hist.sum() + 1e-7)
        v_hist = v_hist.flatten() / (v_hist.sum() + 1e-7)

        features.extend(h_hist[:16])  # Use first 16 bins
        features.extend(s_hist[:16])
        features.extend(v_hist[:16])

        # 3. Texture features (simplified LBP)
        # Compute local binary pattern approximation
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        texture_energy = np.sqrt(sobelx**2 + sobely**2).mean()
        features.append(texture_energy)

        # 4. Spatial moments (for shape analysis)
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        features.extend(hu_moments[:4])  # First 4 Hu moments

        # 5. Brightness and contrast
        brightness = gray.mean()
        contrast = gray.std()
        features.extend([brightness, contrast])

        # 6. Contour statistics (for detecting people/objects)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = len(contours)
        avg_contour_area = np.mean([cv2.contourArea(c) for c in contours]) if contours else 0
        features.extend([num_contours, avg_contour_area])

        return np.array(features, dtype=np.float32)

    def extract_motion_features(self, prev_frame: np.ndarray,
                                curr_frame: np.ndarray) -> np.ndarray:
        """
        Extract motion-based features using optical flow

        Critical for detecting:
        - Fights (rapid, chaotic motion)
        - Falls (sudden downward motion)
        - Running/fleeing (directional motion)
        """
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # Extract motion statistics
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        features = [
            magnitude.mean(),          # Average motion intensity
            magnitude.std(),           # Motion variance (chaotic vs smooth)
            magnitude.max(),           # Peak motion (sudden movements)
            (magnitude > 5).sum() / magnitude.size,  # Proportion of high motion

            # Directional motion statistics
            np.cos(angle).mean(),      # Horizontal motion bias
            np.sin(angle).mean(),      # Vertical motion bias

            # Spatial motion distribution
            magnitude[:len(magnitude)//3].mean(),     # Top third (upper body)
            magnitude[len(magnitude)//3:2*len(magnitude)//3].mean(),  # Middle
            magnitude[2*len(magnitude)//3:].mean(),   # Bottom third (legs)
        ]

        return np.array(features, dtype=np.float32)

    def extract_deep_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract features using pretrained ResNet"""
        if not self.use_deep_features:
            return np.array([])

        # Preprocess frame
        input_tensor = self.transform(frame)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.cuda()

        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(input_batch)

        # Convert to numpy
        features = features.cpu().numpy().flatten()

        return features

    def extract_combined_features(self, curr_frame: np.ndarray,
                                  prev_frame: Optional[np.ndarray] = None) -> np.ndarray:
        """Extract all features and combine"""
        # Traditional CV features
        trad_features = self.extract_traditional_features(curr_frame)

        # Motion features (if previous frame available)
        if prev_frame is not None:
            motion_features = self.extract_motion_features(prev_frame, curr_frame)
        else:
            motion_features = np.zeros(9, dtype=np.float32)

        # Deep features (if enabled)
        if self.use_deep_features:
            deep_features = self.extract_deep_features(curr_frame)
            # Reduce dimensionality via PCA or take subset
            deep_features = deep_features[:100]  # Use first 100 features
        else:
            deep_features = np.array([])

        # Combine all features
        combined = np.concatenate([trad_features, motion_features, deep_features])

        return combined


# ============================================================================
# cuML EVENT DETECTION MODELS
# ============================================================================

class FightDetector:
    """
    Detect physical altercations using cuML Random Forest

    Key indicators:
    - High motion intensity
    - Chaotic, non-directional movement
    - Multiple people in close proximity
    - Rapid gestures (punching, grappling)
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            max_features='sqrt',
            n_bins=128,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, X_train: cp.ndarray, y_train: cp.ndarray):
        """Train fight detection model"""
        # Normalize features
        X_scaled = self.scaler.fit_transform(X_train)

        # Train model
        self.model.fit(X_scaled, y_train)
        self.is_trained = True

    def predict(self, features: np.ndarray) -> Tuple[bool, float]:
        """
        Predict if fight is occurring

        Returns:
            (is_fight, confidence)
        """
        if not self.is_trained:
            # Use heuristic if not trained
            return self._heuristic_predict(features)

        # Convert to cupy
        features_gpu = cp.array(features.reshape(1, -1), dtype=cp.float32)

        # Scale features
        features_scaled = self.scaler.transform(features_gpu)

        # Get prediction probability
        proba = self.model.predict_proba(features_scaled)
        confidence = float(proba[0][1])  # Probability of fight class

        is_fight = confidence >= CONFIDENCE_THRESHOLDS[EventType.FIGHT]

        return is_fight, confidence

    def _heuristic_predict(self, features: np.ndarray) -> Tuple[bool, float]:
        """Heuristic-based prediction if model not trained"""
        # Features: [edge_density, ..., motion_mean, motion_std, motion_max, ...]
        # Typical indices (adjust based on actual feature order)

        # Extract key features for heuristic
        motion_mean = features[48] if len(features) > 48 else 0  # Motion intensity
        motion_std = features[49] if len(features) > 49 else 0   # Motion chaos
        motion_max = features[50] if len(features) > 50 else 0   # Peak motion

        # Fight characteristics: high motion, high variance, high peaks
        fight_score = (
            (motion_mean > 8.0) * 0.4 +
            (motion_std > 5.0) * 0.3 +
            (motion_max > 15.0) * 0.3
        )

        is_fight = fight_score > 0.6
        confidence = min(fight_score, 1.0)

        return is_fight, confidence


class FallDetector:
    """
    Detect person falling using cuML SVM

    Key indicators:
    - Sudden downward motion
    - Change from vertical to horizontal orientation
    - Rapid deceleration at ground level
    - No recovery motion
    """

    def __init__(self):
        self.model = SVC(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            probability=True
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, X_train: cp.ndarray, y_train: cp.ndarray):
        """Train fall detection model"""
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_trained = True

    def predict(self, features: np.ndarray) -> Tuple[bool, float]:
        """
        Predict if fall is occurring

        Returns:
            (is_fall, confidence)
        """
        if not self.is_trained:
            return self._heuristic_predict(features)

        features_gpu = cp.array(features.reshape(1, -1), dtype=cp.float32)
        features_scaled = self.scaler.transform(features_gpu)

        proba = self.model.predict_proba(features_scaled)
        confidence = float(proba[0][1])

        is_fall = confidence >= CONFIDENCE_THRESHOLDS[EventType.FALL]

        return is_fall, confidence

    def _heuristic_predict(self, features: np.ndarray) -> Tuple[bool, float]:
        """Heuristic-based fall detection"""
        # Fall indicators:
        # - Strong vertical motion (downward)
        # - Motion concentrated in bottom frame area
        # - Sudden motion followed by stillness

        vertical_motion = features[53] if len(features) > 53 else 0  # sin(angle) mean
        bottom_motion = features[56] if len(features) > 56 else 0    # Bottom third motion
        motion_max = features[50] if len(features) > 50 else 0       # Peak motion

        # Fall score based on downward motion and bottom concentration
        fall_score = (
            (vertical_motion < -0.3) * 0.4 +  # Downward motion
            (bottom_motion > 10.0) * 0.3 +    # Motion at bottom
            (motion_max > 12.0) * 0.3         # Sudden movement
        )

        is_fall = fall_score > 0.65
        confidence = min(fall_score, 1.0)

        return is_fall, confidence


class MedicalEmergencyDetector:
    """
    Detect medical emergencies (collapse, seizure, distress)

    Key indicators:
    - Person on ground (not moving)
    - Jerky, irregular movements (seizure)
    - Crowd gathering around person
    - Unusual postures
    """

    def __init__(self):
        self.model = KNeighborsClassifier(
            n_neighbors=5,
            metric='euclidean',
            algorithm='brute'
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, X_train: cp.ndarray, y_train: cp.ndarray):
        """Train medical emergency detection model"""
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_trained = True

    def predict(self, features: np.ndarray) -> Tuple[bool, float]:
        """Predict if medical emergency is occurring"""
        if not self.is_trained:
            return self._heuristic_predict(features)

        features_gpu = cp.array(features.reshape(1, -1), dtype=cp.float32)
        features_scaled = self.scaler.transform(features_gpu)

        # KNN doesn't have predict_proba in cuML, use decision function
        prediction = self.model.predict(features_scaled)

        # Use distance to neighbors as confidence proxy
        distances, _ = self.model.kneighbors(features_scaled, n_neighbors=5)
        avg_distance = float(cp.mean(distances))

        # Inverse distance as confidence (closer = more confident)
        confidence = 1.0 / (1.0 + avg_distance)

        is_emergency = bool(prediction[0] == 1)

        return is_emergency, confidence

    def _heuristic_predict(self, features: np.ndarray) -> Tuple[bool, float]:
        """Heuristic-based medical emergency detection"""
        # Medical emergency indicators:
        # - Low motion (person down)
        # - OR erratic motion (seizure)
        # - Unusual spatial distribution

        motion_mean = features[48] if len(features) > 48 else 0
        motion_std = features[49] if len(features) > 49 else 0
        bottom_motion = features[56] if len(features) > 56 else 0

        # Two scenarios: collapse (low motion) or seizure (erratic)
        collapse_score = (
            (motion_mean < 3.0) * 0.5 +      # Very low motion
            (bottom_motion > 5.0) * 0.5      # Activity at ground level
        )

        seizure_score = (
            (motion_std > 8.0) * 0.6 +       # Highly erratic
            (motion_mean > 5.0) * 0.4        # Some motion
        )

        emergency_score = max(collapse_score, seizure_score)
        is_emergency = emergency_score > 0.55
        confidence = min(emergency_score, 1.0)

        return is_emergency, confidence


# ============================================================================
# PERSON DETECTION (DEPRECATED - Use nvidia_nim_visual.py instead)
# ============================================================================

class PersonDetector:
    """
    Detect and track people in frame.

    DEPRECATED: This class uses local OpenCV DNN for person detection.
    For production use, migrate to nvidia_nim_visual.py which uses
    NVIDIA NIM hosted models (Grounding DINO, Florence-2, BodyPose).

    See: nvidia_nim_visual.py for cloud-based inference
    """

    def __init__(self, model_type: str = "opencv"):
        print("WARNING: PersonDetector is DEPRECATED. Use nvidia_nim_visual.py instead.")
        print("  - NVIDIA NIM provides better accuracy with cloud-hosted models")
        print("  - Models: nvidia/grounding-dino, microsoft/florence-2")
        self.model_type = model_type
        self._init_opencv_dnn()

    def _init_opencv_dnn(self):
        """Initialize OpenCV DNN with MobileNet SSD"""
        # Using MobileNet SSD for person detection
        # Download model files if needed
        prototxt = "models/MobileNetSSD_deploy.prototxt"
        caffemodel = "models/MobileNetSSD_deploy.caffemodel"

        # Check if models exist
        if os.path.exists(prototxt) and os.path.exists(caffemodel):
            self.net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            print("⚠️ Person detection models not found - using basic detection")
            self.net = None

    def detect(self, frame: np.ndarray) -> List[List[int]]:
        """
        Detect people in frame

        Returns:
            List of bounding boxes [[x1, y1, x2, y2], ...]
        """
        if self.net is None:
            return self._detect_basic(frame)

        # Prepare frame for detection
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            0.007843, (300, 300), 127.5
        )

        self.net.setInput(blob)
        detections = self.net.forward()

        # Parse detections
        h, w = frame.shape[:2]
        boxes = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                class_id = int(detections[0, 0, i, 1])

                # Class 15 is 'person' in COCO
                if class_id == 15:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    boxes.append(box.astype(int).tolist())

        return boxes

    def _detect_basic(self, frame: np.ndarray) -> List[List[int]]:
        """Basic person detection using contours"""
        # Very basic fallback - just detect large contours
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # Minimum size for a person
                x, y, w, h = cv2.boundingRect(contour)
                boxes.append([x, y, x+w, y+h])

        return boxes


# ============================================================================
# MAIN VISUAL EVENT DETECTOR
# ============================================================================

class VisualEventDetector:
    """Main orchestrator for visual event detection"""

    def __init__(self):
        print("🎥 Initializing Visual Event Detector...")

        # Feature extraction
        self.feature_extractor = VisualFeatureExtractor(use_deep_features=TORCH_AVAILABLE)
        print("✅ Feature extractor initialized")

        # Event detection models
        self.fight_detector = FightDetector()
        self.fall_detector = FallDetector()
        self.medical_detector = MedicalEmergencyDetector()
        print("✅ Event detection models initialized")

        # Person detection
        self.person_detector = PersonDetector()
        print("✅ Person detector initialized")

        # Frame buffer for temporal analysis
        self.prev_frame = None
        self.frame_count = 0

    def process_frame(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Process a single video frame

        Returns:
            List of detected events
        """
        self.frame_count += 1
        detections = []

        # Detect people in frame
        person_boxes = self.person_detector.detect(frame)
        num_people = len(person_boxes)

        # Extract features
        features = self.feature_extractor.extract_combined_features(
            frame, self.prev_frame
        )

        # Detect fights
        is_fight, fight_conf = self.fight_detector.predict(features)
        if is_fight:
            detections.append(DetectionResult(
                event_type=EventType.FIGHT,
                confidence=fight_conf,
                timestamp=datetime.now(),
                frame_number=self.frame_count,
                bounding_boxes=person_boxes,
                num_people=num_people,
                motion_intensity=float(features[48]) if len(features) > 48 else 0.0,
                alert_level="CRITICAL",
                metadata={"model": "fight_detector_rf"}
            ))

        # Detect falls
        is_fall, fall_conf = self.fall_detector.predict(features)
        if is_fall:
            detections.append(DetectionResult(
                event_type=EventType.FALL,
                confidence=fall_conf,
                timestamp=datetime.now(),
                frame_number=self.frame_count,
                bounding_boxes=person_boxes,
                num_people=num_people,
                motion_intensity=float(features[48]) if len(features) > 48 else 0.0,
                alert_level="CRITICAL",
                metadata={"model": "fall_detector_svm"}
            ))

        # Detect medical emergencies
        is_medical, medical_conf = self.medical_detector.predict(features)
        if is_medical:
            detections.append(DetectionResult(
                event_type=EventType.MEDICAL_EMERGENCY,
                confidence=medical_conf,
                timestamp=datetime.now(),
                frame_number=self.frame_count,
                bounding_boxes=person_boxes,
                num_people=num_people,
                motion_intensity=float(features[48]) if len(features) > 48 else 0.0,
                alert_level="HIGH",
                metadata={"model": "medical_detector_knn"}
            ))

        # Update previous frame
        self.prev_frame = frame.copy()

        return detections

    def process_video(self, video_path: str, output_path: Optional[str] = None,
                     display: bool = False) -> List[DetectionResult]:
        """
        Process entire video file

        Args:
            video_path: Path to video file
            output_path: Optional path to save annotated video
            display: Whether to display frames in real-time

        Returns:
            List of all detections
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"📹 Processing video: {video_path}")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Total frames: {total_frames}")

        # Setup video writer if saving output
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        all_detections = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            detections = self.process_frame(frame)
            all_detections.extend(detections)

            # Annotate frame
            annotated_frame = self._annotate_frame(frame, detections)

            # Write to output video
            if writer:
                writer.write(annotated_frame)

            # Display
            if display:
                cv2.imshow('Visual Event Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Progress
            if self.frame_count % 30 == 0:
                progress = (self.frame_count / total_frames) * 100
                print(f"   Progress: {progress:.1f}% ({self.frame_count}/{total_frames})")

        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        print(f"✅ Processing complete: {len(all_detections)} events detected")

        return all_detections

    def _annotate_frame(self, frame: np.ndarray,
                       detections: List[DetectionResult]) -> np.ndarray:
        """Draw detections on frame"""
        annotated = frame.copy()

        for detection in detections:
            # Draw bounding boxes
            for box in detection.bounding_boxes:
                x1, y1, x2, y2 = box

                # Color by event type
                if detection.event_type == EventType.FIGHT:
                    color = (0, 0, 255)  # Red
                elif detection.event_type == EventType.FALL:
                    color = (0, 165, 255)  # Orange
                elif detection.event_type == EventType.MEDICAL_EMERGENCY:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (255, 0, 0)  # Blue

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw event label
            label = f"{detection.event_type.value.upper()}: {detection.confidence:.2f}"
            cv2.putText(annotated, label, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Draw alert level
            alert_label = f"ALERT: {detection.alert_level}"
            cv2.putText(annotated, alert_label, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return annotated

    def save_detections(self, detections: List[DetectionResult], output_file: str):
        """Save detections to JSON file"""
        detections_dict = [
            {
                **asdict(d),
                'event_type': d.event_type.value,
                'timestamp': d.timestamp.isoformat()
            }
            for d in detections
        ]

        with open(output_file, 'w') as f:
            json.dump(detections_dict, f, indent=2)

        print(f"💾 Detections saved to: {output_file}")


# ============================================================================
# DEMO / TESTING
# ============================================================================

def demo_webcam():
    """Real-time demo using webcam"""
    detector = VisualEventDetector()

    cap = cv2.VideoCapture(0)  # Use webcam

    print("\n🎥 Starting webcam demo...")
    print("   Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        detections = detector.process_frame(frame)

        # Annotate
        annotated = detector._annotate_frame(frame, detections)

        # Display
        cv2.imshow('Visual Event Detection - Webcam', annotated)

        # Print detections
        for d in detections:
            print(f"🚨 {d.event_type.value.upper()} detected! "
                  f"Confidence: {d.confidence:.2f}, "
                  f"Alert: {d.alert_level}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def demo_video_file():
    """Demo with sample video file"""
    detector = VisualEventDetector()

    # Process video (replace with actual path)
    video_path = "sample_video.mp4"
    output_path = "output_annotated.mp4"

    if not os.path.exists(video_path):
        print(f"⚠️ Video not found: {video_path}")
        print("   Please provide a sample video file")
        return

    detections = detector.process_video(
        video_path,
        output_path=output_path,
        display=True
    )

    # Save detections
    detector.save_detections(detections, "detections.json")

    # Summary
    print(f"\n📊 DETECTION SUMMARY:")
    print(f"   Total events: {len(detections)}")

    event_counts = {}
    for d in detections:
        event_type = d.event_type.value
        event_counts[event_type] = event_counts.get(event_type, 0) + 1

    for event_type, count in event_counts.items():
        print(f"   {event_type.upper()}: {count}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("🎥 VISUAL EVENT DETECTOR")
    print("   NVIDIA RAPIDS cuML + Computer Vision")
    print("="*70)

    # Check RAPIDS availability
    try:
        import cuml
        print("✅ NVIDIA RAPIDS cuML detected")
    except ImportError:
        print("⚠️ RAPIDS cuML not found - using CPU fallback")

    # Check PyTorch availability
    if TORCH_AVAILABLE:
        print("✅ PyTorch detected - using deep features")
    else:
        print("⚠️ PyTorch not found - using traditional CV features")

    print("\nSelect demo mode:")
    print("1. Webcam (real-time)")
    print("2. Video file")
    print("3. Exit")

    choice = input("\nEnter choice (1-3): ")

    if choice == "1":
        demo_webcam()
    elif choice == "2":
        demo_video_file()
    else:
        print("Exiting...")


if __name__ == "__main__":
    main()
