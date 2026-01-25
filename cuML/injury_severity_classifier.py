#!/usr/bin/env python3
"""
Injury Severity Classifier - cuML GPU-Accelerated Model

Fine-tuned visual reasoning model for determining injury seriousness from
visual input and triggering predictive responses instead of reactive ones.

Uses NVIDIA RAPIDS cuML for GPU-accelerated inference:
- Gradient Boosting for severity classification
- Random Forest for injury type detection
- Time-series analysis for predictive response

Severity Levels:
- 0: MINOR (no immediate action needed)
- 1: MODERATE (monitor, possible intervention)
- 2: SEVERE (urgent response required)
- 3: CRITICAL (immediate life-saving intervention)
"""

import numpy as np
import cupy as cp
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

# RAPIDS cuML imports
from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.linear_model import LogisticRegression as cuLR
from cuml.preprocessing import StandardScaler as cuScaler
from cuml.cluster import DBSCAN as cuDBSCAN
from cuml.neighbors import KNeighborsClassifier as cuKNN

# For gradient boosting (XGBoost with GPU)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available - using Random Forest fallback")


# ============================================================================
# SEVERITY LEVELS & CONFIGURATION
# ============================================================================

class SeverityLevel(IntEnum):
    """Injury severity classification levels"""
    MINOR = 0       # Scrapes, minor bruises - no immediate action
    MODERATE = 1    # Sprains, cuts - monitor, possible medical attention
    SEVERE = 2      # Fractures, head injuries - urgent medical response
    CRITICAL = 3    # Life-threatening - immediate intervention


@dataclass
class InjuryAssessment:
    """Complete injury severity assessment result"""
    severity_level: SeverityLevel
    severity_score: float  # 0.0 to 1.0 continuous score
    confidence: float
    injury_type: str
    body_region: str
    timestamp: datetime
    visual_indicators: List[str]
    recommended_response: str
    response_urgency_seconds: int  # How quickly response is needed
    predicted_deterioration_risk: float  # Risk of condition worsening
    resource_requirements: List[str]
    metadata: Dict = field(default_factory=dict)


@dataclass
class PredictiveAlert:
    """Predictive alert for anticipated emergency"""
    alert_id: str
    prediction_confidence: float
    predicted_event_type: str
    predicted_severity: SeverityLevel
    predicted_time_window: Tuple[datetime, datetime]
    location_cluster: str
    risk_factors: List[str]
    recommended_preemptive_action: str
    resource_pre_positioning: Dict[str, int]


# Response time requirements by severity (seconds)
RESPONSE_TIME_REQUIREMENTS = {
    SeverityLevel.MINOR: 3600,      # 1 hour
    SeverityLevel.MODERATE: 900,    # 15 minutes
    SeverityLevel.SEVERE: 300,      # 5 minutes
    SeverityLevel.CRITICAL: 60,     # 1 minute
}

# Feature indices for injury-specific analysis
INJURY_FEATURE_CONFIG = {
    'motion_stillness_threshold': 2.0,  # Below this = person not moving
    'fall_impact_threshold': 15.0,      # Motion spike indicating impact
    'bleeding_color_hue_range': (0, 10),  # Red hue in HSV
    'posture_change_threshold': 0.5,    # Significant posture change
}


# ============================================================================
# INJURY SEVERITY CLASSIFIER (cuML)
# ============================================================================

class InjurySeverityClassifier:
    """
    GPU-accelerated injury severity classification using cuML

    Multi-stage classification:
    1. Binary triage: Is this an injury? (yes/no)
    2. Severity classification: minor/moderate/severe/critical
    3. Injury type classification: fall, assault, medical, other
    4. Body region detection: head, torso, limbs
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path

        # Stage 1: Injury detection (binary)
        self.injury_detector = cuRF(
            n_estimators=100,
            max_depth=12,
            max_features='sqrt',
            n_bins=128,
            random_state=42
        )

        # Stage 2: Severity classifier (4-class)
        if XGBOOST_AVAILABLE:
            self.severity_classifier = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                tree_method='gpu_hist',  # GPU acceleration
                gpu_id=0,
                objective='multi:softprob',
                num_class=4,
                random_state=42
            )
        else:
            self.severity_classifier = cuRF(
                n_estimators=200,
                max_depth=14,
                max_features='sqrt',
                n_bins=128,
                random_state=42
            )

        # Stage 3: Injury type classifier
        self.injury_type_classifier = cuKNN(
            n_neighbors=7,
            metric='euclidean'
        )

        # Stage 4: Body region classifier
        self.body_region_classifier = cuLR(
            max_iter=1000,
            tol=1e-4
        )

        # Feature scalers
        self.scaler = cuScaler()

        # Training state
        self.is_trained = False
        self.injury_types = ['fall', 'assault', 'medical_collapse', 'accident', 'other']
        self.body_regions = ['head', 'torso', 'upper_limb', 'lower_limb', 'multiple']

        # Load pretrained if available
        if model_path and Path(model_path).exists():
            self.load_models(model_path)

    def extract_injury_features(self, visual_features: np.ndarray,
                                motion_history: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """
        Extract injury-specific features from visual input

        Args:
            visual_features: Base visual features from VisualFeatureExtractor
            motion_history: List of recent motion features for temporal analysis

        Returns:
            Enhanced feature vector for injury assessment
        """
        injury_features = []

        # Extract base features
        if len(visual_features) >= 57:
            edge_density = visual_features[0]
            color_hist = visual_features[1:49]  # H, S, V histograms
            texture_energy = visual_features[49] if len(visual_features) > 49 else 0
            hu_moments = visual_features[50:54] if len(visual_features) > 53 else [0, 0, 0, 0]
            # brightness and contrast available at indices 54, 55 if needed
        else:
            # Fallback for shorter feature vectors
            edge_density = visual_features[0] if len(visual_features) > 0 else 0
            color_hist = visual_features[1:min(49, len(visual_features))]
            texture_energy = 0
            hu_moments = [0, 0, 0, 0]

        # 1. Stillness indicator (person not moving = potential injury)
        motion_features_start = 57 if len(visual_features) > 57 else len(visual_features)
        if len(visual_features) > motion_features_start:
            motion_mean = visual_features[motion_features_start]
            motion_std = visual_features[motion_features_start + 1] if len(visual_features) > motion_features_start + 1 else 0
            # motion_max available at motion_features_start + 2 if needed for impact detection
        else:
            motion_mean, motion_std = 0, 0

        stillness_score = 1.0 / (1.0 + motion_mean)  # Higher when less motion
        injury_features.append(stillness_score)

        # 2. Impact detection (sudden high motion followed by stillness)
        impact_indicator = 0.0
        if motion_history and len(motion_history) >= 3:
            recent_motion = [np.mean(mf) for mf in motion_history[-5:]]
            if len(recent_motion) >= 3:
                # Look for spike followed by drop
                max_motion = max(recent_motion[:-1]) if len(recent_motion) > 1 else 0
                current_motion = recent_motion[-1]
                if max_motion > INJURY_FEATURE_CONFIG['fall_impact_threshold'] and current_motion < 5.0:
                    impact_indicator = min(max_motion / 20.0, 1.0)
        injury_features.append(impact_indicator)

        # 3. Red color presence (potential blood)
        red_presence = 0.0
        if len(color_hist) >= 16:
            # Hue histogram - check for red (hue 0-10 and 170-180 in OpenCV)
            hue_hist = color_hist[:16]
            red_presence = (hue_hist[0] + hue_hist[1]) / max(np.sum(hue_hist), 1e-7)
        injury_features.append(red_presence)

        # 4. Posture abnormality score
        posture_score = 0.0
        if len(hu_moments) >= 4:
            # Hu moments deviation from upright human posture
            # Lower moment values often indicate horizontal position
            posture_score = 1.0 - min(abs(hu_moments[0]) / 0.5, 1.0)
        injury_features.append(posture_score)

        # 5. Ground-level activity indicator
        ground_activity = 0.0
        if len(visual_features) > motion_features_start + 8:
            # Bottom third motion vs top third
            top_motion = visual_features[motion_features_start + 6] if len(visual_features) > motion_features_start + 6 else 0
            bottom_motion = visual_features[motion_features_start + 8] if len(visual_features) > motion_features_start + 8 else 0
            if top_motion + bottom_motion > 0:
                ground_activity = bottom_motion / (top_motion + bottom_motion + 1e-7)
        injury_features.append(ground_activity)

        # 6. Crowd gathering indicator (bystander response)
        crowd_gathering = 0.0
        if motion_history and len(motion_history) >= 5:
            # Increasing motion variance might indicate people gathering
            motion_variances = [np.std(mf) for mf in motion_history[-5:]]
            if len(motion_variances) >= 3:
                variance_trend = motion_variances[-1] - motion_variances[0]
                crowd_gathering = max(0, min(variance_trend / 5.0, 1.0))
        injury_features.append(crowd_gathering)

        # 7. Scene chaos score (multiple irregular motions)
        chaos_score = motion_std / (motion_mean + 1e-7) if motion_mean > 0 else 0
        chaos_score = min(chaos_score / 2.0, 1.0)
        injury_features.append(chaos_score)

        # 8. Duration of incident (based on motion history)
        incident_duration = 0.0
        if motion_history and len(motion_history) >= 2:
            # Count frames with abnormal motion
            abnormal_frames = sum(1 for mf in motion_history if np.mean(mf) > 10 or np.mean(mf) < 2)
            incident_duration = abnormal_frames / len(motion_history)
        injury_features.append(incident_duration)

        # 9. Recovery indicator (is person trying to get up?)
        recovery_indicator = 0.0
        if motion_history and len(motion_history) >= 3:
            recent_motion = [np.mean(mf) for mf in motion_history[-3:]]
            # Gradually increasing motion might indicate recovery attempt
            if all(recent_motion[i] <= recent_motion[i+1] for i in range(len(recent_motion)-1)):
                recovery_indicator = min(recent_motion[-1] / 10.0, 1.0)
        injury_features.append(recovery_indicator)

        # 10. Edge/shape complexity (distorted body position)
        shape_complexity = edge_density * texture_energy / 100.0
        injury_features.append(min(shape_complexity, 1.0))

        # Combine with original features
        combined = np.concatenate([
            np.array(injury_features, dtype=np.float32),
            visual_features.astype(np.float32)
        ])

        return combined

    def classify_severity(self, features: np.ndarray) -> Tuple[SeverityLevel, float, float]:
        """
        Classify injury severity

        Returns:
            (severity_level, severity_score, confidence)
        """
        if not self.is_trained:
            return self._heuristic_severity(features)

        # Scale features
        features_gpu = cp.array(features.reshape(1, -1), dtype=cp.float32)
        features_scaled = self.scaler.transform(features_gpu)

        # Stage 1: Check if this is an injury
        injury_prob = self.injury_detector.predict_proba(features_scaled)
        is_injury_prob = float(injury_prob[0][1])

        if is_injury_prob < 0.5:
            return SeverityLevel.MINOR, 0.1, 1.0 - is_injury_prob

        # Stage 2: Classify severity
        if XGBOOST_AVAILABLE:
            features_np = cp.asnumpy(features_scaled)
            severity_probs = self.severity_classifier.predict_proba(features_np)
        else:
            severity_probs = self.severity_classifier.predict_proba(features_scaled)
            severity_probs = cp.asnumpy(severity_probs)

        # Get predicted class and confidence
        predicted_class = int(np.argmax(severity_probs[0]))
        confidence = float(severity_probs[0][predicted_class])

        # Calculate continuous severity score (weighted average)
        severity_score = sum(
            prob * level for level, prob in enumerate(severity_probs[0])
        ) / 3.0  # Normalize to 0-1

        return SeverityLevel(predicted_class), severity_score, confidence

    def classify_injury_type(self, features: np.ndarray) -> Tuple[str, float]:
        """Classify the type of injury"""
        if not self.is_trained:
            return self._heuristic_injury_type(features)

        features_gpu = cp.array(features.reshape(1, -1), dtype=cp.float32)
        features_scaled = self.scaler.transform(features_gpu)

        prediction = self.injury_type_classifier.predict(features_scaled)
        predicted_idx = int(prediction[0])

        # Get distance-based confidence
        distances, _ = self.injury_type_classifier.kneighbors(features_scaled, n_neighbors=7)
        confidence = 1.0 / (1.0 + float(cp.mean(distances)))

        return self.injury_types[predicted_idx], confidence

    def classify_body_region(self, features: np.ndarray) -> Tuple[str, float]:
        """Classify affected body region"""
        if not self.is_trained:
            return self._heuristic_body_region(features)

        features_gpu = cp.array(features.reshape(1, -1), dtype=cp.float32)
        features_scaled = self.scaler.transform(features_gpu)

        prediction = self.body_region_classifier.predict(features_scaled)
        probs = self.body_region_classifier.predict_proba(features_scaled)

        predicted_idx = int(prediction[0])
        confidence = float(probs[0][predicted_idx])

        return self.body_regions[predicted_idx], confidence

    def assess_injury(self, visual_features: np.ndarray,
                     motion_history: Optional[List[np.ndarray]] = None) -> InjuryAssessment:
        """
        Complete injury assessment from visual input

        Args:
            visual_features: Features from VisualFeatureExtractor
            motion_history: Recent motion feature history

        Returns:
            Complete InjuryAssessment with all classification results
        """
        # Extract injury-specific features
        injury_features = self.extract_injury_features(visual_features, motion_history)

        # Classify severity
        severity_level, severity_score, severity_conf = self.classify_severity(injury_features)

        # Classify injury type
        injury_type, type_conf = self.classify_injury_type(injury_features)

        # Classify body region
        body_region, region_conf = self.classify_body_region(injury_features)

        # Determine visual indicators present
        visual_indicators = self._detect_visual_indicators(injury_features)

        # Calculate deterioration risk
        deterioration_risk = self._calculate_deterioration_risk(
            severity_level, injury_type, injury_features
        )

        # Determine recommended response
        response, urgency = self._determine_response(severity_level, injury_type, deterioration_risk)

        # Determine resource requirements
        resources = self._determine_resource_requirements(severity_level, injury_type, body_region)

        return InjuryAssessment(
            severity_level=severity_level,
            severity_score=severity_score,
            confidence=min(severity_conf, type_conf, region_conf),
            injury_type=injury_type,
            body_region=body_region,
            timestamp=datetime.now(),
            visual_indicators=visual_indicators,
            recommended_response=response,
            response_urgency_seconds=urgency,
            predicted_deterioration_risk=deterioration_risk,
            resource_requirements=resources,
            metadata={
                'severity_confidence': severity_conf,
                'type_confidence': type_conf,
                'region_confidence': region_conf,
                'feature_count': len(injury_features)
            }
        )

    def _heuristic_severity(self, features: np.ndarray) -> Tuple[SeverityLevel, float, float]:
        """Heuristic-based severity classification when model not trained"""
        # Extract key indicators from injury features
        if len(features) >= 10:
            stillness = features[0]
            impact = features[1]
            blood_presence = features[2]
            posture_abnormal = features[3]
            ground_activity = features[4]
            crowd_gathering = features[5]
            chaos = features[6]
            duration = features[7]
            recovery = features[8]
        else:
            # Default values
            stillness = impact = blood_presence = posture_abnormal = 0
            ground_activity = crowd_gathering = chaos = duration = recovery = 0

        # Calculate severity score
        critical_indicators = (
            (stillness > 0.8) * 0.3 +       # Not moving
            (impact > 0.7) * 0.25 +          # High impact detected
            (blood_presence > 0.3) * 0.2 +   # Blood visible
            (recovery < 0.1) * 0.25          # No recovery attempt
        )

        severe_indicators = (
            (posture_abnormal > 0.6) * 0.3 +
            (ground_activity > 0.7) * 0.3 +
            (crowd_gathering > 0.5) * 0.2 +
            (duration > 0.5) * 0.2
        )

        moderate_indicators = (
            (chaos > 0.5) * 0.4 +
            (impact > 0.3) * 0.3 +
            (recovery > 0.3) * 0.3
        )

        # Determine severity level
        if critical_indicators > 0.6:
            return SeverityLevel.CRITICAL, critical_indicators, 0.7
        elif severe_indicators > 0.5 or critical_indicators > 0.4:
            return SeverityLevel.SEVERE, max(severe_indicators, critical_indicators * 0.8), 0.65
        elif moderate_indicators > 0.4:
            return SeverityLevel.MODERATE, moderate_indicators, 0.6
        else:
            return SeverityLevel.MINOR, 0.2, 0.55

    def _heuristic_injury_type(self, features: np.ndarray) -> Tuple[str, float]:
        """Heuristic injury type classification"""
        if len(features) >= 10:
            impact = features[1]
            chaos = features[6]
            stillness = features[0]
        else:
            impact = chaos = stillness = 0

        if impact > 0.6 and stillness > 0.5:
            return 'fall', 0.6
        elif chaos > 0.6:
            return 'assault', 0.55
        elif stillness > 0.8:
            return 'medical_collapse', 0.6
        else:
            return 'other', 0.4

    def _heuristic_body_region(self, features: np.ndarray) -> Tuple[str, float]:
        """Heuristic body region classification"""
        if len(features) >= 10:
            ground_activity = features[4]
            posture = features[3]
        else:
            ground_activity = posture = 0

        # Ground activity often indicates lower body or full body
        if ground_activity > 0.7:
            return 'lower_limb', 0.5
        elif posture > 0.6:
            return 'torso', 0.5
        else:
            return 'multiple', 0.4

    def _detect_visual_indicators(self, features: np.ndarray) -> List[str]:
        """Detect which visual indicators are present"""
        indicators = []

        if len(features) >= 10:
            if features[0] > 0.7:
                indicators.append("person_not_moving")
            if features[1] > 0.5:
                indicators.append("impact_detected")
            if features[2] > 0.2:
                indicators.append("possible_blood")
            if features[3] > 0.5:
                indicators.append("abnormal_posture")
            if features[4] > 0.6:
                indicators.append("ground_level_position")
            if features[5] > 0.4:
                indicators.append("bystander_response")
            if features[6] > 0.5:
                indicators.append("chaotic_scene")
            if features[8] < 0.2 and features[0] > 0.5:
                indicators.append("no_recovery_attempt")

        return indicators

    def _calculate_deterioration_risk(self, severity: SeverityLevel,
                                      inj_type: str,
                                      features: np.ndarray) -> float:
        """Calculate risk of condition worsening"""
        base_risk = {
            SeverityLevel.MINOR: 0.1,
            SeverityLevel.MODERATE: 0.3,
            SeverityLevel.SEVERE: 0.6,
            SeverityLevel.CRITICAL: 0.9
        }[severity]

        # Adjust based on injury type
        type_multiplier = {
            'fall': 1.2,  # Head injuries can deteriorate
            'assault': 1.1,
            'medical_collapse': 1.4,  # Medical emergencies often worsen
            'accident': 1.0,
            'other': 1.0
        }.get(inj_type, 1.0)

        # Adjust based on features
        if len(features) >= 10:
            stillness = features[0]
            recovery = features[8]

            # High stillness + no recovery = higher risk
            if stillness > 0.8 and recovery < 0.2:
                type_multiplier *= 1.3

        return min(base_risk * type_multiplier, 1.0)

    def _determine_response(self, severity: SeverityLevel,
                           _injury_type: str,
                           deterioration_risk: float) -> Tuple[str, int]:
        """Determine recommended response and urgency"""
        base_urgency = RESPONSE_TIME_REQUIREMENTS[severity]

        # Adjust urgency based on deterioration risk
        # Note: _injury_type reserved for future injury-specific response customization
        if deterioration_risk > 0.7:
            base_urgency = int(base_urgency * 0.5)

        responses = {
            SeverityLevel.CRITICAL: (
                "IMMEDIATE 911 - Deploy closest emergency resources. "
                "Advise bystanders on emergency first aid. "
                "Pre-alert trauma center.",
                max(30, base_urgency)
            ),
            SeverityLevel.SEVERE: (
                "URGENT 911 - Dispatch EMS immediately. "
                "Request additional resources if available. "
                "Monitor for deterioration.",
                max(120, base_urgency)
            ),
            SeverityLevel.MODERATE: (
                "311 or Non-Emergency Medical - Dispatch medical response. "
                "Assess need for ambulance vs. clinic referral. "
                "Follow up within response window.",
                base_urgency
            ),
            SeverityLevel.MINOR: (
                "Monitor situation - No immediate action required. "
                "Log incident for review. "
                "Provide self-care guidance if contact available.",
                base_urgency
            )
        }

        return responses[severity]

    def _determine_resource_requirements(self, severity: SeverityLevel,
                                        injury_type: str,
                                        body_region: str) -> List[str]:
        """Determine required resources for response"""
        resources = []

        # Base resources by severity
        if severity >= SeverityLevel.CRITICAL:
            resources.extend([
                "ALS_ambulance",
                "paramedic_team",
                "trauma_alert"
            ])
        elif severity >= SeverityLevel.SEVERE:
            resources.extend([
                "BLS_ambulance",
                "EMT_team"
            ])
        elif severity >= SeverityLevel.MODERATE:
            resources.append("medical_response_unit")

        # Additional resources by injury type
        if injury_type == 'medical_collapse':
            resources.append("cardiac_monitor")
            resources.append("naloxone_kit")  # In case of overdose
        elif injury_type == 'assault':
            resources.append("police_unit")
        elif injury_type == 'fall' and body_region in ['head', 'multiple']:
            resources.append("spinal_immobilization")

        return list(set(resources))

    def train(self, X_train: np.ndarray, y_severity: np.ndarray,
             y_type: np.ndarray, y_region: np.ndarray):
        """
        Train all classification models

        Args:
            X_train: Training features
            y_severity: Severity labels (0-3)
            y_type: Injury type labels
            y_region: Body region labels
        """
        print("Training injury severity classifier...")

        # Convert to GPU arrays
        X_gpu = cp.array(X_train, dtype=cp.float32)
        y_severity_gpu = cp.array(y_severity, dtype=cp.int32)
        y_type_gpu = cp.array(y_type, dtype=cp.int32)
        y_region_gpu = cp.array(y_region, dtype=cp.int32)

        # Scale features
        X_scaled = self.scaler.fit_transform(X_gpu)

        # Train injury detector (binary: injury vs no injury)
        y_injury = (y_severity_gpu > 0).astype(cp.int32)
        self.injury_detector.fit(X_scaled, y_injury)
        print("  Injury detector trained")

        # Train severity classifier
        if XGBOOST_AVAILABLE:
            X_np = cp.asnumpy(X_scaled)
            y_np = cp.asnumpy(y_severity_gpu)
            self.severity_classifier.fit(X_np, y_np)
        else:
            self.severity_classifier.fit(X_scaled, y_severity_gpu)
        print("  Severity classifier trained")

        # Train injury type classifier
        self.injury_type_classifier.fit(X_scaled, y_type_gpu)
        print("  Injury type classifier trained")

        # Train body region classifier
        self.body_region_classifier.fit(X_scaled, y_region_gpu)
        print("  Body region classifier trained")

        self.is_trained = True
        print("Training complete!")

    def save_models(self, path: str):
        """Save trained models to disk"""
        Path(path).mkdir(parents=True, exist_ok=True)

        # Save cuML models (using pickle)
        with open(f"{path}/injury_detector.pkl", 'wb') as f:
            pickle.dump(self.injury_detector, f)

        with open(f"{path}/injury_type_classifier.pkl", 'wb') as f:
            pickle.dump(self.injury_type_classifier, f)

        with open(f"{path}/body_region_classifier.pkl", 'wb') as f:
            pickle.dump(self.body_region_classifier, f)

        with open(f"{path}/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)

        # Save XGBoost model separately
        if XGBOOST_AVAILABLE:
            self.severity_classifier.save_model(f"{path}/severity_classifier.json")
        else:
            with open(f"{path}/severity_classifier.pkl", 'wb') as f:
                pickle.dump(self.severity_classifier, f)

        # Save metadata
        metadata = {
            'injury_types': self.injury_types,
            'body_regions': self.body_regions,
            'is_trained': self.is_trained,
            'xgboost_available': XGBOOST_AVAILABLE
        }
        with open(f"{path}/metadata.json", 'w') as f:
            json.dump(metadata, f)

        print(f"Models saved to {path}")

    def load_models(self, path: str):
        """Load trained models from disk"""
        try:
            with open(f"{path}/injury_detector.pkl", 'rb') as f:
                self.injury_detector = pickle.load(f)

            with open(f"{path}/injury_type_classifier.pkl", 'rb') as f:
                self.injury_type_classifier = pickle.load(f)

            with open(f"{path}/body_region_classifier.pkl", 'rb') as f:
                self.body_region_classifier = pickle.load(f)

            with open(f"{path}/scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)

            # Load XGBoost or fallback
            if XGBOOST_AVAILABLE and Path(f"{path}/severity_classifier.json").exists():
                self.severity_classifier.load_model(f"{path}/severity_classifier.json")
            else:
                with open(f"{path}/severity_classifier.pkl", 'rb') as f:
                    self.severity_classifier = pickle.load(f)

            # Load metadata
            with open(f"{path}/metadata.json", 'r') as f:
                metadata = json.load(f)
                self.injury_types = metadata['injury_types']
                self.body_regions = metadata['body_regions']
                self.is_trained = metadata['is_trained']

            print(f"Models loaded from {path}")

        except Exception as e:
            print(f"Error loading models: {e}")
            self.is_trained = False


# ============================================================================
# PREDICTIVE RESPONSE ENGINE
# ============================================================================

class PredictiveResponseEngine:
    """
    GPU-accelerated predictive engine for anticipating emergencies

    Uses historical patterns to:
    1. Predict likely emergency locations/times
    2. Pre-position resources
    3. Alert before incidents escalate
    """

    def __init__(self):
        # Spatial clustering for hotspot detection
        self.spatial_clusterer = cuDBSCAN(
            eps=0.01,  # ~1km at SF latitude
            min_samples=5
        )

        # Temporal pattern model
        self.temporal_model = cuRF(
            n_estimators=50,
            max_depth=8,
            random_state=42
        )

        # Historical data storage
        self.incident_history: List[Dict] = []
        self.hotspots: List[Dict] = []
        self.is_trained = False

    def add_incident(self, timestamp: datetime, latitude: float, longitude: float,
                    severity: SeverityLevel, incident_type: str):
        """Add incident to history for pattern learning"""
        self.incident_history.append({
            'timestamp': timestamp,
            'latitude': latitude,
            'longitude': longitude,
            'severity': int(severity),
            'incident_type': incident_type,
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'month': timestamp.month
        })

    def identify_hotspots(self) -> List[Dict]:
        """Identify spatial hotspots from incident history"""
        if len(self.incident_history) < 10:
            return []

        # Extract coordinates
        coords = np.array([
            [inc['latitude'], inc['longitude']]
            for inc in self.incident_history
        ], dtype=np.float32)

        # GPU clustering
        coords_gpu = cp.array(coords)
        labels = self.spatial_clusterer.fit_predict(coords_gpu)
        labels_np = cp.asnumpy(labels)

        # Analyze clusters
        hotspots = []
        unique_labels = set(labels_np)

        for label in unique_labels:
            if label == -1:  # Noise
                continue

            mask = labels_np == label
            cluster_incidents = [
                inc for inc, m in zip(self.incident_history, mask) if m
            ]

            if len(cluster_incidents) >= 5:
                center_lat = np.mean([inc['latitude'] for inc in cluster_incidents])
                center_lon = np.mean([inc['longitude'] for inc in cluster_incidents])
                avg_severity = np.mean([inc['severity'] for inc in cluster_incidents])

                hotspots.append({
                    'cluster_id': int(label),
                    'center_latitude': float(center_lat),
                    'center_longitude': float(center_lon),
                    'incident_count': len(cluster_incidents),
                    'avg_severity': float(avg_severity),
                    'peak_hours': self._find_peak_hours(cluster_incidents),
                    'peak_days': self._find_peak_days(cluster_incidents)
                })

        self.hotspots = sorted(hotspots, key=lambda x: x['incident_count'], reverse=True)
        return self.hotspots

    def _find_peak_hours(self, incidents: List[Dict]) -> List[int]:
        """Find peak hours for incidents"""
        hours = [inc['hour'] for inc in incidents]
        hour_counts = {}
        for h in hours:
            hour_counts[h] = hour_counts.get(h, 0) + 1

        # Return top 3 hours
        sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
        return [h for h, _ in sorted_hours[:3]]

    def _find_peak_days(self, incidents: List[Dict]) -> List[int]:
        """Find peak days of week for incidents"""
        days = [inc['day_of_week'] for inc in incidents]
        day_counts = {}
        for d in days:
            day_counts[d] = day_counts.get(d, 0) + 1

        sorted_days = sorted(day_counts.items(), key=lambda x: x[1], reverse=True)
        return [d for d, _ in sorted_days[:3]]

    def predict_risk(self, latitude: float, longitude: float,
                    timestamp: datetime) -> Tuple[float, List[str]]:
        """
        Predict risk level for a location at a given time

        Returns:
            (risk_score 0-1, list of risk factors)
        """
        risk_factors = []
        risk_score = 0.1  # Base risk

        # Check proximity to hotspots
        for hotspot in self.hotspots:
            distance = self._haversine_distance(
                latitude, longitude,
                hotspot['center_latitude'], hotspot['center_longitude']
            )

            if distance < 0.5:  # Within 0.5 miles
                risk_score += 0.3 * (1 - distance / 0.5)
                risk_factors.append(f"near_hotspot_cluster_{hotspot['cluster_id']}")

                # Check if current hour is peak hour
                if timestamp.hour in hotspot['peak_hours']:
                    risk_score += 0.2
                    risk_factors.append("peak_hour")

                # Check if current day is peak day
                if timestamp.weekday() in hotspot['peak_days']:
                    risk_score += 0.1
                    risk_factors.append("peak_day")

        # Time-based risk adjustments
        if 22 <= timestamp.hour or timestamp.hour <= 4:
            risk_score += 0.1
            risk_factors.append("nighttime")

        if timestamp.weekday() in [4, 5]:  # Friday, Saturday
            risk_score += 0.05
            risk_factors.append("weekend")

        return min(risk_score, 1.0), risk_factors

    def generate_predictive_alerts(self, current_time: datetime,
                                   prediction_window_hours: int = 6) -> List[PredictiveAlert]:
        """
        Generate predictive alerts for upcoming high-risk periods

        Args:
            current_time: Current timestamp
            prediction_window_hours: How far ahead to predict

        Returns:
            List of predictive alerts
        """
        alerts = []

        # Check each hotspot for upcoming high-risk periods
        for hotspot in self.hotspots[:5]:  # Top 5 hotspots
            for hour_offset in range(prediction_window_hours):
                check_time = current_time + timedelta(hours=hour_offset)

                risk_score, risk_factors = self.predict_risk(
                    hotspot['center_latitude'],
                    hotspot['center_longitude'],
                    check_time
                )

                if risk_score > 0.6:  # High risk threshold
                    predicted_severity = (
                        SeverityLevel.CRITICAL if risk_score > 0.8 else
                        SeverityLevel.SEVERE if risk_score > 0.7 else
                        SeverityLevel.MODERATE
                    )

                    # Determine preemptive action
                    if predicted_severity >= SeverityLevel.SEVERE:
                        action = "Pre-position ambulance within 1 mile of hotspot"
                        resources = {
                            'ambulance': 1,
                            'naloxone_kits': 5,
                            'first_responders': 2
                        }
                    else:
                        action = "Alert nearby patrol units"
                        resources = {
                            'patrol_unit': 1,
                            'naloxone_kits': 2
                        }

                    alert = PredictiveAlert(
                        alert_id=f"PRED-{hotspot['cluster_id']}-{check_time.strftime('%Y%m%d%H')}",
                        prediction_confidence=risk_score,
                        predicted_event_type='medical_emergency',  # Most common
                        predicted_severity=predicted_severity,
                        predicted_time_window=(
                            check_time,
                            check_time + timedelta(hours=1)
                        ),
                        location_cluster=f"Cluster-{hotspot['cluster_id']}",
                        risk_factors=risk_factors,
                        recommended_preemptive_action=action,
                        resource_pre_positioning=resources
                    )
                    alerts.append(alert)

        # Deduplicate overlapping alerts
        return self._deduplicate_alerts(alerts)

    def _deduplicate_alerts(self, alerts: List[PredictiveAlert]) -> List[PredictiveAlert]:
        """Remove overlapping alerts, keeping highest confidence"""
        if not alerts:
            return []

        # Sort by confidence descending
        sorted_alerts = sorted(alerts, key=lambda x: x.prediction_confidence, reverse=True)

        final_alerts = []
        seen_clusters_hours = set()

        for alert in sorted_alerts:
            key = (alert.location_cluster, alert.predicted_time_window[0].hour)
            if key not in seen_clusters_hours:
                final_alerts.append(alert)
                seen_clusters_hours.add(key)

        return final_alerts

    def _haversine_distance(self, lat1: float, lon1: float,
                           lat2: float, lon2: float) -> float:
        """Calculate distance in miles between two coordinates"""
        R = 3959  # Earth radius in miles

        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)

        a = (np.sin(delta_lat/2)**2 +
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2)
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c


# ============================================================================
# MAIN / DEMO
# ============================================================================

def demo_injury_classification():
    """Demo injury severity classification"""
    print("\n" + "="*70)
    print("INJURY SEVERITY CLASSIFIER DEMO")
    print("="*70)

    classifier = InjurySeverityClassifier()

    # Simulate visual features for different scenarios
    scenarios = [
        {
            'name': 'Person collapsed, not moving',
            'features': np.array([
                0.9, 0.6, 0.1, 0.7, 0.8, 0.4, 0.3, 0.6, 0.1, 0.2
            ] + [0.5] * 50, dtype=np.float32)
        },
        {
            'name': 'Minor fall, person getting up',
            'features': np.array([
                0.3, 0.4, 0.0, 0.2, 0.3, 0.1, 0.2, 0.2, 0.7, 0.1
            ] + [0.5] * 50, dtype=np.float32)
        },
        {
            'name': 'Assault in progress',
            'features': np.array([
                0.2, 0.3, 0.15, 0.4, 0.2, 0.6, 0.8, 0.5, 0.4, 0.5
            ] + [0.5] * 50, dtype=np.float32)
        },
        {
            'name': 'Medical emergency - possible overdose',
            'features': np.array([
                0.95, 0.2, 0.05, 0.8, 0.9, 0.5, 0.1, 0.7, 0.05, 0.1
            ] + [0.5] * 50, dtype=np.float32)
        }
    ]

    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print("-" * 50)

        assessment = classifier.assess_injury(scenario['features'])

        print(f"Severity: {assessment.severity_level.name} "
              f"(score: {assessment.severity_score:.2f}, conf: {assessment.confidence:.2f})")
        print(f"Injury Type: {assessment.injury_type}")
        print(f"Body Region: {assessment.body_region}")
        print(f"Visual Indicators: {', '.join(assessment.visual_indicators)}")
        print(f"Deterioration Risk: {assessment.predicted_deterioration_risk:.2f}")
        print(f"Response Urgency: {assessment.response_urgency_seconds}s")
        print(f"Recommended: {assessment.recommended_response[:80]}...")
        print(f"Resources: {assessment.resource_requirements}")


def demo_predictive_engine():
    """Demo predictive response engine"""
    print("\n" + "="*70)
    print("PREDICTIVE RESPONSE ENGINE DEMO")
    print("="*70)

    engine = PredictiveResponseEngine()

    # Add simulated historical incidents
    base_time = datetime.now() - timedelta(days=30)

    # Tenderloin hotspot
    for i in range(25):
        engine.add_incident(
            timestamp=base_time + timedelta(days=i, hours=np.random.randint(18, 23)),
            latitude=37.7838 + np.random.normal(0, 0.005),
            longitude=-122.4167 + np.random.normal(0, 0.005),
            severity=SeverityLevel(np.random.choice([2, 3], p=[0.6, 0.4])),
            incident_type='medical_collapse'
        )

    # SOMA hotspot
    for i in range(15):
        engine.add_incident(
            timestamp=base_time + timedelta(days=i, hours=np.random.randint(20, 24)),
            latitude=37.7749 + np.random.normal(0, 0.003),
            longitude=-122.4194 + np.random.normal(0, 0.003),
            severity=SeverityLevel(np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])),
            incident_type='fall'
        )

    print(f"\nLoaded {len(engine.incident_history)} historical incidents")

    # Identify hotspots
    print("\nIdentifying hotspots...")
    hotspots = engine.identify_hotspots()

    for hs in hotspots:
        print(f"\n  Cluster {hs['cluster_id']}:")
        print(f"    Location: ({hs['center_latitude']:.4f}, {hs['center_longitude']:.4f})")
        print(f"    Incidents: {hs['incident_count']}")
        print(f"    Avg Severity: {hs['avg_severity']:.2f}")
        print(f"    Peak Hours: {hs['peak_hours']}")

    # Generate predictive alerts
    print("\nGenerating predictive alerts for next 6 hours...")
    alerts = engine.generate_predictive_alerts(datetime.now(), prediction_window_hours=6)

    for alert in alerts:
        print(f"\n  Alert: {alert.alert_id}")
        print(f"    Confidence: {alert.prediction_confidence:.2f}")
        print(f"    Predicted Severity: {alert.predicted_severity.name}")
        print(f"    Time Window: {alert.predicted_time_window[0].strftime('%H:%M')} - "
              f"{alert.predicted_time_window[1].strftime('%H:%M')}")
        print(f"    Risk Factors: {', '.join(alert.risk_factors)}")
        print(f"    Action: {alert.recommended_preemptive_action}")
        print(f"    Resources: {alert.resource_pre_positioning}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("cuML INJURY SEVERITY CLASSIFIER")
    print("NVIDIA RAPIDS GPU-Accelerated")
    print("="*70)

    demo_injury_classification()
    demo_predictive_engine()
