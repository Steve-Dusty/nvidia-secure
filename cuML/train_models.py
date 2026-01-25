#!/usr/bin/env python3
"""
Unified Training Script for cuML Emergency Detection Models

Trains all models:
1. Visual Event Detectors (Fight, Fall, Medical)
2. Injury Severity Classifier (4-class)
3. Injury Type & Body Region Classifiers

Usage:
    python train_models.py
    python train_models.py --samples 500 --output models/
"""

import argparse
import numpy as np
import cupy as cp
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# cuML imports
from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.svm import SVC as cuSVC
from cuml.neighbors import KNeighborsClassifier as cuKNN
from cuml.preprocessing import StandardScaler

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ============================================================================
# DATA GENERATORS
# ============================================================================

class VisualDataGenerator:
    """Generate synthetic data for visual event detection"""

    def __init__(self, seed=42):
        np.random.seed(seed)

    def generate_fight_data(self, n_samples):
        """Generate fight vs normal samples"""
        fight_features = []
        normal_features = []

        for _ in range(n_samples):
            # Fight: high motion, high variance, chaotic
            f = self._base_features()
            f[48:51] = [np.random.uniform(8, 20),   # motion_mean (high)
                        np.random.uniform(5, 12),   # motion_std (high)
                        np.random.uniform(15, 30)]  # motion_max (high)
            fight_features.append(f)

            # Normal: low-moderate motion, low variance
            n = self._base_features()
            n[48:51] = [np.random.uniform(1, 5),    # motion_mean (low)
                        np.random.uniform(0.5, 2),  # motion_std (low)
                        np.random.uniform(3, 10)]   # motion_max (low)
            normal_features.append(n)

        X = np.vstack([fight_features, normal_features])
        y = np.array([1]*n_samples + [0]*n_samples)
        return X.astype(np.float32), y.astype(np.int32)

    def generate_fall_data(self, n_samples):
        """Generate fall vs normal samples"""
        fall_features = []
        normal_features = []

        for _ in range(n_samples):
            # Fall: downward motion, bottom-heavy, sudden
            f = self._base_features()
            f[53] = np.random.uniform(-0.8, -0.3)   # vertical_motion (downward)
            f[56] = np.random.uniform(10, 25)       # bottom_motion (high)
            f[50] = np.random.uniform(12, 25)       # motion_max (sudden)
            fall_features.append(f)

            # Normal: balanced motion
            n = self._base_features()
            n[53] = np.random.uniform(-0.2, 0.2)
            n[56] = np.random.uniform(2, 8)
            n[50] = np.random.uniform(3, 10)
            normal_features.append(n)

        X = np.vstack([fall_features, normal_features])
        y = np.array([1]*n_samples + [0]*n_samples)
        return X.astype(np.float32), y.astype(np.int32)

    def generate_medical_data(self, n_samples):
        """Generate medical emergency vs normal samples"""
        medical_features = []
        normal_features = []

        for _ in range(n_samples):
            # Medical: very low motion (collapse) or erratic (seizure)
            m = self._base_features()
            if np.random.random() > 0.5:
                # Collapse scenario
                m[48:51] = [np.random.uniform(0, 2),
                            np.random.uniform(0, 1),
                            np.random.uniform(0, 3)]
            else:
                # Seizure scenario
                m[48:51] = [np.random.uniform(5, 10),
                            np.random.uniform(8, 15),
                            np.random.uniform(10, 20)]
            m[56] = np.random.uniform(5, 15)  # ground activity
            medical_features.append(m)

            # Normal: moderate motion
            n = self._base_features()
            n[48:51] = [np.random.uniform(3, 8),
                        np.random.uniform(1, 4),
                        np.random.uniform(5, 12)]
            normal_features.append(n)

        X = np.vstack([medical_features, normal_features])
        y = np.array([1]*n_samples + [0]*n_samples)
        return X.astype(np.float32), y.astype(np.int32)

    def _base_features(self):
        """Generate base 57 features + 9 motion = 66 total"""
        features = np.zeros(66)
        features[0] = np.random.uniform(0.1, 0.5)  # edge_density
        features[1:49] = np.random.uniform(0, 0.15, 48)  # color histograms
        features[49] = np.random.uniform(0.1, 0.8)  # texture
        features[50:54] = np.random.uniform(-0.3, 0.3, 4)  # hu moments
        features[54:56] = [np.random.uniform(50, 180),  # brightness
                          np.random.uniform(20, 80)]    # contrast
        features[56:58] = [np.random.uniform(5, 30),    # contours
                          np.random.uniform(100, 5000)] # contour area
        # Motion features at 48-56 (overlapping for simplicity)
        return features


class SeverityDataGenerator:
    """Generate synthetic data for injury severity classification"""

    def __init__(self, seed=42):
        np.random.seed(seed)

    def generate_dataset(self, n_per_class):
        """Generate severity dataset with all labels"""
        X_all, y_sev, y_type, y_region = [], [], [], []

        for severity in range(4):  # 0=MINOR, 1=MODERATE, 2=SEVERE, 3=CRITICAL
            for _ in range(n_per_class):
                features = self._generate_features(severity)
                X_all.append(features)
                y_sev.append(severity)
                y_type.append(np.random.randint(0, 5))
                y_region.append(np.random.randint(0, 5))

        X = np.array(X_all, dtype=np.float32)
        indices = np.random.permutation(len(X))

        return (X[indices],
                np.array(y_sev)[indices].astype(np.int32),
                np.array(y_type)[indices].astype(np.int32),
                np.array(y_region)[indices].astype(np.int32))

    def _generate_features(self, severity):
        """Generate 76 features (10 injury + 66 visual)"""
        # Injury-specific features (10)
        if severity == 3:  # CRITICAL
            injury = [np.random.uniform(0.8, 1.0),   # stillness
                      np.random.uniform(0.5, 0.9),   # impact
                      np.random.uniform(0.0, 0.4),   # blood
                      np.random.uniform(0.7, 1.0),   # posture
                      np.random.uniform(0.7, 1.0),   # ground
                      np.random.uniform(0.3, 0.7),   # crowd
                      np.random.uniform(0.1, 0.4),   # chaos
                      np.random.uniform(0.5, 1.0),   # duration
                      np.random.uniform(0.0, 0.15),  # recovery
                      np.random.uniform(0.3, 0.6)]   # complexity
        elif severity == 2:  # SEVERE
            injury = [np.random.uniform(0.5, 0.85),
                      np.random.uniform(0.4, 0.8),
                      np.random.uniform(0.1, 0.35),
                      np.random.uniform(0.5, 0.8),
                      np.random.uniform(0.5, 0.85),
                      np.random.uniform(0.3, 0.6),
                      np.random.uniform(0.2, 0.5),
                      np.random.uniform(0.4, 0.8),
                      np.random.uniform(0.1, 0.35),
                      np.random.uniform(0.3, 0.6)]
        elif severity == 1:  # MODERATE
            injury = [np.random.uniform(0.25, 0.6),
                      np.random.uniform(0.2, 0.6),
                      np.random.uniform(0.0, 0.2),
                      np.random.uniform(0.3, 0.6),
                      np.random.uniform(0.3, 0.7),
                      np.random.uniform(0.2, 0.5),
                      np.random.uniform(0.3, 0.6),
                      np.random.uniform(0.2, 0.5),
                      np.random.uniform(0.3, 0.6),
                      np.random.uniform(0.2, 0.5)]
        else:  # MINOR
            injury = [np.random.uniform(0.0, 0.35),
                      np.random.uniform(0.0, 0.35),
                      np.random.uniform(0.0, 0.1),
                      np.random.uniform(0.0, 0.35),
                      np.random.uniform(0.0, 0.4),
                      np.random.uniform(0.0, 0.3),
                      np.random.uniform(0.0, 0.4),
                      np.random.uniform(0.0, 0.3),
                      np.random.uniform(0.5, 1.0),
                      np.random.uniform(0.1, 0.4)]

        # Visual features (66)
        visual = np.random.uniform(0, 1, 66)

        return np.array(injury + list(visual), dtype=np.float32)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_visual_detectors(n_samples=150, output_dir="models"):
    """Train fight, fall, and medical detectors"""
    print("\n" + "="*60)
    print("TRAINING VISUAL EVENT DETECTORS")
    print("="*60)

    generator = VisualDataGenerator()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {}

    # Train Fight Detector
    print("\n[1/3] Training Fight Detector (Random Forest)...")
    X, y = generator.generate_fight_data(n_samples)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_gpu = cp.array(X_train)
    X_test_gpu = cp.array(X_test)
    X_train_scaled = scaler.fit_transform(X_train_gpu)
    X_test_scaled = scaler.transform(X_test_gpu)

    fight_model = cuRF(n_estimators=100, max_depth=16, random_state=42)
    fight_model.fit(X_train_scaled, cp.array(y_train))

    y_pred = cp.asnumpy(fight_model.predict(X_test_scaled))
    acc = np.mean(y_pred == y_test)
    print(f"   Accuracy: {acc:.2%}")
    results['fight_accuracy'] = acc

    # Train Fall Detector
    print("\n[2/3] Training Fall Detector (SVM)...")
    X, y = generator.generate_fall_data(n_samples)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_gpu = cp.array(X_train)
    X_test_gpu = cp.array(X_test)
    scaler2 = StandardScaler()
    X_train_scaled = scaler2.fit_transform(X_train_gpu)
    X_test_scaled = scaler2.transform(X_test_gpu)

    fall_model = cuSVC(kernel='rbf', C=10.0, probability=True)
    fall_model.fit(X_train_scaled, cp.array(y_train))

    y_pred = cp.asnumpy(fall_model.predict(X_test_scaled))
    acc = np.mean(y_pred == y_test)
    print(f"   Accuracy: {acc:.2%}")
    results['fall_accuracy'] = acc

    # Train Medical Detector
    print("\n[3/3] Training Medical Detector (KNN)...")
    X, y = generator.generate_medical_data(n_samples)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_gpu = cp.array(X_train)
    X_test_gpu = cp.array(X_test)
    scaler3 = StandardScaler()
    X_train_scaled = scaler3.fit_transform(X_train_gpu)
    X_test_scaled = scaler3.transform(X_test_gpu)

    medical_model = cuKNN(n_neighbors=5, metric='euclidean')
    medical_model.fit(X_train_scaled, cp.array(y_train))

    y_pred = cp.asnumpy(medical_model.predict(X_test_scaled))
    acc = np.mean(y_pred == y_test)
    print(f"   Accuracy: {acc:.2%}")
    results['medical_accuracy'] = acc

    print("\nVisual detectors trained!")
    return results


def train_severity_classifier(n_per_class=200, output_dir="models"):
    """Train injury severity classifier"""
    print("\n" + "="*60)
    print("TRAINING INJURY SEVERITY CLASSIFIER")
    print("="*60)

    generator = SeverityDataGenerator()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate data
    print(f"\nGenerating {n_per_class * 4} samples...")
    X, y_sev, y_type, y_region = generator.generate_dataset(n_per_class)
    print(f"   Shape: {X.shape}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_sev, test_size=0.2, random_state=42, stratify=y_sev
    )

    # Scale
    scaler = StandardScaler()
    X_train_gpu = cp.array(X_train)
    X_test_gpu = cp.array(X_test)
    X_train_scaled = scaler.fit_transform(X_train_gpu)
    X_test_scaled = scaler.transform(X_test_gpu)

    # Train severity model
    print("\nTraining severity classifier...")
    if XGBOOST_AVAILABLE:
        print("   Using XGBoost GPU")
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=10, learning_rate=0.1,
            tree_method='gpu_hist', gpu_id=0, random_state=42
        )
        model.fit(cp.asnumpy(X_train_scaled), y_train)
        y_pred = model.predict(cp.asnumpy(X_test_scaled))
    else:
        print("   Using cuML Random Forest (XGBoost not available)")
        model = cuRF(n_estimators=200, max_depth=14, random_state=42)
        model.fit(X_train_scaled, cp.array(y_train))
        y_pred = cp.asnumpy(model.predict(X_test_scaled))

    # Evaluate
    labels = ['MINOR', 'MODERATE', 'SEVERE', 'CRITICAL']
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))

    acc = np.mean(y_pred == y_test)
    print(f"\nOverall Accuracy: {acc:.2%}")

    return {'severity_accuracy': acc}


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train cuML emergency detection models')
    parser.add_argument('--samples', type=int, default=200, help='Samples per class')
    parser.add_argument('--output', type=str, default='models', help='Output directory')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("cuML EMERGENCY DETECTION MODEL TRAINING")
    print("NVIDIA RAPIDS GPU-Accelerated")
    print("="*60)

    # Check GPU
    try:
        import cuml
        print(f"\ncuML version: {cuml.__version__}")
        print("GPU acceleration: ENABLED")
    except ImportError:
        print("\nWARNING: cuML not available")

    # Train all models
    visual_results = train_visual_detectors(args.samples, args.output)
    severity_results = train_severity_classifier(args.samples, args.output)

    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("\nResults:")
    print(f"  Fight Detector: {visual_results['fight_accuracy']:.2%}")
    print(f"  Fall Detector: {visual_results['fall_accuracy']:.2%}")
    print(f"  Medical Detector: {visual_results['medical_accuracy']:.2%}")
    print(f"  Severity Classifier: {severity_results['severity_accuracy']:.2%}")

    # Save metadata
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'samples_per_class': args.samples,
        **visual_results,
        **severity_results
    }

    Path(args.output).mkdir(parents=True, exist_ok=True)
    with open(f"{args.output}/training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to {args.output}/training_metadata.json")


if __name__ == "__main__":
    main()
