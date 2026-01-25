#!/usr/bin/env python3
"""
Train cuML Visual Event Detection Models

This script trains the GPU-accelerated cuML models for:
- Fight detection (Random Forest)
- Fall detection (SVM)
- Medical emergency detection (KNN)

Uses synthetic and augmented training data
"""

import os
import cv2
import numpy as np
import cupy as cp
import cudf
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from typing import Tuple, List
import json

from visual_event_detector import (
    VisualFeatureExtractor,
    FightDetector,
    FallDetector,
    MedicalEmergencyDetector
)


# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================

class SyntheticDataGenerator:
    """Generate synthetic training data for visual event detection"""

    def __init__(self, img_size: Tuple[int, int] = (640, 480)):
        self.img_size = img_size
        self.feature_extractor = VisualFeatureExtractor(use_deep_features=False)

    def generate_fight_sequence(self, num_frames: int = 30) -> List[np.ndarray]:
        """
        Generate synthetic fight sequence

        Characteristics:
        - Multiple moving objects (people)
        - High motion intensity
        - Chaotic movement patterns
        - Rapid direction changes
        """
        frames = []

        for i in range(num_frames):
            frame = np.zeros((*self.img_size[::-1], 3), dtype=np.uint8)

            # Create multiple "people" (represented as rectangles)
            num_people = np.random.randint(2, 4)

            for _ in range(num_people):
                # Random position with high motion
                x = int(320 + np.random.randn() * 150)
                y = int(240 + np.random.randn() * 100)

                # Random size
                w = np.random.randint(40, 80)
                h = np.random.randint(80, 150)

                # Random color
                color = tuple(np.random.randint(0, 255, 3).tolist())

                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, -1)

                # Add motion blur to simulate movement
                kernel_size = np.random.randint(5, 15)
                kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
                frame = cv2.filter2D(frame, -1, kernel)

            # Add noise (chaos)
            noise = np.random.randn(*frame.shape) * 30
            frame = np.clip(frame + noise, 0, 255).astype(np.uint8)

            frames.append(frame)

        return frames

    def generate_fall_sequence(self, num_frames: int = 30) -> List[np.ndarray]:
        """
        Generate synthetic fall sequence

        Characteristics:
        - Single person
        - Vertical motion (downward)
        - Rotation from vertical to horizontal
        - Sudden stop at bottom
        """
        frames = []

        # Start position (standing)
        person_y = 100
        person_h = 150
        person_w = 60

        for i in range(num_frames):
            frame = np.zeros((*self.img_size[::-1], 3), dtype=np.uint8)

            # Person position (falling)
            progress = i / num_frames

            if progress < 0.7:  # Falling phase
                # Move downward
                person_y = int(100 + progress * 300)

                # Rotate (vertical to horizontal)
                angle = progress * 90  # 0 to 90 degrees

                # Draw rotated rectangle (simplified)
                center = (320, person_y + person_h // 2)

                if angle > 45:
                    # Horizontal person (fallen)
                    cv2.rectangle(frame,
                                (center[0] - person_h//2, center[1] - person_w//2),
                                (center[0] + person_h//2, center[1] + person_w//2),
                                (100, 150, 200), -1)
                else:
                    # Vertical person (falling)
                    cv2.rectangle(frame,
                                (center[0] - person_w//2, center[1] - person_h//2),
                                (center[0] + person_w//2, center[1] + person_h//2),
                                (100, 150, 200), -1)
            else:
                # On ground, not moving
                cv2.rectangle(frame,
                            (320 - 75, 400 - 30),
                            (320 + 75, 400 + 30),
                            (100, 150, 200), -1)

            frames.append(frame)

        return frames

    def generate_normal_sequence(self, num_frames: int = 30) -> List[np.ndarray]:
        """
        Generate normal activity sequence

        Characteristics:
        - Smooth, directional motion
        - Low motion intensity
        - Predictable movement
        """
        frames = []

        # Person walking across frame
        for i in range(num_frames):
            frame = np.zeros((*self.img_size[::-1], 3), dtype=np.uint8)

            # Position (walking left to right)
            x = int(50 + (i / num_frames) * 540)
            y = 300

            # Draw person
            cv2.rectangle(frame, (x-30, y-75), (x+30, y+75), (150, 200, 100), -1)

            frames.append(frame)

        return frames

    def generate_medical_emergency_sequence(self, num_frames: int = 30) -> List[np.ndarray]:
        """
        Generate medical emergency sequence

        Characteristics:
        - Person on ground
        - Minimal or erratic motion
        - Other people gathering
        """
        frames = []

        for i in range(num_frames):
            frame = np.zeros((*self.img_size[::-1], 3), dtype=np.uint8)

            # Person on ground (not moving)
            cv2.rectangle(frame, (250, 380), (400, 420), (100, 100, 150), -1)

            # Bystanders gathering (if later in sequence)
            if i > 10:
                num_bystanders = min((i - 10) // 5, 3)
                for j in range(num_bystanders):
                    x = 200 + j * 80
                    y = 250
                    cv2.rectangle(frame, (x-20, y-60), (x+20, y+60),
                                (200, 150, 100), -1)

            # Add slight motion blur for seizure variant
            if i % 3 == 0:
                kernel = np.ones((3, 3), np.float32) / 9
                frame = cv2.filter2D(frame, -1, kernel)

            frames.append(frame)

        return frames

    def extract_features_from_sequence(self, frames: List[np.ndarray]) -> np.ndarray:
        """Extract features from frame sequence"""
        features_list = []

        for i, frame in enumerate(frames):
            prev_frame = frames[i-1] if i > 0 else None
            features = self.feature_extractor.extract_combined_features(frame, prev_frame)
            features_list.append(features)

        # Average features across sequence
        avg_features = np.mean(features_list, axis=0)

        return avg_features


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def generate_training_data(num_samples_per_class: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data

    Returns:
        X: Feature matrix (num_samples, num_features)
        y: Labels (num_samples,)
    """
    print("🔄 Generating synthetic training data...")

    generator = SyntheticDataGenerator()

    X_list = []
    y_list = []

    # Generate fight samples (label = 1)
    print(f"   Generating {num_samples_per_class} fight sequences...")
    for i in range(num_samples_per_class):
        frames = generator.generate_fight_sequence(num_frames=20)
        features = generator.extract_features_from_sequence(frames)
        X_list.append(features)
        y_list.append(1)

        if (i + 1) % 20 == 0:
            print(f"      Progress: {i+1}/{num_samples_per_class}")

    # Generate normal samples (label = 0)
    print(f"   Generating {num_samples_per_class} normal sequences...")
    for i in range(num_samples_per_class):
        frames = generator.generate_normal_sequence(num_frames=20)
        features = generator.extract_features_from_sequence(frames)
        X_list.append(features)
        y_list.append(0)

        if (i + 1) % 20 == 0:
            print(f"      Progress: {i+1}/{num_samples_per_class}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    print(f"✅ Generated {len(X)} training samples")
    print(f"   Feature dimensionality: {X.shape[1]}")

    return X, y


def generate_fall_training_data(num_samples_per_class: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Generate fall-specific training data"""
    print("🔄 Generating fall detection training data...")

    generator = SyntheticDataGenerator()

    X_list = []
    y_list = []

    # Generate fall samples (label = 1)
    print(f"   Generating {num_samples_per_class} fall sequences...")
    for i in range(num_samples_per_class):
        frames = generator.generate_fall_sequence(num_frames=20)
        features = generator.extract_features_from_sequence(frames)
        X_list.append(features)
        y_list.append(1)

    # Generate normal samples (label = 0)
    print(f"   Generating {num_samples_per_class} normal sequences...")
    for i in range(num_samples_per_class):
        frames = generator.generate_normal_sequence(num_frames=20)
        features = generator.extract_features_from_sequence(frames)
        X_list.append(features)
        y_list.append(0)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    print(f"✅ Generated {len(X)} fall training samples")

    return X, y


def generate_medical_training_data(num_samples_per_class: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Generate medical emergency training data"""
    print("🔄 Generating medical emergency training data...")

    generator = SyntheticDataGenerator()

    X_list = []
    y_list = []

    # Generate medical emergency samples (label = 1)
    print(f"   Generating {num_samples_per_class} medical emergency sequences...")
    for i in range(num_samples_per_class):
        frames = generator.generate_medical_emergency_sequence(num_frames=20)
        features = generator.extract_features_from_sequence(frames)
        X_list.append(features)
        y_list.append(1)

    # Generate normal samples (label = 0)
    print(f"   Generating {num_samples_per_class} normal sequences...")
    for i in range(num_samples_per_class):
        frames = generator.generate_normal_sequence(num_frames=20)
        features = generator.extract_features_from_sequence(frames)
        X_list.append(features)
        y_list.append(0)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    print(f"✅ Generated {len(X)} medical emergency samples")

    return X, y


def train_fight_detector(num_samples: int = 200):
    """Train fight detection model"""
    print("\n" + "="*70)
    print("TRAINING FIGHT DETECTOR")
    print("="*70)

    # Generate data
    X, y = generate_training_data(num_samples_per_class=num_samples)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n📊 Data split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")

    # Convert to CuPy arrays
    X_train_gpu = cp.array(X_train, dtype=cp.float32)
    y_train_gpu = cp.array(y_train, dtype=cp.int32)
    X_test_gpu = cp.array(X_test, dtype=cp.float32)
    y_test_gpu = cp.array(y_test, dtype=cp.int32)

    # Train model
    print("\n🔄 Training Random Forest model...")
    detector = FightDetector()
    detector.train(X_train_gpu, y_train_gpu)
    print("✅ Training complete")

    # Evaluate
    print("\n📈 Evaluating model...")
    y_pred = []
    for i in range(len(X_test)):
        _, confidence = detector.predict(X_test[i])
        y_pred.append(1 if confidence >= 0.75 else 0)

    y_pred = np.array(y_pred)

    # Metrics
    print("\n" + classification_report(y_test, y_pred,
                                      target_names=['Normal', 'Fight']))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    model_path = "models/fight_detector.pkl"
    os.makedirs("models", exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(detector, f)
    print(f"\n💾 Model saved to: {model_path}")

    return detector


def train_fall_detector(num_samples: int = 200):
    """Train fall detection model"""
    print("\n" + "="*70)
    print("TRAINING FALL DETECTOR")
    print("="*70)

    X, y = generate_fall_training_data(num_samples_per_class=num_samples)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n📊 Data split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")

    X_train_gpu = cp.array(X_train, dtype=cp.float32)
    y_train_gpu = cp.array(y_train, dtype=cp.int32)
    X_test_gpu = cp.array(X_test, dtype=cp.float32)

    print("\n🔄 Training SVM model...")
    detector = FallDetector()
    detector.train(X_train_gpu, y_train_gpu)
    print("✅ Training complete")

    # Evaluate
    print("\n📈 Evaluating model...")
    y_pred = []
    for i in range(len(X_test)):
        _, confidence = detector.predict(X_test[i])
        y_pred.append(1 if confidence >= 0.80 else 0)

    y_pred = np.array(y_pred)

    print("\n" + classification_report(y_test, y_pred,
                                      target_names=['Normal', 'Fall']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    model_path = "models/fall_detector.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(detector, f)
    print(f"\n💾 Model saved to: {model_path}")

    return detector


def train_medical_detector(num_samples: int = 200):
    """Train medical emergency detection model"""
    print("\n" + "="*70)
    print("TRAINING MEDICAL EMERGENCY DETECTOR")
    print("="*70)

    X, y = generate_medical_training_data(num_samples_per_class=num_samples)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n📊 Data split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")

    X_train_gpu = cp.array(X_train, dtype=cp.float32)
    y_train_gpu = cp.array(y_train, dtype=cp.int32)

    print("\n🔄 Training KNN model...")
    detector = MedicalEmergencyDetector()
    detector.train(X_train_gpu, y_train_gpu)
    print("✅ Training complete")

    # Evaluate
    print("\n📈 Evaluating model...")
    y_pred = []
    for i in range(len(X_test)):
        is_emergency, confidence = detector.predict(X_test[i])
        y_pred.append(1 if is_emergency else 0)

    y_pred = np.array(y_pred)

    print("\n" + classification_report(y_test, y_pred,
                                      target_names=['Normal', 'Medical Emergency']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    model_path = "models/medical_detector.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(detector, f)
    print(f"\n💾 Model saved to: {model_path}")

    return detector


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("🧠 cuML VISUAL EVENT DETECTOR - MODEL TRAINING")
    print("="*70)

    # Create models directory
    os.makedirs("models", exist_ok=True)

    # Train all models
    num_samples = 150  # Per class

    try:
        # Fight detector
        fight_detector = train_fight_detector(num_samples)

        # Fall detector
        fall_detector = train_fall_detector(num_samples)

        # Medical emergency detector
        medical_detector = train_medical_detector(num_samples)

        print("\n" + "="*70)
        print("✅ ALL MODELS TRAINED SUCCESSFULLY")
        print("="*70)
        print("\n📁 Saved models:")
        print("   - models/fight_detector.pkl")
        print("   - models/fall_detector.pkl")
        print("   - models/medical_detector.pkl")
        print("\n🚀 Ready for inference with visual_event_detector.py")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
