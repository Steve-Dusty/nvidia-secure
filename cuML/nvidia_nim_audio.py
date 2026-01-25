#!/usr/bin/env python3
"""
NVIDIA NIM Audio Inference Module

Uses NVIDIA hosted models (NIM) for audio detection:
- Speech recognition for "help" calls
- Audio event detection: coughing, screaming, fighting sounds
- Ambient chaos/disturbance detection

All inference runs on NVIDIA's cloud infrastructure via API.
"""

import os
import base64
import json
import time
import wave
import io
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import requests
import numpy as np

# NVIDIA NIM API Configuration
NVIDIA_API_BASE = "https://ai.api.nvidia.com/v1"
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")


class AudioEventType(Enum):
    """Detected audio events"""
    SILENCE = "silence"
    SPEECH_NORMAL = "speech_normal"
    SPEECH_DISTRESS = "speech_distress"
    HELP_CALL = "help_call"
    SCREAM = "scream"
    COUGHING = "coughing"
    FIGHTING_SOUNDS = "fighting_sounds"
    GLASS_BREAKING = "glass_breaking"
    GUNSHOT = "gunshot"
    CROWD_NOISE = "crowd_noise"
    CHAOS = "chaos"
    UNKNOWN = "unknown"


class AudioSeverity(Enum):
    """Audio event severity levels"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AudioEvent:
    """Represents a detected audio event"""
    event_type: AudioEventType
    confidence: float
    start_time_ms: float
    end_time_ms: float
    transcript: Optional[str] = None
    keywords_detected: List[str] = field(default_factory=list)


@dataclass
class AudioAnalysisResult:
    """Complete result from audio analysis"""
    timestamp: datetime
    duration_ms: float
    events: List[AudioEvent]
    primary_event: AudioEventType
    severity: AudioSeverity
    help_detected: bool = False
    coughing_detected: bool = False
    fighting_detected: bool = False
    chaos_level: float = 0.0
    transcript: Optional[str] = None
    inference_time_ms: float = 0.0
    raw_responses: Dict = field(default_factory=dict)


class NVIDIANIMAudioInference:
    """
    Audio inference using NVIDIA NIM hosted models.

    Models used:
    - nvidia/parakeet-ctc-1.1b: Automatic Speech Recognition (ASR)
    - nvidia/canary-1b: Multilingual ASR with better accuracy
    - nvidia/audio-embedding: Audio classification embeddings

    Detection capabilities:
    - "Help" and distress calls via ASR
    - Coughing detection via audio classification
    - Fighting/chaos via sound event detection
    """

    # NVIDIA NIM Model Endpoints
    MODELS = {
        "asr_primary": "nvidia/parakeet-ctc-1.1b",
        "asr_multilingual": "nvidia/canary-1b",
        "audio_classification": "nvidia/audio-embedding",
    }

    # Keywords indicating distress
    DISTRESS_KEYWORDS = [
        "help", "help me", "somebody help", "call 911", "emergency",
        "stop", "no", "please stop", "get off", "let go",
        "fire", "gun", "knife", "attack", "police",
        "hurt", "pain", "can't breathe", "dying", "bleeding"
    ]

    # Audio event energy thresholds
    ENERGY_THRESHOLDS = {
        "silence": 0.01,
        "speech": 0.1,
        "loud": 0.5,
        "scream": 0.8
    }

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with NVIDIA API key"""
        self.api_key = api_key or NVIDIA_API_KEY
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY environment variable required")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }

        # Audio state tracking
        self.audio_history: List[AudioAnalysisResult] = []
        self.sample_count = 0

    def _encode_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Encode audio to base64 WAV format for API"""
        # Convert to 16-bit PCM
        if audio_data.dtype != np.int16:
            audio_data = (audio_data * 32767).astype(np.int16)

        # Create WAV in memory
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    def _call_nim_asr(self, audio_b64: str, model: str = "nvidia/parakeet-ctc-1.1b") -> Dict:
        """Call NVIDIA NIM ASR endpoint"""
        url = f"{NVIDIA_API_BASE}/asr/{model}"

        payload = {
            "audio": audio_b64,
            "config": {
                "language": "en",
                "punctuation": True,
                "word_timestamps": True
            }
        }

        try:
            response = requests.post(
                url,
                headers={**self.headers, "Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"NIM ASR error: {e}")
            return {"error": str(e)}

    def _call_nim_audio_classification(self, audio_b64: str) -> Dict:
        """
        Call NVIDIA NIM audio classification endpoint.

        Returns embeddings and classification scores for:
        - Speech vs non-speech
        - Sound event categories
        """
        url = f"{NVIDIA_API_BASE}/audio/nvidia/audio-embedding"

        payload = {
            "audio": audio_b64,
            "return_embeddings": False,
            "return_classifications": True
        }

        try:
            response = requests.post(
                url,
                headers={**self.headers, "Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"NIM audio classification error: {e}")
            return {"error": str(e)}

    def _analyze_audio_features(self, audio_data: np.ndarray,
                                 sample_rate: int = 16000) -> Dict:
        """
        Extract audio features for event detection.

        Features:
        - RMS energy (loudness)
        - Zero crossing rate (voice vs noise)
        - Spectral centroid (pitch)
        - Energy variance (chaos indicator)
        """
        # Normalize audio
        audio_float = audio_data.astype(np.float32)
        if audio_float.max() > 1.0:
            audio_float = audio_float / 32768.0

        # RMS Energy
        rms = np.sqrt(np.mean(audio_float ** 2))

        # Zero Crossing Rate
        zcr = np.sum(np.abs(np.diff(np.sign(audio_float)))) / (2 * len(audio_float))

        # Simple spectral analysis
        fft = np.fft.rfft(audio_float)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(audio_float), 1/sample_rate)

        # Spectral centroid
        spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)

        # Energy variance (windowed)
        window_size = sample_rate // 10  # 100ms windows
        n_windows = len(audio_float) // window_size
        window_energies = []
        for i in range(n_windows):
            window = audio_float[i*window_size:(i+1)*window_size]
            window_energies.append(np.sqrt(np.mean(window ** 2)))
        energy_variance = np.var(window_energies) if window_energies else 0

        return {
            "rms_energy": float(rms),
            "zero_crossing_rate": float(zcr),
            "spectral_centroid": float(spectral_centroid),
            "energy_variance": float(energy_variance),
            "peak_amplitude": float(np.max(np.abs(audio_float))),
            "duration_seconds": len(audio_float) / sample_rate
        }

    def _detect_coughing(self, features: Dict, classifications: Dict) -> Tuple[bool, float]:
        """
        Detect coughing from audio features.

        Coughing characteristics:
        - Short bursts of energy
        - Specific spectral pattern
        - Repeated pattern
        """
        # Check classification results first
        if "cough" in classifications.get("labels", []):
            idx = classifications["labels"].index("cough")
            conf = classifications.get("scores", [0])[idx]
            if conf > 0.5:
                return True, conf

        # Heuristic detection
        # Coughing: high energy bursts with high variance
        rms = features["rms_energy"]
        variance = features["energy_variance"]
        zcr = features["zero_crossing_rate"]

        # Cough pattern: moderate-high energy, high variance, specific ZCR range
        if rms > 0.1 and variance > 0.01 and 0.1 < zcr < 0.4:
            confidence = min(0.7, rms * 2 + variance * 10)
            return True, confidence

        return False, 0.1

    def _detect_scream(self, features: Dict, transcript: str) -> Tuple[bool, float]:
        """
        Detect screaming from audio features.

        Scream characteristics:
        - Very high energy
        - High pitch (spectral centroid)
        - Sustained or sudden onset
        """
        rms = features["rms_energy"]
        centroid = features["spectral_centroid"]
        peak = features["peak_amplitude"]

        # High energy + high pitch = likely scream
        if rms > self.ENERGY_THRESHOLDS["scream"] or peak > 0.9:
            if centroid > 1500:  # High pitch
                return True, 0.85
            return True, 0.7

        # Check transcript for scream indicators
        if any(word in transcript.lower() for word in ["aah", "ahhh", "aaah", "help"]):
            return True, 0.75

        return False, 0.1

    def _detect_fighting_sounds(self, features: Dict, classifications: Dict) -> Tuple[bool, float]:
        """
        Detect fighting sounds (impacts, grunts, chaos).

        Fighting characteristics:
        - Irregular high-energy bursts
        - Multiple voice sources
        - Impact sounds
        """
        variance = features["energy_variance"]
        rms = features["rms_energy"]
        zcr = features["zero_crossing_rate"]

        # Fighting: high variance (chaotic), moderate-high energy
        chaos_score = variance * 100 + rms * 2

        # Check classification for fighting-related sounds
        fighting_labels = ["impact", "fighting", "punch", "slap", "grunt", "shout"]
        for label in fighting_labels:
            if label in str(classifications.get("labels", [])).lower():
                return True, 0.8

        if chaos_score > 1.0 and rms > 0.3:
            return True, min(0.9, chaos_score / 2)

        return False, 0.1

    def _detect_distress_speech(self, transcript: str) -> Tuple[bool, List[str], float]:
        """
        Detect distress keywords in transcript.

        Returns: (is_distress, keywords_found, confidence)
        """
        transcript_lower = transcript.lower()
        found_keywords = []

        for keyword in self.DISTRESS_KEYWORDS:
            if keyword in transcript_lower:
                found_keywords.append(keyword)

        if found_keywords:
            # Higher confidence for more keywords or critical keywords
            critical_keywords = ["help", "911", "emergency", "gun", "fire", "can't breathe"]
            has_critical = any(kw in found_keywords for kw in critical_keywords)

            confidence = 0.7 + (len(found_keywords) * 0.05)
            if has_critical:
                confidence = min(0.95, confidence + 0.15)

            return True, found_keywords, confidence

        return False, [], 0.1

    def _determine_severity(self, events: List[AudioEvent],
                            help_detected: bool,
                            fighting_detected: bool,
                            features: Dict) -> AudioSeverity:
        """Determine overall audio severity"""
        # Critical: help calls, screams + fighting
        if help_detected and fighting_detected:
            return AudioSeverity.CRITICAL

        # Critical: explicit help calls
        if help_detected:
            for event in events:
                if event.event_type == AudioEventType.HELP_CALL:
                    return AudioSeverity.CRITICAL

        # High: screams or fighting
        if fighting_detected:
            return AudioSeverity.HIGH

        for event in events:
            if event.event_type in [AudioEventType.SCREAM, AudioEventType.GUNSHOT]:
                return AudioSeverity.HIGH

        # Medium: coughing (medical) or distress speech
        for event in events:
            if event.event_type in [AudioEventType.COUGHING, AudioEventType.SPEECH_DISTRESS]:
                return AudioSeverity.MEDIUM

        # Low: crowd noise, general chaos
        if features["energy_variance"] > 0.05:
            return AudioSeverity.LOW

        return AudioSeverity.NONE

    def analyze_audio(self, audio_data: np.ndarray,
                      sample_rate: int = 16000) -> AudioAnalysisResult:
        """
        Complete audio analysis of audio segment.

        Uses:
        1. Parakeet ASR for speech recognition
        2. Audio classification for sound events
        3. Feature analysis for chaos/coughing detection
        """
        start_time = time.time()
        self.sample_count += 1

        duration_ms = (len(audio_data) / sample_rate) * 1000

        # Encode audio
        audio_b64 = self._encode_audio(audio_data, sample_rate)

        # Step 1: Speech recognition
        asr_result = self._call_nim_asr(audio_b64)
        transcript = asr_result.get("text", asr_result.get("transcript", ""))

        # Step 2: Audio classification
        classification_result = self._call_nim_audio_classification(audio_b64)

        # Step 3: Feature extraction
        features = self._analyze_audio_features(audio_data, sample_rate)

        # Step 4: Detect specific events
        events = []

        # Check for distress speech
        is_distress, keywords, distress_conf = self._detect_distress_speech(transcript)
        help_detected = "help" in keywords or "911" in keywords

        if is_distress:
            event_type = AudioEventType.HELP_CALL if help_detected else AudioEventType.SPEECH_DISTRESS
            events.append(AudioEvent(
                event_type=event_type,
                confidence=distress_conf,
                start_time_ms=0,
                end_time_ms=duration_ms,
                transcript=transcript,
                keywords_detected=keywords
            ))

        # Check for coughing
        coughing_detected, cough_conf = self._detect_coughing(features, classification_result)
        if coughing_detected:
            events.append(AudioEvent(
                event_type=AudioEventType.COUGHING,
                confidence=cough_conf,
                start_time_ms=0,
                end_time_ms=duration_ms
            ))

        # Check for scream
        scream_detected, scream_conf = self._detect_scream(features, transcript)
        if scream_detected:
            events.append(AudioEvent(
                event_type=AudioEventType.SCREAM,
                confidence=scream_conf,
                start_time_ms=0,
                end_time_ms=duration_ms
            ))

        # Check for fighting
        fighting_detected, fight_conf = self._detect_fighting_sounds(features, classification_result)
        if fighting_detected:
            events.append(AudioEvent(
                event_type=AudioEventType.FIGHTING_SOUNDS,
                confidence=fight_conf,
                start_time_ms=0,
                end_time_ms=duration_ms
            ))

        # Determine primary event and severity
        if events:
            events.sort(key=lambda e: e.confidence, reverse=True)
            primary_event = events[0].event_type
        elif features["rms_energy"] < self.ENERGY_THRESHOLDS["silence"]:
            primary_event = AudioEventType.SILENCE
        elif transcript:
            primary_event = AudioEventType.SPEECH_NORMAL
        else:
            primary_event = AudioEventType.UNKNOWN

        severity = self._determine_severity(
            events, help_detected, fighting_detected, features
        )

        # Calculate chaos level
        chaos_level = min(1.0, features["energy_variance"] * 50 + features["rms_energy"])

        inference_time = (time.time() - start_time) * 1000

        result = AudioAnalysisResult(
            timestamp=datetime.now(),
            duration_ms=duration_ms,
            events=events,
            primary_event=primary_event,
            severity=severity,
            help_detected=help_detected,
            coughing_detected=coughing_detected,
            fighting_detected=fighting_detected,
            chaos_level=chaos_level,
            transcript=transcript if transcript else None,
            inference_time_ms=inference_time,
            raw_responses={
                "asr": asr_result,
                "classification": classification_result,
                "features": features
            }
        )

        self.audio_history.append(result)
        return result

    def analyze_audio_stream(self, audio_generator, callback=None,
                             chunk_duration_ms: int = 2000):
        """
        Analyze audio stream in chunks.

        Args:
            audio_generator: Iterator yielding (timestamp, audio_chunk) tuples
            callback: Optional function called with each AudioAnalysisResult
            chunk_duration_ms: Size of audio chunks to analyze
        """
        for timestamp, audio_chunk in audio_generator:
            result = self.analyze_audio(audio_chunk)

            if callback:
                callback(result)

            # Alert on high severity
            if result.severity.value >= AudioSeverity.HIGH.value:
                print(f"[AUDIO ALERT] {timestamp}: {result.severity.name}")
                if result.help_detected:
                    print(f"  - HELP DETECTED: {result.transcript}")
                if result.fighting_detected:
                    print(f"  - FIGHTING SOUNDS DETECTED")
                if result.coughing_detected:
                    print(f"  - COUGHING DETECTED")

            yield result


# Convenience function for simple usage
def create_audio_analyzer(api_key: Optional[str] = None) -> NVIDIANIMAudioInference:
    """Create configured audio analyzer instance"""
    return NVIDIANIMAudioInference(api_key)


if __name__ == "__main__":
    print("NVIDIA NIM Audio Inference Module")
    print("=" * 50)
    print(f"Models configured:")
    for name, model in NVIDIANIMAudioInference.MODELS.items():
        print(f"  {name}: {model}")
    print()
    print("Detection capabilities:")
    print("  - Distress speech recognition ('help', '911', etc.)")
    print("  - Coughing detection")
    print("  - Scream detection")
    print("  - Fighting/chaos detection")
    print()

    if not NVIDIA_API_KEY:
        print("Set NVIDIA_API_KEY environment variable to test")
    else:
        print("API key configured - ready for inference")
