# NVIDIA NIM Emergency Detection System

**Cloud-Based Real-Time Emergency Detection using NVIDIA Hosted Models**

All visual and audio inference runs on NVIDIA's cloud infrastructure via NIM (NVIDIA Inference Microservices). No local GPU required for inference.

---

## Quick Start

```bash
# 1. Set API key
export NVIDIA_API_KEY="nvapi-xxxx"  # Get from build.nvidia.com

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run detection
python main.py --webcam
```

---

## System Architecture

```
Camera → NVIDIA NIM (Visual) → Analysis → Dispatch
           │                                 │
           │ Florence-2, DINO               │ Llama (NGC)
           │                                 │
Audio  → NVIDIA NIM (Audio)  → ───────────→┘
           │
           │ Parakeet ASR, Audio Embedding
```

**All models run on NVIDIA cloud - no local GPU inference required.**

---

## Models Used

### Visual Detection (NVIDIA NIM)

| Model | Purpose | Detects |
|-------|---------|---------|
| `microsoft/florence-2` | Scene understanding | Running, standing, walking, fighting, falling |
| `nvidia/grounding-dino` | Person detection | Bounding boxes, tracking |
| `meta/sam2-hiera-large` | Segmentation | Precise person outlines |
| `nvidia/bodypose-estimation` | Pose analysis | 17 keypoints for movement |

### Audio Detection (NVIDIA NIM)

| Model | Purpose | Detects |
|-------|---------|---------|
| `nvidia/parakeet-ctc-1.1b` | Speech recognition | "Help", "911", distress calls |
| `nvidia/canary-1b` | Multilingual ASR | Non-English distress |
| `nvidia/audio-embedding` | Sound classification | Coughing, screams, fighting |

### Dispatch (NGC - Unchanged)

| Model | Purpose |
|-------|---------|
| Llama 3 70B (fine-tuned) | Emergency response routing |

---

## Detection Capabilities

### Visual Actions

| Action | Detection Method |
|--------|------------------|
| **Standing** | Vertical posture, minimal movement |
| **Walking** | Moderate leg movement, steady pace |
| **Running** | High leg movement, rapid changes |
| **Fighting** | Erratic movements, multiple persons |
| **Falling** | Rapid vertical change, body horizontal |

### Audio Events

| Event | Detection Method |
|-------|------------------|
| **Help calls** | ASR keyword detection |
| **Coughing** | Audio classification + energy patterns |
| **Screaming** | High energy + high pitch |
| **Fighting sounds** | Irregular bursts, chaos score |

---

## Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point - unified system |
| `nvidia_nim_visual.py` | Visual inference (Florence-2, DINO, BodyPose) |
| `nvidia_nim_audio.py` | Audio inference (Parakeet, Audio Embedding) |
| `nvidia_nim_integrated.py` | Combined visual + audio analysis |
| `requirements.txt` | Dependencies |

### Legacy Files (Deprecated)

| File | Status |
|------|--------|
| `visual_event_detector.py` | Deprecated - cuML-based (local GPU) |
| `injury_severity_classifier.py` | Deprecated - cuML-based (local GPU) |
| `train_models.py` | Deprecated - no training needed for NIM |

---

## Usage

### Command Line

```bash
# Webcam with visual only
python main.py --webcam

# Video file with audio
python main.py --video incident.mp4 --audio

# Headless mode
python main.py --video stream.mp4 --no-display

# Disable dispatch
python main.py --webcam --no-dispatch
```

### Python API

```python
from nvidia_nim_integrated import NVIDIANIMIntegratedSystem

# Initialize
system = NVIDIANIMIntegratedSystem()

# Set alert callback
def on_alert(result):
    print(f"ALERT: {result.emergency_type.value}")
    print(f"Severity: {result.overall_severity.name}")
    if result.dispatch_recommended:
        print(f"Action: {result.recommended_response}")

system.set_alert_callback(on_alert)

# Process frame
result = system.analyze_frame(video_frame, audio_chunk)

# Check detections
print(f"Persons: {result.persons_detected}")
print(f"Actions: {[a.value for a in result.visual_actions]}")
print(f"Fight: {result.fight_detected}")
print(f"Fall: {result.fall_detected}")
print(f"Help call: {result.help_detected}")
```

### Visual Only

```python
from nvidia_nim_visual import NVIDIANIMVisualInference

analyzer = NVIDIANIMVisualInference()
result = analyzer.analyze_frame(frame)

print(f"Scene action: {result.scene_action.value}")
print(f"Fight: {result.fight_detected}")
print(f"Fall: {result.fall_detected}")
```

### Audio Only

```python
from nvidia_nim_audio import NVIDIANIMAudioInference

analyzer = NVIDIANIMAudioInference()
result = analyzer.analyze_audio(audio_samples, sample_rate=16000)

print(f"Transcript: {result.transcript}")
print(f"Help detected: {result.help_detected}")
print(f"Coughing: {result.coughing_detected}")
```

---

## Configuration

### Environment

```bash
# Required
export NVIDIA_API_KEY="nvapi-xxxx"

# Get key from: https://build.nvidia.com/
```

### Cameras

```python
# main.py
CAMERAS = {
    "CAM-001": CameraConfig(
        camera_id="CAM-001",
        latitude=37.7838,
        longitude=-122.4167,
        address="455 Golden Gate Ave, Tenderloin"
    ),
}
```

### Skip Frames

```python
# Adjust for API efficiency vs responsiveness
skip_frames = 5  # Process every 5th frame (default)
```

---

## Severity Levels

| Level | Response | Trigger |
|-------|----------|---------|
| **CRITICAL** | Immediate 911 | Fight + fall, help call detected |
| **HIGH** | Urgent dispatch | Fight OR fall OR medical |
| **MEDIUM** | Monitor closely | Coughing, distress speech |
| **LOW** | Log only | Crowd activity |
| **NONE** | Normal | No incidents |

---

## API Costs

NVIDIA NIM is usage-based:

| Model | Cost |
|-------|------|
| Florence-2 | ~$0.001/request |
| Grounding DINO | ~$0.001/request |
| Parakeet ASR | ~$0.006/minute |
| Audio Embedding | ~$0.002/request |

**Optimization:**
- Skip frames (every 5th = 80% reduction)
- Batch audio (2-second chunks)
- Cache static scenes

---

## Requirements

```
# Core
opencv-python>=4.5
numpy>=1.20
requests>=2.25

# Optional (for local display)
# No GPU required - all inference on NVIDIA cloud
```

**Python:** 3.10+
**NVIDIA API Key:** Required (free tier available)

---

## Comparison: NIM vs Local cuML

| Aspect | NVIDIA NIM (New) | Local cuML (Old) |
|--------|------------------|------------------|
| GPU Required | No | Yes (NVIDIA) |
| Setup | API key only | CUDA + RAPIDS |
| Models | State-of-the-art | Custom trained |
| Accuracy | Higher | Good |
| Latency | ~100-500ms | ~20-50ms |
| Cost | Per-request | Hardware |
| Maintenance | None | Model updates |

**Recommendation:** Use NIM for simplicity and accuracy. Use local cuML only if latency-critical (<50ms needed).

---

## Documentation

See `/NVIDIA_NIM_MODELS.md` at project root for complete model documentation.

---

## License

MIT License

**NVIDIA NIM:** Requires API key from [build.nvidia.com](https://build.nvidia.com/)
