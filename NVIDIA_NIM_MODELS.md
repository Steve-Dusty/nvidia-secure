# NVIDIA NIM Models - Emergency Detection System

**Complete reference for all NVIDIA hosted models used in the emergency detection system.**

All inference runs on NVIDIA's cloud infrastructure via the NIM (NVIDIA Inference Microservices) API. This eliminates the need for local GPU resources while providing access to state-of-the-art models.

---

## Quick Reference

| Category | Model | Purpose |
|----------|-------|---------|
| **Visual - Scene Understanding** | microsoft/florence-2 | Action detection, scene description |
| **Visual - Person Detection** | nvidia/grounding-dino | Zero-shot object detection |
| **Visual - Segmentation** | meta/sam2-hiera-large | Precise person segmentation |
| **Visual - Pose Estimation** | nvidia/bodypose-estimation | Body keypoint detection |
| **Audio - Speech Recognition** | nvidia/parakeet-ctc-1.1b | English ASR (help calls) |
| **Audio - Multilingual ASR** | nvidia/canary-1b | Multi-language speech |
| **Audio - Sound Classification** | nvidia/audio-embedding | Audio event detection |
| **Dispatch - Reasoning** | NGC Llama (fine-tuned) | Emergency response routing |

---

## Visual Inference Models

### 1. Microsoft Florence-2

**Model ID:** `microsoft/florence-2`
**API Endpoint:** `https://ai.api.nvidia.com/v1/vlm/microsoft/florence-2`

**Purpose:** Vision-language model for scene understanding and action recognition.

**What it detects:**
- Human actions: running, standing, walking, fighting, falling, lying down
- Scene description and context
- Multiple people and their interactions
- Emergency situations and visual indicators

**How it's used:**

```python
# Action detection prompts
ACTION_PROMPTS = {
    "detect_actions": "Describe what each person in this image is doing. List actions: standing, walking, running, fighting, falling, lying down.",
    "detect_fight": "Is there a fight or physical altercation happening in this image? Answer yes or no and describe.",
    "detect_fall": "Has anyone fallen or is lying on the ground in this image? Answer yes or no and describe.",
    "detect_emergency": "Is there a medical emergency, injury, or dangerous situation in this image? Describe any concerns.",
}
```

**Response format:**
- Natural language description of scene
- Identified actions per person
- Yes/no answers for specific queries with reasoning

**Why this model:**
- State-of-the-art vision-language understanding
- Can answer complex visual questions
- Provides reasoning, not just labels
- Handles multiple prompts for comprehensive analysis

---

### 2. NVIDIA Grounding DINO

**Model ID:** `nvidia/grounding-dino`
**API Endpoint:** `https://ai.api.nvidia.com/v1/vlm/nvidia/grounding-dino`

**Purpose:** Zero-shot object detection with natural language grounding.

**What it detects:**
- People/persons in frame
- Bounding box coordinates for each person
- Confidence scores per detection

**How it's used:**

```python
# Detection prompt
payload = {
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "person. human. people."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        ]
    }]
}
```

**Response format:**
- List of bounding boxes [x1, y1, x2, y2] (normalized coordinates)
- Confidence score per detection
- Label (person/human)

**Why this model:**
- Zero-shot: No training needed for new categories
- Text-guided: Natural language object specification
- High accuracy person detection
- Returns precise bounding boxes for tracking

---

### 3. Meta SAM2 (Segment Anything 2)

**Model ID:** `meta/sam2-hiera-large`
**API Endpoint:** `https://ai.api.nvidia.com/v1/cv/meta/sam2-hiera-large`

**Purpose:** Precise segmentation of detected persons.

**What it does:**
- Pixel-level person segmentation
- Separates individuals in crowded scenes
- Enables accurate pose analysis

**How it's used:**
- Takes bounding boxes from Grounding DINO
- Returns precise segmentation masks
- Used for occlusion handling in crowds

**Why this model:**
- Best-in-class segmentation quality
- Works with any object (prompted by bounding box)
- Handles partial occlusions
- Enables precise body region analysis

---

### 4. NVIDIA BodyPose Estimation

**Model ID:** `nvidia/bodypose-estimation`
**API Endpoint:** `https://ai.api.nvidia.com/v1/cv/nvidia/bodypose-estimation`

**Purpose:** Detect body keypoints for movement analysis.

**Keypoints detected (17 points):**
```
nose, left_eye, right_eye, left_ear, right_ear,
left_shoulder, right_shoulder, left_elbow, right_elbow,
left_wrist, right_wrist, left_hip, right_hip,
left_knee, right_knee, left_ankle, right_ankle
```

**What it enables:**
- **Fall detection:** Hip-shoulder angle, head below hips
- **Running vs walking:** Leg movement magnitude
- **Standing:** Vertical posture, minimal movement
- **Fighting:** Arm movement patterns, body orientation

**How it's used:**

```python
def is_person_fallen(keypoints):
    """Detect fall from keypoint positions"""
    shoulder_y = (keypoints["left_shoulder"][1] + keypoints["right_shoulder"][1]) / 2
    hip_y = (keypoints["left_hip"][1] + keypoints["right_hip"][1]) / 2
    nose_y = keypoints["nose"][1]

    # Body horizontal (shoulders at same level as hips)
    vertical_diff = abs(shoulder_y - hip_y)

    # Head below hips = likely fallen
    head_below_hips = nose_y > hip_y

    if vertical_diff < 50 and head_below_hips:
        return True, 0.9  # High confidence fall
```

**Why this model:**
- NVIDIA-optimized for speed
- 17 keypoints sufficient for action analysis
- Works with multiple people
- Sub-millisecond inference on NIM

---

## Audio Inference Models

### 5. NVIDIA Parakeet CTC

**Model ID:** `nvidia/parakeet-ctc-1.1b`
**API Endpoint:** `https://ai.api.nvidia.com/v1/asr/nvidia/parakeet-ctc-1.1b`

**Purpose:** Primary automatic speech recognition (ASR) for English.

**What it detects:**
- Spoken words and phrases
- "Help", "911", "emergency" calls
- Distress keywords in speech
- Word-level timestamps

**Distress keywords monitored:**
```python
DISTRESS_KEYWORDS = [
    "help", "help me", "somebody help", "call 911", "emergency",
    "stop", "no", "please stop", "get off", "let go",
    "fire", "gun", "knife", "attack", "police",
    "hurt", "pain", "can't breathe", "dying", "bleeding"
]
```

**Response format:**
```json
{
    "text": "help me please somebody call 911",
    "words": [
        {"word": "help", "start": 0.1, "end": 0.3},
        {"word": "me", "start": 0.35, "end": 0.5},
        ...
    ]
}
```

**Why this model:**
- 1.1B parameters - highly accurate
- CTC architecture for streaming
- Word timestamps for event timing
- Optimized for noisy environments

---

### 6. NVIDIA Canary

**Model ID:** `nvidia/canary-1b`
**API Endpoint:** `https://ai.api.nvidia.com/v1/asr/nvidia/canary-1b`

**Purpose:** Multilingual ASR for diverse populations.

**Languages supported:**
- English, Spanish, German, French
- More languages with similar accuracy

**When used:**
- Backup/secondary ASR
- Non-English speaking areas
- Code-switching detection

**Why this model:**
- Same architecture as Parakeet
- Multilingual without quality loss
- Critical for diverse communities
- Better handling of accents

---

### 7. NVIDIA Audio Embedding

**Model ID:** `nvidia/audio-embedding`
**API Endpoint:** `https://ai.api.nvidia.com/v1/audio/nvidia/audio-embedding`

**Purpose:** Audio event classification beyond speech.

**Sound events detected:**
- **Coughing:** Respiratory distress indicator
- **Screaming:** Distress, fear, pain
- **Fighting sounds:** Impacts, grunts, chaos
- **Glass breaking:** Property damage, violence
- **Gunshot:** Critical emergency (future)
- **Crowd noise:** Gathering, incident

**How it's used:**

```python
def _detect_coughing(features, classifications):
    """Detect coughing from audio features"""
    # Check classification results
    if "cough" in classifications.get("labels", []):
        idx = classifications["labels"].index("cough")
        confidence = classifications["scores"][idx]
        if confidence > 0.5:
            return True, confidence

    # Heuristic fallback: energy bursts + high variance
    if rms > 0.1 and variance > 0.01 and 0.1 < zcr < 0.4:
        return True, 0.7
```

**Audio features analyzed:**
- RMS energy (loudness)
- Zero crossing rate (voice vs noise)
- Spectral centroid (pitch)
- Energy variance (chaos indicator)

**Why this model:**
- Non-speech audio understanding
- Embeddings enable custom classifiers
- Complements ASR for complete audio picture
- Fast inference for real-time use

---

## Dispatch Model (Unchanged)

### NGC Fine-tuned Llama

**Model:** Llama 3 70B (fine-tuned on NGC)
**Location:** NVIDIA NGC (not NIM API)

**Purpose:** Emergency response reasoning and dispatch decisions.

**NOT replaced because:**
- Custom fine-tuned for SF emergency data
- Domain-specific routing knowledge
- Trained on 50,000+ real EMS incidents
- Integrates with VAPI for voice dispatch

**What it does:**
- Determines response type (911 vs 311)
- Routes to nearest appropriate facility
- Estimates traffic-adjusted ETAs
- Generates dispatch instructions

**Integration:**
```python
# Llama receives detections from NIM models
event = EmergencyEvent(
    symptoms=["physical_altercation", "person_fallen"],  # From NIM visual
    location=camera.location,
    additional_context="NIM Detection - Fight and fall detected"
)

# Llama reasons about response
emergency_agent.process_emergency(event)
```

---

## System Architecture

```
                    NVIDIA NIM Cloud
                          │
    ┌─────────────────────┼─────────────────────┐
    │                     │                     │
    ▼                     ▼                     ▼
┌────────┐          ┌────────┐          ┌────────┐
│Florence│          │Grounding│         │Parakeet│
│   -2   │          │  DINO  │          │  ASR   │
└───┬────┘          └───┬────┘          └───┬────┘
    │                   │                   │
    │ Actions           │ Bboxes            │ Transcript
    │                   │                   │
    └─────────┬─────────┴─────────┬─────────┘
              │                   │
              ▼                   ▼
        ┌──────────┐        ┌──────────┐
        │BodyPose  │        │  Audio   │
        │Estimation│        │Embedding │
        └────┬─────┘        └────┬─────┘
             │                   │
             │ Keypoints         │ Events
             │                   │
             └─────────┬─────────┘
                       │
                       ▼
              ┌────────────────┐
              │   Integrated   │
              │   Analysis     │
              └───────┬────────┘
                      │
                      │ Detections
                      │
                      ▼
              ┌────────────────┐
              │  NGC Llama     │
              │  (fine-tuned)  │
              └───────┬────────┘
                      │
                      │ Dispatch
                      ▼
              ┌────────────────┐
              │     VAPI       │
              │  Voice Calls   │
              └────────────────┘
```

---

## Detection Capabilities Summary

### Visual Detections

| Action | Model | Detection Method |
|--------|-------|------------------|
| **Standing** | Florence-2 + BodyPose | Vertical posture, minimal keypoint movement |
| **Walking** | Florence-2 + BodyPose | Moderate leg keypoint changes, steady pace |
| **Running** | Florence-2 + BodyPose | High leg movement, rapid keypoint changes |
| **Fighting** | Florence-2 + Grounding DINO | Erratic movements, multiple persons in contact |
| **Falling** | Florence-2 + BodyPose | Rapid vertical change, hip-shoulder angle horizontal |
| **Lying down** | Florence-2 + BodyPose | Body horizontal, head at hip level |

### Audio Detections

| Event | Model | Detection Method |
|-------|-------|------------------|
| **Help calls** | Parakeet ASR | Keyword match in transcript |
| **Coughing** | Audio Embedding | Classification + energy variance |
| **Screaming** | Parakeet + Embedding | High energy + high pitch + transcript |
| **Fighting sounds** | Audio Embedding | Irregular energy bursts, chaos score |
| **Chaos** | Audio Embedding | High energy variance across windows |

---

## API Configuration

### Environment Variable

```bash
export NVIDIA_API_KEY="nvapi-xxxx"
```

### Get API Key

1. Go to [build.nvidia.com](https://build.nvidia.com/)
2. Create account or sign in
3. Navigate to any model
4. Click "Get API Key"
5. Copy and set as environment variable

### Rate Limits

- Default: 100 requests/minute
- Enterprise: Custom limits available
- Batch processing recommended for video

---

## Usage Examples

### Basic Visual Detection

```python
from nvidia_nim_visual import NVIDIANIMVisualInference
import cv2

# Initialize
analyzer = NVIDIANIMVisualInference()

# Process frame
frame = cv2.imread("scene.jpg")
result = analyzer.analyze_frame(frame)

print(f"Persons: {len(result.persons)}")
print(f"Actions: {[p.action.value for p in result.persons]}")
print(f"Fight: {result.fight_detected}")
print(f"Fall: {result.fall_detected}")
```

### Basic Audio Detection

```python
from nvidia_nim_audio import NVIDIANIMAudioInference
import numpy as np

# Initialize
analyzer = NVIDIANIMAudioInference()

# Process audio (16kHz, mono)
audio = np.random.randn(16000 * 2)  # 2 seconds
result = analyzer.analyze_audio(audio, sample_rate=16000)

print(f"Transcript: {result.transcript}")
print(f"Help detected: {result.help_detected}")
print(f"Coughing: {result.coughing_detected}")
print(f"Severity: {result.severity.name}")
```

### Integrated System

```python
from nvidia_nim_integrated import NVIDIANIMIntegratedSystem

# Initialize with alert callback
def on_alert(result):
    print(f"ALERT: {result.emergency_type.value}")
    print(f"Severity: {result.overall_severity.name}")

system = NVIDIANIMIntegratedSystem()
system.set_alert_callback(on_alert)

# Process frame + audio together
result = system.analyze_frame(video_frame, audio_chunk)

if result.dispatch_recommended:
    print(f"Dispatch: {result.recommended_response}")
```

### Command Line

```bash
# Webcam with visual only
python main.py --webcam

# Video file with audio
python main.py --video incident.mp4 --audio

# Headless mode (no display)
python main.py --video stream.mp4 --no-display
```

---

## Files Reference

### Main Inference (cuML/)

| File | Purpose |
|------|---------|
| `cuML/nvidia_nim_visual.py` | Visual inference (Florence-2, DINO, SAM2, BodyPose) |
| `cuML/nvidia_nim_audio.py` | Audio inference (Parakeet, Canary, Audio Embedding) |
| `cuML/nvidia_nim_integrated.py` | Combined visual + audio analysis |
| `cuML/main.py` | Entry point and real-time monitoring |

### ARM Deployment (Scripts-ARM/)

| File | Purpose |
|------|---------|
| `Scripts-ARM/run_inference.sh` | Main runner script for DGX Spark |
| `Scripts-ARM/setup_arm.sh` | Environment setup for ARM64 |
| `Scripts-ARM/nim_inference_arm.py` | ARM-optimized inference runner |
| `Scripts-ARM/Dockerfile.arm` | Docker image for ARM64 deployment |
| `Scripts-ARM/nvidia_nim_*.py` | NIM modules (same as cuML/) |

### Web Application (webapp/)

| File | Purpose |
|------|---------|
| `webapp/backend.py` | WebSocket server with NIM detection |

### Llama Fine-tuning (scripts-medresp/)

| File | Purpose |
|------|---------|
| `scripts-medresp/generate_routing_training.py` | Training data generator |
| `response-output/emergency_response_agent.py` | Llama dispatch agent |

### Deprecated Files (Local GPU)

| File | Status |
|------|--------|
| `cuML/visual_event_detector.py` | Deprecated - uses OpenCV DNN (local) |
| `cuML/injury_severity_classifier.py` | Deprecated - uses cuML (local GPU) |
| `cuML/train_models.py` | Deprecated - no training needed for NIM |

---

## DGX Spark / ARM Deployment

For deployment on NVIDIA DGX Spark with ARM (Grace) CPU:

```bash
cd Scripts-ARM

# Setup environment
./setup_arm.sh

# Run inference
./run_inference.sh --webcam

# Or with Docker
docker build -t nvidia-nim-arm:latest -f Dockerfile.arm .
docker run -e NVIDIA_API_KEY=$NVIDIA_API_KEY nvidia-nim-arm:latest
```

See `Scripts-ARM/README.md` for detailed ARM deployment documentation.

---

## Cost Considerations

NVIDIA NIM is usage-based:

| Model | Approximate Cost |
|-------|------------------|
| Florence-2 | ~$0.001/request |
| Grounding DINO | ~$0.001/request |
| Parakeet ASR | ~$0.006/minute |
| Audio Embedding | ~$0.002/request |

**Optimization strategies:**
- Skip frames (process every 5th frame)
- Batch audio chunks (2-second windows)
- Cache results for static scenes
- Use haiku model for low-severity routing

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-25 | Initial NVIDIA NIM integration |

---

**License:** MIT
**Requires:** NVIDIA API Key from [build.nvidia.com](https://build.nvidia.com/)
