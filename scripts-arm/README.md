# NVIDIA NIM Inference - ARM Architecture (DGX Spark)

**Optimized inference runner for NVIDIA DGX Spark with ARM (Grace) CPU.**

All model inference runs on NVIDIA NIM cloud APIs - no local GPU inference required. The ARM optimizations focus on efficient frame processing and API communication.

---

## Hardware Target

| Component | Specification |
|-----------|---------------|
| **Platform** | NVIDIA DGX Spark |
| **CPU** | NVIDIA Grace (ARM64) |
| **Architecture** | aarch64 |
| **GPU** | NVIDIA GPU (for display/encoding) |

---

## Quick Start

```bash
# 1. Set API key
export NVIDIA_API_KEY="nvapi-xxxx"

# 2. Setup environment
./setup_arm.sh

# 3. Run inference
./run_inference.sh --webcam
```

---

## Files

| File | Purpose |
|------|---------|
| `run_inference.sh` | Main runner script |
| `setup_arm.sh` | Environment setup for ARM |
| `nim_inference_arm.py` | ARM-optimized inference runner |
| `nvidia_nim_visual.py` | Visual inference (Florence-2, DINO, BodyPose) |
| `nvidia_nim_audio.py` | Audio inference (Parakeet, Audio Embedding) |
| `nvidia_nim_integrated.py` | Combined visual + audio analysis |
| `Dockerfile.arm` | Docker image for ARM64 deployment |
| `requirements_arm.txt` | ARM-optimized Python dependencies |

---

## NVIDIA NIM Models

### Visual Detection

| Model | Purpose | API Endpoint |
|-------|---------|--------------|
| `microsoft/florence-2` | Action detection | `/vlm/microsoft/florence-2` |
| `nvidia/grounding-dino` | Person detection | `/vlm/nvidia/grounding-dino` |
| `nvidia/bodypose-estimation` | Pose analysis | `/cv/nvidia/bodypose-estimation` |

### Audio Detection

| Model | Purpose | API Endpoint |
|-------|---------|--------------|
| `nvidia/parakeet-ctc-1.1b` | Speech recognition | `/asr/nvidia/parakeet-ctc-1.1b` |
| `nvidia/canary-1b` | Multilingual ASR | `/asr/nvidia/canary-1b` |
| `nvidia/audio-embedding` | Sound classification | `/audio/nvidia/audio-embedding` |

---

## Detection Capabilities

### Visual Actions
- Standing
- Walking
- Running
- Fighting
- Falling
- Lying down

### Audio Events
- "Help" calls and distress speech
- Coughing
- Screaming
- Fighting sounds
- Chaos/crowd noise

---

## Usage

### Command Line

```bash
# Webcam
./run_inference.sh --webcam

# Video file
./run_inference.sh --video /path/to/video.mp4

# RTSP stream
./run_inference.sh --stream rtsp://camera/stream

# With audio analysis
./run_inference.sh --webcam --audio

# Headless (no display)
./run_inference.sh --video file.mp4 --no-display
```

### Python API

```python
from nim_inference_arm import ARMInferenceRunner

runner = ARMInferenceRunner(enable_audio=True)
runner.process_video(source=0, display=True, skip_frames=5)
```

### Docker

```bash
# Build
docker build -t nvidia-nim-arm:latest -f Dockerfile.arm .

# Run
docker run -e NVIDIA_API_KEY=$NVIDIA_API_KEY nvidia-nim-arm:latest --video /data/video.mp4

# With webcam
docker run --device=/dev/video0 \
    -e NVIDIA_API_KEY=$NVIDIA_API_KEY \
    nvidia-nim-arm:latest --webcam
```

---

## Setup

### Option 1: Native Installation

```bash
# Run setup script
./setup_arm.sh

# Activate environment
source venv/bin/activate

# Verify
python -c "import cv2, numpy, requests; print('OK')"
```

### Option 2: Docker

```bash
# Build ARM64 image
./setup_arm.sh --docker

# Or manually
docker build -t nvidia-nim-arm:latest -f Dockerfile.arm .
```

---

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NVIDIA_API_KEY` | Yes | NIM API key from build.nvidia.com |
| `DISPLAY` | No | X11 display for GUI (empty for headless) |

### Performance Tuning

```bash
# Skip more frames for faster processing
./run_inference.sh --webcam --skip-frames 10

# Headless for maximum throughput
./run_inference.sh --video file.mp4 --no-display --skip-frames 3
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DGX Spark (ARM64)                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Camera/    │    │  ARM CPU     │    │   NVIDIA     │  │
│  │   Video      │───>│  (Grace)     │───>│   NIM API    │  │
│  │   Input      │    │  Processing  │    │   (Cloud)    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                             │                    │          │
│                             │                    │          │
│                             v                    v          │
│                      ┌──────────────┐    ┌──────────────┐  │
│                      │   Frame      │    │   Florence   │  │
│                      │   Encoding   │    │   DINO       │  │
│                      │   (OpenCV)   │    │   BodyPose   │  │
│                      └──────────────┘    └──────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              v
                    ┌──────────────────┐
                    │  Alert/Dispatch  │
                    │  (Llama - NGC)   │
                    └──────────────────┘
```

---

## Performance

### Expected Latency

| Operation | Latency |
|-----------|---------|
| Frame encoding | ~5ms |
| API round-trip | ~100-300ms |
| Total per frame | ~150-350ms |

### Throughput

| Mode | Effective FPS |
|------|---------------|
| Skip 5 frames | ~6 FPS |
| Skip 10 frames | ~3 FPS |
| Headless + skip 3 | ~10 FPS |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `NVIDIA_API_KEY not set` | `export NVIDIA_API_KEY="nvapi-xxxx"` |
| OpenCV import error | `pip install opencv-python-headless` |
| Connection timeout | Check network, increase timeout in code |
| Low FPS | Increase `--skip-frames` value |
| Display not working | Use `--no-display` for headless |

---

## API Key

Get your NVIDIA API key:
1. Go to [build.nvidia.com](https://build.nvidia.com/)
2. Create account or sign in
3. Navigate to any model
4. Click "Get API Key"

---

## License

MIT License

**Requires:** NVIDIA API Key from [build.nvidia.com](https://build.nvidia.com/)
