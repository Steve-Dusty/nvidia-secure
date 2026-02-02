# Nimverse Security Monitor

Real-time security surveillance system powered entirely by **NVIDIA technologies** for fall detection, fight detection, distress signal recognition, and emergency response dispatch.

## Quick Start

```bash
# 1. Clone and setup
git clone <repository-url>
cd nvidia-secure

# 2. Create environment file
cp .env.example .env
# Edit .env with your API keys (see Environment Variables section)

# 3. Install Python dependencies
pip install -r requirements.txt
pip install -r inference/requirements.txt

# 4. Start the backend
python webapp/backend.py

# 5. Start the web server (separate terminal)
cd webapp && npm install && npm start

# 6. Open in browser
# https://localhost:3000
```

## Tech Stack & Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              NIMVERSE ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                  │
│   │  Camera 1   │     │  Camera 2   │     │  Camera N   │                  │
│   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘                  │
│          │                   │                   │                          │
│          └───────────────────┼───────────────────┘                          │
│                              │                                              │
│                              ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                  NVIDIA NIM LOCAL INFERENCE (SELF-HOSTED)            │  │
│   ├─────────────────────────────────────────────────────────────────────┤  │
│   │                                                                      │  │
│   │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │  │
│   │  │   Florence-2     │  │  Grounding DINO  │  │   SAM2 Hiera     │  │  │
│   │  │  Scene Analysis  │  │ Person Detection │  │  Segmentation    │  │  │
│   │  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘  │  │
│   │           │                     │                     │             │  │
│   │           └─────────────────────┼─────────────────────┘             │  │
│   │                                 │                                    │  │
│   │                                 ▼                                    │  │
│   │           ┌─────────────────────────────────────────┐               │  │
│   │           │       NVIDIA BodyPose Estimation        │               │  │
│   │           │    17-point skeleton + action classify  │               │  │
│   │           └────────────────────┬────────────────────┘               │  │
│   │                                │                                     │  │
│   └────────────────────────────────┼─────────────────────────────────────┘  │
│                                    │                                        │
│   ┌────────────────────────────────┼─────────────────────────────────────┐  │
│   │                    NVIDIA NIM AUDIO PIPELINE                         │  │
│   ├────────────────────────────────┼─────────────────────────────────────┤  │
│   │                                │                                      │  │
│   │  ┌──────────────────┐  ┌──────┴───────────┐  ┌──────────────────┐   │  │
│   │  │ Parakeet CTC 1.1B│  │ Audio Embedding  │  │   Canary 1B      │   │  │
│   │  │  Speech-to-Text  │  │ Sound Classify   │  │ Multilingual ASR │   │  │
│   │  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘   │  │
│   │           │                     │                     │              │  │
│   │           └─────────────────────┼─────────────────────┘              │  │
│   │                                 │                                     │  │
│   └─────────────────────────────────┼─────────────────────────────────────┘  │
│                                     │                                        │
│                                     ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                   INCIDENT DETECTION ENGINE                          │   │
│   │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │   │
│   │  │  FALL   │  │  FIGHT  │  │ DISTRESS│  │  HELP   │  │  AUDIO  │   │   │
│   │  │  DETECT │  │  DETECT │  │  SIGNAL │  │  CALL   │  │  ALERT  │   │   │
│   │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘   │   │
│   │       └────────────┴────────────┴────────────┴────────────┘         │   │
│   │                                 │                                    │   │
│   └─────────────────────────────────┼────────────────────────────────────┘   │
│                                     │                                        │
│                                     ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │              NGC LLAMA 3 70B EMERGENCY DISPATCH                      │   │
│   │         Fine-tuned on 50K+ SF Medical Incident Records              │   │
│   │                                                                      │   │
│   │   Input: Incident details, location, severity                       │   │
│   │   Output: Optimal facility routing, ETA, resource allocation        │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### NVIDIA Technologies Used

| Component | NVIDIA Technology | Model/Service |
|-----------|------------------|---------------|
| Scene Understanding | NIM (Self-hosted) | Florence-2 |
| Person Detection | NIM (Self-hosted) | Grounding DINO |
| Segmentation | NIM (Self-hosted) | SAM2 Hiera Large |
| Pose Estimation | NIM (Self-hosted) | BodyPose Estimation |
| Speech Recognition | NIM (Self-hosted) | Parakeet CTC 1.1B |
| Multilingual ASR | NIM (Self-hosted) | Canary 1B |
| Sound Classification | NIM (Self-hosted) | Audio Embedding |
| Emergency Dispatch | NIM (Self-hosted) | Llama 3 70B (Fine-tuned) |
| Edge Deployment | DGX Spark | ARM-optimized inference |

## Environment Variables

Create a `.env` file in the project root:

```bash
# NVIDIA NIM API Key (required)
# Get from: https://build.nvidia.com/
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# NGC API Key (for Llama dispatch model)
# Get from: https://ngc.nvidia.com/
NGC_API_KEY=xxxxxxxxxxxxxxxxxxxx

# Mapbox (for map visualization - optional)
MAPBOX_API_KEY=pk.xxxxxxxxxxxxxxxxxxxxxxxx

# WebSocket Server
WS_HOST=0.0.0.0
WS_PORT=8765
```

### Getting API Keys

1. **NVIDIA NIM API Key**
   - Visit [NVIDIA AI Foundation](https://build.nvidia.com/)
   - Create account and generate API key
   - Free tier available for development

2. **NGC API Key**
   - Visit [NGC Catalog](https://ngc.nvidia.com/)
   - Navigate to Setup > API Key
   - Required for Llama dispatch agent

## Datasets & Synthetic Data

All datasets are sourced from **San Francisco Open Data Portal** ([data.sfgov.org](https://data.sfgov.org/)):

### Medical Incident Data
- **Source**: [SF Fire Department Calls for Service](https://data.sfgov.org/Public-Safety/Fire-Department-Calls-for-Service/nuek-vuh3)
- **Records**: 50,000+ medical incident records
- **Fields**: Call type, location, response time, priority, unit dispatch
- **Usage**: Training data for Llama emergency dispatch routing

### Health Facilities
- **Source**: [SF Health Care Facilities](https://data.sfgov.org/Health-and-Social-Services/Map-of-Health-Care-Facilities/6m8j-m5rs)
- **Records**: All hospitals, urgent care, clinics in SF
- **Fields**: Name, address, coordinates, facility type, capacity
- **Usage**: Optimal routing destinations for emergency dispatch

### Pharmacy Locations
- **Source**: [SF Registered Pharmacies](https://data.sfgov.org/Health-and-Social-Services/Registered-Pharmacies-in-San-Francisco/f4yy-5v6h)
- **Records**: All registered pharmacies in SF
- **Fields**: Name, address, coordinates, hours
- **Usage**: Non-emergency medical routing

### Synthetic Training Data
Located in `training-medresp/`:
- `nvidia_llama_complete_training.jsonl` - 50K+ training examples
- `routing_training.jsonl` - Facility routing training data
- `response_time_analysis.json` - Response time patterns

All synthetic data was generated using real SF incident patterns and facility data for realistic emergency dispatch training.

## Project Structure

```
nvidia-secure/
├── inference/                     # NVIDIA NIM inference (x86)
│   ├── nvidia_nim_visual.py       # Visual inference (Florence-2, DINO, SAM2)
│   ├── nvidia_nim_audio.py        # Audio inference (Parakeet, Canary)
│   ├── nvidia_nim_integrated.py   # Combined pipeline
│   └── main.py                    # Entry point
├── inference-arm/                 # DGX Spark ARM deployment
│   ├── Dockerfile.arm
│   ├── nim_inference_arm.py
│   └── setup_arm.sh
├── training-medresp/              # Emergency response training data
│   ├── generate_llama_training.py
│   ├── sf_medical_incidents.json
│   └── sf_health_facilities.json
├── agents/                        # Llama dispatch agent
│   └── emergency_response_agent.py
├── pipeline/                      # DeepStream detection pipeline
│   ├── sf_security_pipeline.py
│   ├── run_detection.py
│   └── download_models.sh
├── webapp/                        # Web dashboard
│   ├── backend.py                 # WebSocket server
│   ├── index.html                 # 9-camera grid
│   ├── demo.html                  # Live webcam demo
│   └── map.html                   # Geographic view
├── config/                        # DeepStream configs
├── .env.example                   # Environment template
└── README.md
```

## Running the Demo

### Option 1: Full System (Recommended)

```bash
# Terminal 1: Start NIM backend
python inference/main.py

# Terminal 2: Start web server
cd webapp && npm start

# Open https://localhost:3000
```

### Option 2: WebSocket Backend Only

```bash
# Start backend with NIM inference
python webapp/backend.py

# Open webapp/index.html directly in browser
```

### Option 3: Docker Deployment

```bash
docker-compose up -d
```

## Known Limitations

1. **Audio Streaming**: Current implementation sends audio in chunks. Continuous streaming would improve real-time ASR performance.

2. **Multi-Camera Scale**: WebSocket backend processes cameras sequentially. Parallel processing would improve throughput for 50+ cameras.

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- NVIDIA NIM for self-hosted inference containers
- NVIDIA NGC for model access
- San Francisco Open Data Portal for public datasets
