#!/bin/bash
#===============================================================================
# NVIDIA NIM Inference Runner - DGX Spark ARM Architecture
#===============================================================================
#
# Runs visual and audio inference models on NVIDIA DGX Spark with ARM CPU.
# All inference is performed via NVIDIA NIM cloud APIs.
#
# Hardware Target: NVIDIA DGX Spark
#   - CPU: ARM-based (Grace CPU)
#   - GPU: NVIDIA GPU (for local acceleration if needed)
#   - Architecture: aarch64 (ARM64)
#
# Usage:
#   ./run_inference.sh                    # Interactive mode
#   ./run_inference.sh --webcam           # Webcam input
#   ./run_inference.sh --video file.mp4   # Video file
#   ./run_inference.sh --stream rtsp://   # RTSP stream
#
#===============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  NVIDIA NIM Inference - DGX Spark ARM Architecture${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

#-------------------------------------------------------------------------------
# Check Architecture
#-------------------------------------------------------------------------------
ARCH=$(uname -m)
echo -e "${YELLOW}Architecture:${NC} $ARCH"

if [[ "$ARCH" == "aarch64" ]] || [[ "$ARCH" == "arm64" ]]; then
    echo -e "${GREEN}✓ Running on ARM64 architecture (DGX Spark compatible)${NC}"
else
    echo -e "${YELLOW}⚠ Not running on ARM64 (detected: $ARCH)${NC}"
    echo -e "${YELLOW}  This script is optimized for NVIDIA DGX Spark ARM architecture${NC}"
    echo -e "${YELLOW}  Continuing anyway...${NC}"
fi

#-------------------------------------------------------------------------------
# Check NVIDIA API Key
#-------------------------------------------------------------------------------
if [[ -z "$NVIDIA_API_KEY" ]]; then
    echo ""
    echo -e "${RED}ERROR: NVIDIA_API_KEY environment variable not set${NC}"
    echo ""
    echo "To get your API key:"
    echo "  1. Go to https://build.nvidia.com/"
    echo "  2. Create account or sign in"
    echo "  3. Navigate to any model"
    echo "  4. Click 'Get API Key'"
    echo ""
    echo "Then set it:"
    echo "  export NVIDIA_API_KEY='nvapi-xxxx'"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ NVIDIA API Key configured${NC}"

#-------------------------------------------------------------------------------
# Check Python
#-------------------------------------------------------------------------------
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo -e "${RED}ERROR: Python not found${NC}"
    echo "Install Python 3.10+ and try again"
    exit 1
fi

PYTHON_VERSION=$($PYTHON --version 2>&1 | cut -d' ' -f2)
echo -e "${GREEN}✓ Python: $PYTHON_VERSION${NC}"

#-------------------------------------------------------------------------------
# Check Dependencies
#-------------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}Checking dependencies...${NC}"

check_package() {
    $PYTHON -c "import $1" 2>/dev/null && echo -e "  ${GREEN}✓${NC} $1" || {
        echo -e "  ${RED}✗${NC} $1 - installing..."
        pip install $2 --quiet
    }
}

check_package "cv2" "opencv-python"
check_package "numpy" "numpy"
check_package "requests" "requests"

echo ""

#-------------------------------------------------------------------------------
# Display Models
#-------------------------------------------------------------------------------
echo -e "${BLUE}NVIDIA NIM Models (Cloud Hosted):${NC}"
echo "  Visual:"
echo "    - microsoft/florence-2      (action detection)"
echo "    - nvidia/grounding-dino     (person detection)"
echo "    - nvidia/bodypose-estimation (pose analysis)"
echo "  Audio:"
echo "    - nvidia/parakeet-ctc-1.1b  (speech recognition)"
echo "    - nvidia/canary-1b          (multilingual ASR)"
echo "    - nvidia/audio-embedding    (sound classification)"
echo ""

#-------------------------------------------------------------------------------
# Parse Arguments
#-------------------------------------------------------------------------------
MODE="interactive"
VIDEO_SOURCE=""
ENABLE_AUDIO=false
NO_DISPLAY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --webcam)
            MODE="webcam"
            VIDEO_SOURCE="0"
            shift
            ;;
        --video)
            MODE="video"
            VIDEO_SOURCE="$2"
            shift 2
            ;;
        --stream)
            MODE="stream"
            VIDEO_SOURCE="$2"
            shift 2
            ;;
        --audio)
            ENABLE_AUDIO=true
            shift
            ;;
        --no-display)
            NO_DISPLAY=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --webcam         Use webcam (device 0)"
            echo "  --video FILE     Process video file"
            echo "  --stream URL     Process RTSP/HTTP stream"
            echo "  --audio          Enable audio analysis"
            echo "  --no-display     Headless mode (no GUI)"
            echo "  --help           Show this help"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

#-------------------------------------------------------------------------------
# Run Inference
#-------------------------------------------------------------------------------
echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}Starting inference...${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Build command
CMD="$PYTHON -u nim_inference_arm.py"

if [[ "$MODE" == "webcam" ]]; then
    CMD="$CMD --webcam"
elif [[ "$MODE" == "video" ]]; then
    CMD="$CMD --video '$VIDEO_SOURCE'"
elif [[ "$MODE" == "stream" ]]; then
    CMD="$CMD --stream '$VIDEO_SOURCE'"
fi

if [[ "$ENABLE_AUDIO" == true ]]; then
    CMD="$CMD --audio"
fi

if [[ "$NO_DISPLAY" == true ]]; then
    CMD="$CMD --no-display"
fi

# Execute
eval $CMD
