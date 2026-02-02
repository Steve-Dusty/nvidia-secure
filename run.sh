#!/bin/bash
# SF Security Camera - Run Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}SF Security Camera System${NC}"
echo -e "${GREEN}Fall & Fight Detection${NC}"
echo -e "${GREEN}================================${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker not installed${NC}"
    exit 1
fi

# Check NVIDIA Docker runtime
if ! docker info 2>/dev/null | grep -q nvidia; then
    echo -e "${YELLOW}Warning: NVIDIA runtime may not be configured${NC}"
fi

# Allow X11 display
xhost +local:docker 2>/dev/null || true

# Create required directories
mkdir -p recordings logs models

# Download models if not present
if [ ! -d "models/peoplenet" ]; then
    echo -e "${YELLOW}Downloading PeopleNet model...${NC}"
    echo "Please download models from NGC or use the download script"
fi

# Parse arguments
MODE="${1:-interactive}"
SOURCE="${2:-}"

case "$MODE" in
    "interactive"|"-i")
        echo -e "${GREEN}Starting in interactive mode...${NC}"
        docker run -it --rm \
            --gpus all \
            -e NVIDIA_DRIVER_CAPABILITIES=all \
            -e DISPLAY=$DISPLAY \
            -v /tmp/.X11-unix:/tmp/.X11-unix \
            -v "$SCRIPT_DIR/config:/app/config:ro" \
            -v "$SCRIPT_DIR/pipeline:/app/scripts:ro" \
            -v "$SCRIPT_DIR/models:/app/models" \
            -v "$SCRIPT_DIR/recordings:/app/recordings" \
            -v "$SCRIPT_DIR/logs:/app/logs" \
            --network host \
            --privileged \
            nvcr.io/nvidia/deepstream:7.1-samples-multiarch \
            bash
        ;;

    "run"|"-r")
        echo -e "${GREEN}Starting detection pipeline...${NC}"
        if [ -n "$SOURCE" ]; then
            SOURCES="$SOURCE"
        else
            SOURCES="/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"
        fi
        docker run -it --rm \
            --gpus all \
            -e NVIDIA_DRIVER_CAPABILITIES=all \
            -e DISPLAY=$DISPLAY \
            -v /tmp/.X11-unix:/tmp/.X11-unix \
            -v "$SCRIPT_DIR/config:/app/config:ro" \
            -v "$SCRIPT_DIR/pipeline:/app/scripts:ro" \
            -v "$SCRIPT_DIR/models:/app/models" \
            -v "$SCRIPT_DIR/recordings:/app/recordings" \
            -v "$SCRIPT_DIR/logs:/app/logs" \
            --network host \
            --privileged \
            nvcr.io/nvidia/deepstream:7.1-samples-multiarch \
            python3 /app/scripts/sf_security_pipeline.py "$SOURCES"
        ;;

    "compose"|"-c")
        echo -e "${GREEN}Starting with docker-compose...${NC}"
        docker-compose up -d
        echo -e "${GREEN}Services started. View logs with: docker-compose logs -f${NC}"
        ;;

    "test"|"-t")
        echo -e "${GREEN}Running test with sample video...${NC}"
        docker run -it --rm \
            --gpus all \
            -e NVIDIA_DRIVER_CAPABILITIES=all \
            -e DISPLAY=$DISPLAY \
            -v /tmp/.X11-unix:/tmp/.X11-unix \
            -v "$SCRIPT_DIR/config:/app/config:ro" \
            -v "$SCRIPT_DIR/pipeline:/app/scripts:ro" \
            -v "$SCRIPT_DIR/models:/app/models" \
            -v "$SCRIPT_DIR/recordings:/app/recordings" \
            -v "$SCRIPT_DIR/logs:/app/logs" \
            --network host \
            --privileged \
            nvcr.io/nvidia/deepstream:7.1-samples-multiarch \
            deepstream-app -c /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/source1_usb_dec_infer_resnet_int8.txt
        ;;

    *)
        echo "Usage: $0 [mode] [source]"
        echo ""
        echo "Modes:"
        echo "  interactive, -i  Start interactive shell in container"
        echo "  run, -r          Run detection pipeline"
        echo "  compose, -c      Start with docker-compose"
        echo "  test, -t         Run test with sample video"
        echo ""
        echo "Examples:"
        echo "  $0 run rtsp://192.168.1.100:554/stream"
        echo "  $0 run /path/to/video.mp4"
        echo "  $0 interactive"
        ;;
esac
