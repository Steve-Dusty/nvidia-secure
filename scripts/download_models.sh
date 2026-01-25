#!/bin/bash
# Download pre-trained models for SF Security Camera
# Requires NGC API key

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../models"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Model Download Script${NC}"
echo -e "${GREEN}================================${NC}"

# Check for NGC CLI
if ! command -v ngc &> /dev/null; then
    echo -e "${YELLOW}NGC CLI not found. Installing...${NC}"
    pip3 install --upgrade nvidia-ngc-cli
fi

# Check for NGC API key
if [ -z "$NGC_API_KEY" ]; then
    echo -e "${RED}NGC_API_KEY environment variable not set${NC}"
    echo ""
    echo "To get an NGC API key:"
    echo "1. Go to https://ngc.nvidia.com"
    echo "2. Sign in or create a free account"
    echo "3. Go to Setup > API Key"
    echo "4. Generate and copy your API key"
    echo ""
    echo "Then run:"
    echo "  export NGC_API_KEY='your_api_key_here'"
    echo "  $0"
    echo ""
    echo "Or create a file at ~/.ngc/config with:"
    echo "[NVIDIA]"
    echo "apikey = your_api_key_here"
    exit 1
fi

# Configure NGC
ngc config set --apikey "$NGC_API_KEY"

mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

echo -e "${GREEN}Downloading PeopleNet (Person Detection)...${NC}"
mkdir -p peoplenet
ngc registry model download-version nvidia/tao/peoplenet:pruned_quantized_decrypted_v2.6.3 \
    --dest peoplenet 2>/dev/null || \
    echo -e "${YELLOW}PeopleNet download requires manual download from NGC${NC}"

# Create labels file
cat > peoplenet/labels.txt << 'EOF'
person
bag
face
EOF

echo -e "${GREEN}Downloading BodyPose (Pose Estimation)...${NC}"
mkdir -p bodypose
ngc registry model download-version nvidia/tao/bodyposenet:deployable_accuracy_v1.0.1 \
    --dest bodypose 2>/dev/null || \
    echo -e "${YELLOW}BodyPose download requires manual download from NGC${NC}"

echo -e "${GREEN}Downloading ActionRecognitionNet...${NC}"
mkdir -p actionrecognition
ngc registry model download-version nvidia/tao/actionrecognitionnet:trainable_v2.0 \
    --dest actionrecognition 2>/dev/null || \
    echo -e "${YELLOW}ActionRecognition download requires manual download from NGC${NC}"

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Model download complete!${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "Models are stored in: $MODELS_DIR"
ls -la "$MODELS_DIR"

echo ""
echo -e "${YELLOW}Note: If downloads failed, manually download from:${NC}"
echo "  https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet"
echo "  https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/bodyposenet"
echo "  https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/actionrecognitionnet"
