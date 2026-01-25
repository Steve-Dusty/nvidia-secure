#!/bin/bash
#===============================================================================
# NVIDIA NIM Setup - DGX Spark ARM Architecture
#===============================================================================
#
# Sets up the environment for running NVIDIA NIM inference on ARM (DGX Spark).
# Installs Python dependencies optimized for ARM64 architecture.
#
# Usage:
#   ./setup_arm.sh
#   ./setup_arm.sh --docker   # Build Docker container instead
#
#===============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  NVIDIA NIM Setup - DGX Spark ARM Architecture${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

#-------------------------------------------------------------------------------
# Check Architecture
#-------------------------------------------------------------------------------
ARCH=$(uname -m)
echo -e "${YELLOW}Detected architecture:${NC} $ARCH"

if [[ "$ARCH" == "aarch64" ]] || [[ "$ARCH" == "arm64" ]]; then
    echo -e "${GREEN}✓ ARM64 architecture detected${NC}"
    IS_ARM=true
else
    echo -e "${YELLOW}⚠ Not ARM64 (detected: $ARCH)${NC}"
    echo -e "${YELLOW}  Installing for current architecture anyway...${NC}"
    IS_ARM=false
fi

#-------------------------------------------------------------------------------
# Parse Arguments
#-------------------------------------------------------------------------------
BUILD_DOCKER=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --docker)
            BUILD_DOCKER=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --docker    Build Docker container for deployment"
            echo "  --help      Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

#-------------------------------------------------------------------------------
# Docker Build
#-------------------------------------------------------------------------------
if [[ "$BUILD_DOCKER" == true ]]; then
    echo ""
    echo -e "${BLUE}Building Docker container for ARM64...${NC}"
    echo ""

    docker build -t nvidia-nim-arm:latest -f Dockerfile.arm .

    echo ""
    echo -e "${GREEN}✓ Docker image built: nvidia-nim-arm:latest${NC}"
    echo ""
    echo "To run:"
    echo "  docker run -e NVIDIA_API_KEY=\$NVIDIA_API_KEY nvidia-nim-arm:latest"
    echo ""
    exit 0
fi

#-------------------------------------------------------------------------------
# Python Setup
#-------------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}Setting up Python environment...${NC}"

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo -e "${RED}ERROR: Python not found${NC}"
    echo "Please install Python 3.10+"
    exit 1
fi

PYTHON_VERSION=$($PYTHON --version 2>&1)
echo -e "${GREEN}✓ $PYTHON_VERSION${NC}"

# Create virtual environment (optional)
if [[ ! -d "venv" ]]; then
    echo ""
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    $PYTHON -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo ""
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Activated: $(which python)${NC}"

#-------------------------------------------------------------------------------
# Install Dependencies
#-------------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}Installing dependencies...${NC}"

# Upgrade pip
pip install --upgrade pip --quiet

# Install ARM-optimized packages
if [[ "$IS_ARM" == true ]]; then
    echo "Installing ARM64-optimized packages..."

    # OpenCV for ARM
    pip install opencv-python-headless --quiet 2>/dev/null || \
    pip install opencv-python --quiet

    # NumPy (will use ARM NEON optimizations)
    pip install numpy --quiet

else
    # Standard packages for x86
    pip install opencv-python --quiet
    pip install numpy --quiet
fi

# Common packages
pip install requests --quiet
pip install python-dotenv --quiet

echo -e "${GREEN}✓ Dependencies installed${NC}"

#-------------------------------------------------------------------------------
# Verify Installation
#-------------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}Verifying installation...${NC}"

$PYTHON << 'EOF'
import sys

packages = [
    ("cv2", "opencv"),
    ("numpy", "numpy"),
    ("requests", "requests"),
]

all_ok = True
for module, name in packages:
    try:
        __import__(module)
        print(f"  ✓ {name}")
    except ImportError:
        print(f"  ✗ {name} - FAILED")
        all_ok = False

if not all_ok:
    sys.exit(1)
EOF

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}✓ All packages verified${NC}"
else
    echo -e "${RED}✗ Some packages failed to install${NC}"
    exit 1
fi

#-------------------------------------------------------------------------------
# Check NVIDIA API Key
#-------------------------------------------------------------------------------
echo ""
if [[ -z "$NVIDIA_API_KEY" ]]; then
    echo -e "${YELLOW}⚠ NVIDIA_API_KEY not set${NC}"
    echo ""
    echo "To use NVIDIA NIM, set your API key:"
    echo "  export NVIDIA_API_KEY='nvapi-xxxx'"
    echo ""
    echo "Get your key from: https://build.nvidia.com/"
else
    echo -e "${GREEN}✓ NVIDIA_API_KEY configured${NC}"
fi

#-------------------------------------------------------------------------------
# Summary
#-------------------------------------------------------------------------------
echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run inference:"
echo "  ./run_inference.sh --webcam"
echo "  ./run_inference.sh --video path/to/video.mp4"
echo ""
echo "Make sure to set NVIDIA_API_KEY before running."
echo ""
