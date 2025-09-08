#!/bin/bash
set -e

CUDA_VERSION=${1:-"12.8"}
TORCH_VERSION=${2:-"2.8.0"}
MAX_JOBS=${3:-"4"}

echo "Building Flash-Attention 3 wheel:"
echo "CUDA Version: $CUDA_VERSION"
echo "PyTorch Version: $TORCH_VERSION"
echo "Max Jobs: $MAX_JOBS"

export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export MAX_JOBS=${MAX_JOBS}
export FLASH_ATTENTION_FORCE_BUILD=TRUE

echo "Installing dependencies..."
pip install --upgrade pip
pip install ninja packaging wheel setuptools numpy change_wheel_version
pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION//./}

WORK_DIR=$(mktemp -d)
cd $WORK_DIR
git clone --recursive https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper

GIT_HASH=$(git rev-parse --short=6 HEAD)
echo "Current git hash: $GIT_HASH"

echo "Building Flash-Attention 3 wheel..."
python setup.py bdist_wheel

ORIGINAL_WHEEL=$(find dist -name "*.whl" | head -1)
if [ -z "$ORIGINAL_WHEEL" ]; then
    echo "Error: Wheel file not found"
    exit 1
fi

echo "Original wheel built: $ORIGINAL_WHEEL"

BUILD_DATE=$(date +%Y%m%d)
echo "Build date: $BUILD_DATE"

CXX11_ABI="FALSE"
if python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)" 2>/dev/null | grep -q "True"; then
    CXX11_ABI="TRUE"
fi

CUDA_VERSION_CLEAN=$(echo $CUDA_VERSION | tr -d '.')
TORCH_VERSION_CLEAN=$(echo $TORCH_VERSION | tr -d '.')

LOCAL_VERSION="${BUILD_DATE}.cu${CUDA_VERSION_CLEAN}torch${TORCH_VERSION_CLEAN}cxx11abi${CXX11_ABI}.${GIT_HASH}"

echo "Local version identifier: $LOCAL_VERSION"

echo "Modifying wheel with local version..."
MODIFIED_WHEEL=$(python -m change_wheel_version "$ORIGINAL_WHEEL" --local-version "$LOCAL_VERSION" --delete-old-wheel)

if [ -z "$MODIFIED_WHEEL" ] || [ ! -f "$MODIFIED_WHEEL" ]; then
    echo "Error: Failed to modify wheel version"
    exit 1
fi

echo "Modified wheel created: $MODIFIED_WHEEL"

WHEEL_NAME=$(basename $MODIFIED_WHEEL)
OUTPUT_DIR="/tmp/wheels"
mkdir -p $OUTPUT_DIR
cp dist/$WHEEL_NAME $OUTPUT_DIR/

echo "Wheel saved to: $OUTPUT_DIR/$WHEEL_NAME"
echo "build_success=true" >> $GITHUB_OUTPUT
echo "wheel_path=$OUTPUT_DIR/$WHEEL_NAME" >> $GITHUB_OUTPUT
echo "wheel_name=$WHEEL_NAME" >> $GITHUB_OUTPUT
