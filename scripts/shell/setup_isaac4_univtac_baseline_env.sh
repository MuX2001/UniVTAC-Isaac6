#!/usr/bin/env bash
# Create an isolated Isaac Sim 4.5 UniVTAC environment. It deliberately does
# not install FTP1/openpi or any task policy, so it cannot alter an Isaac Sim 6
# installation or accidentally run policy inference.
set -euo pipefail

FTP1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
UNIVTAC_ROOT="$FTP1_ROOT/UniVTAC"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-univtac-isaac4-baseline}"
ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/IsaacLab-2.1.1}"
CMAKE_TOOLCHAIN_FILE="${CMAKE_TOOLCHAIN_FILE:-$HOME/Toolchain/vcpkg/scripts/buildsystems/vcpkg.cmake}"

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda is required to create $CONDA_ENV_NAME." >&2
    exit 1
fi
if [[ ! -d "$ISAACLAB_PATH" || ! -x "$ISAACLAB_PATH/isaaclab.sh" ]]; then
    echo "[ERROR] Isaac Lab 2.1.1 is required at ISAACLAB_PATH=$ISAACLAB_PATH." >&2
    echo "        Clone NVIDIA IsaacLab and check out tag v2.1.1 before rerunning." >&2
    exit 1
fi
if [[ ! -f "$CMAKE_TOOLCHAIN_FILE" ]]; then
    echo "[ERROR] TacEx UIPC requires vcpkg's CMake toolchain." >&2
    echo "        Set CMAKE_TOOLCHAIN_FILE to .../vcpkg/scripts/buildsystems/vcpkg.cmake." >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV_NAME"; then
    conda create -n "$CONDA_ENV_NAME" python=3.10 -y
fi
conda activate "$CONDA_ENV_NAME"

python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

git -C "$ISAACLAB_PATH" fetch --tags
git -C "$ISAACLAB_PATH" checkout v2.1.1

# Do not use `isaaclab.sh --install`: it forcibly upgrades PyTorch to the
# CUDA-12.8 build and installs optional RL/Mimic packages.  Isaac Sim 4.5
# requires the CUDA-11.8 PyTorch version installed above, and UniVTAC only uses
# Isaac Lab's core APIs.
python -m pip install \
    "numpy<2" \
    onnx==1.16.1 \
    prettytable==3.3.0 \
    toml \
    hidapi==0.14.0.post2 \
    gymnasium==1.2.0 \
    trimesh \
    "pyglet<2" \
    transformers \
    einops \
    warp-lang \
    pillow==11.2.1 \
    starlette==0.45.3 \
    pytest \
    pytest-mock \
    junitparser \
    flatdict==4.1.0 \
    pin-pink==3.1.0 \
    dex-retargeting==0.4.6
python -m pip install --no-deps -e "$ISAACLAB_PATH/source/isaaclab"

export ISAACLAB_PATH
export TACEX_PATH="$UNIVTAC_ROOT/third_party/TacEx"
export CMAKE_TOOLCHAIN_FILE
export VCPKG_ROOT="${VCPKG_ROOT:-$(cd "$(dirname "$CMAKE_TOOLCHAIN_FILE")/../.." && pwd)}"

# TacEx UIPC compiles a CUDA extension.  Keep the compiler inside this Conda
# environment so neither the desktop CUDA installation nor Isaac Sim 6 changes.
if [[ ! -x "$CONDA_PREFIX/bin/nvcc" ]]; then
    echo "[ERROR] CUDA nvcc is required for TacEx UIPC. Install it in this env:" >&2
    echo "        conda install -n $CONDA_ENV_NAME -c nvidia/label/cuda-11.8.0 cuda-nvcc" >&2
    exit 1
fi

# The PyTorch CUDA-11.8 wheel already supplies matching CUDART headers/libs. A
# standalone conda nvcc package omits those development files, so expose them to
# nvcc without installing a system-wide CUDA toolkit.
PYTHON_SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
PYTORCH_CUDART_ROOT="$PYTHON_SITE_PACKAGES/nvidia/cuda_runtime"
if [[ ! -f "$CONDA_PREFIX/include/cuda_runtime.h" && -f "$PYTORCH_CUDART_ROOT/include/cuda_runtime.h" ]]; then
    mkdir -p "$CONDA_PREFIX/include" "$CONDA_PREFIX/lib"
    for header in "$PYTORCH_CUDART_ROOT/include"/*; do
        ln -sfn "$header" "$CONDA_PREFIX/include/$(basename "$header")"
    done
    ln -sfn "$PYTORCH_CUDART_ROOT/lib/libcudart.so.11.0" "$CONDA_PREFIX/lib/libcudart.so.11.0"
    ln -sfn "$PYTORCH_CUDART_ROOT/lib/libcudart.so.11.0" "$CONDA_PREFIX/lib/libcudart.so"
fi

# The PyTorch dependency bundle also carries the static CUDA runtime needed by
# nvcc's compiler-link test.  Use it only if a complete CUDA development
# toolkit is not already installed in this Conda environment.
PYTORCH_CUDA13_LIB="$PYTHON_SITE_PACKAGES/nvidia/cu13/lib"
if [[ ! -f "$CONDA_PREFIX/lib/libcudadevrt.a" && -f "$PYTORCH_CUDA13_LIB/libcudadevrt.a" ]]; then
    ln -sfn "$PYTORCH_CUDA13_LIB/libcudadevrt.a" "$CONDA_PREFIX/lib/libcudadevrt.a"
    ln -sfn "$PYTORCH_CUDA13_LIB/libcudart_static.a" "$CONDA_PREFIX/lib/libcudart_static.a"
fi
export CUDAToolkit_ROOT="$CONDA_PREFIX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export PATH="$CONDA_PREFIX/bin:$PATH"

# CUDA 11.8 does not support the system GCC 13 host compiler.  Ubuntu's
# GCC/G++ 11 pair is installed alongside it and is used only for TacEx's
# native CUDA extension build.
if [[ -x /usr/bin/gcc-11 && -x /usr/bin/g++-11 ]]; then
    export CC=/usr/bin/gcc-11
    export CXX=/usr/bin/g++-11
    export CUDAHOSTCXX=/usr/bin/g++-11
fi

git -C "$VCPKG_ROOT" pull --ff-only
bash "$TACEX_PATH/tacex.sh" -i all

python - <<'PY'
import isaacsim
import isaaclab
import tacex_uipc

print("Isaac Sim 4 / Isaac Lab / TacEx import check passed")
PY

echo "[setup] Ready. Activate with: conda activate $CONDA_ENV_NAME"
echo "[setup] Then run: cd $UNIVTAC_ROOT && bash scripts/shell/run_isaac4_raw_baseline.sh <gpu_id>"
