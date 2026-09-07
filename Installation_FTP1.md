# UniVTAC Simulator and FTP1 Runtime Guide

This is the index for the runtime documentation in this fork. It documents the provided Docker images used for simulator debugging and FTP1 evaluation, followed by the retained legacy Conda build instructions.

For the validated Isaac Sim 6 migration, use the [Isaac Sim 6 Migration and Runtime Guide](./docs/IsaacSim6_Migration_and_Runtime.md). It contains the safe resource guard, portable-image procedure, verified task status, and operational commands. The [Isaac Sim 6 Validation Comparison](./docs/IsaacSim6_Validation_Comparison.md) is the evidence-oriented companion for review meetings.

## Current Docker runtime

Use the provided images instead of rebuilding the simulator stack. The Docker build context that produced these images is not tracked in this repository. `third_party/TacEx/docker/` is a legacy Isaac Sim 4.5 / Isaac Lab 2.1.1 recipe and does not reproduce the current Isaac Sim 6 runtime.

| Image | Use | Included runtime |
| --- | --- | --- |
| `user10/univtac-isaac60-lab3-tacex:curobo-permissions-prettytable` | Earlier simulator-only debugging image | Isaac Sim 6.0 runtime, Python 3.12.13, Isaac Lab 6.1.17, PyTorch 2.10.0+cu128, TacEx/UIPC, and cuRobo |
| `user10/univtac-isaac60-lab3-tacex:ftp1-pytorch-runtime` | Earlier FTP1-runtime layer | Simulator stack plus FTP1 runtime dependencies |
| `user10/univtac-isaac60-lab3-tacex:ftp1-pytorch-runtime-fem-tactile-curobo-sm120-uipc-sm120-resetcache` | Validated development base | The pinned Isaac Sim 6 runtime used by the migration validation |
| `univtac-isaac60-lab3-tacex:ftp1-migrated-20260903-r2` | Portable migrated simulator | The validated base plus the curated UniVTAC source and the local TacEx package sources; created by `scripts/shell/export_migrated_docker_image.sh` |

The earlier images are packaged in `univtac-simulator-and-ftp1-images.tar`. The portable migrated image is exported separately as `univtac-isaacsim6-migrated-20260903-r2.tar`; it is self-contained for the September 3 simulator snapshot but intentionally omits local validation outputs, published HDF files, FTP1 checkpoints, and the root `openpi` source. The September 7 camera and recorder fixes are in Git, so use the current checkout with the source mounts below instead of exporting another identical 25.6 GB runtime. Load the offline image on a target machine with:

```bash
docker load -i univtac-isaacsim6-migrated-20260903-r2.tar
```

The host needs a compatible NVIDIA driver, Docker, and the NVIDIA Container Toolkit. The image declares a minimum driver version of 570.169.

After cloning either the standalone simulator repository or the full policy repository, the recommended bounded acceptance is:

```bash
cd <UniVTAC-checkout>
UNIVTAC_IMAGE='univtac-isaac60-lab3-tacex:ftp1-migrated-20260903-r2' \
  bash scripts/shell/run_guarded_validation.sh lift_can 41 episode target-r1
```

Omit `UNIVTAC_IMAGE` when the pinned development image is already available. The launcher performs the three required source mounts, preflights current GPU use, stops at the validated VRAM threshold, writes to `runtime-output/`, and retains the stopped container for inspection.

### Debug the simulator without FTP1

For a development checkout, mount `UniVTAC/` and both edited TacEx package roots. The image has no bare `python` command; invoke the Isaac Sim interpreter explicitly. Use the resource guard in the Isaac Sim 6 migration guide rather than the unguarded historical example below.

```bash
cd <repository-root>
docker run --name univtac-dev-shell -it \
  --gpus all --memory=15g --memory-swap=15g --cpus=6 --pids-limit=512 \
  --ulimit core=0 --shm-size=64m --user 1234:1000 \
  -v "<repository-root>/UniVTAC:/workspace/UniVTAC" \
  -v "<repository-root>/UniVTAC/third_party/TacEx/source/tacex/tacex:/opt/tacex/source/tacex/tacex" \
  -v "<repository-root>/UniVTAC/third_party/TacEx/source/tacex_uipc/tacex_uipc:/opt/tacex/source/tacex_uipc/tacex_uipc" \
  -w /workspace/UniVTAC --entrypoint /bin/bash \
  user10/univtac-isaac60-lab3-tacex:ftp1-pytorch-runtime-fem-tactile-curobo-sm120-uipc-sm120-resetcache

/isaac-sim/python.sh scripts/collect_data.py --help
```

The image embeds TacEx at `/opt/tacex` and cuRobo at `/opt/curobo`. The validated local migration deliberately overlays the two edited TacEx **package** roots under `/opt/tacex`; see the migration guide for the exact mounts. This is required for the local Isaac Sim 6 compatibility fixes and does not rebuild or replace the simulator stack.

### Run FTP1 in the simulator

The FTP1 runtime image contains dependencies but intentionally does not contain this repository `openpi` source. Mount the full repository, expose both source roots through `PYTHONPATH`, and separately mount or copy the checkpoint directory.

```bash
cd <repository-root>
docker run --name univtac-ftp1-eval \
  --gpus all --memory=15g --memory-swap=15g --cpus=6 --pids-limit=512 \
  --ulimit core=0 --shm-size=64m --user 1234:1000 \
  -v "<repository-root>:/workspace" -v "<checkpoint-parent>:/checkpoints:ro" \
  -w /workspace/UniVTAC -e PYTHONPATH=/workspace/src:/workspace/packages/openpi-client/src \
  --entrypoint /isaac-sim/python.sh \
  user10/univtac-isaac60-lab3-tacex:ftp1-pytorch-runtime \
  scripts/eval_ftp1.py --checkpoint_dir /checkpoints/<experiment>/<step> --domain_name <domain-name> --task_list insert_HDMI --task_config contact.yml --total_num 1
```

The evaluator launches Isaac Lab headlessly with cameras enabled, the headless rendering kit, no livestream, and `--reset-user`. Start with one worker; each additional worker starts a separate Isaac Sim process. Use `--low_memory` to place FTP1 inference on CPU when the GPU cannot hold both the model and simulator.

### Fork-specific dependency and loader changes

- The full policy repository requires Python `>=3.11`; the current Docker runtime uses Python 3.12. The Python 3.10 environment below belongs only to the historical Isaac Sim 4.5 build and is not a reason to lower the policy project's requirement.
- The Docker runtime owns its Isaac Sim and PyTorch stack. Do not run a normal dependency-resolving `pip install -e .` inside it, because it can replace simulator-pinned packages. The legacy installer uses `pip install --no-deps -e` for `openpi` and `openpi-client`.
- FTP1 checkpoint weights are read and applied on CPU, then the complete model is moved once to its requested device. This avoids very slow direct safetensors-to-CUDA loading in the Isaac Sim runtime. Set `FTP1_LOAD_DIAGNOSTICS=1` to emit load-stage timing, RSS, and allocated-VRAM telemetry.
- The root `uv.lock` records the policy development environment; it is not the mechanism for changing Docker runtime packages.

## Legacy Conda build (Isaac Sim 4.5 / Isaac Lab 2.1.1)

The remaining sections document the older source-based setup. Use them only when rebuilding the legacy Conda environment; they are not compatible with the current Docker images described above.

UniVTAC depends on the following major components:

- Linux with an NVIDIA GPU
- Python 3.10
- NVIDIA Isaac Sim `4.5.0`
- Isaac Lab `2.1.1`
- cuRobo
- TacEx from the local modified source at `UniVTAC/third_party/TacEx`
- FTP1 inference integrated into the Isaac Sim environment

The recommended setup is to use one Conda environment for:

- Isaac Sim
- Isaac Lab
- TacEx
- FTP1 inference

This guide assumes the environment name is `uni`.

## Prerequisites

Before starting, make sure you have:

- a Linux machine with NVIDIA drivers installed
- `conda`
- `git`
- `gcc` and `g++`
- enough disk space for Isaac Sim, Isaac Lab, TacEx build artifacts, and model dependencies

## Step 0: Remove an old environment if needed

If you want to rebuild the environment from scratch:

```bash
conda env remove -n uni
```

## Step 1: Clone the repository


Clone the repository:

```bash
# git clone https://github.com/byml-c/UniVTAC.git
cd UniVTAC
```

## Step 2: Create the Conda environment

Create and activate the recommended shared environment:

```bash
CONDA_SOLVER=classic conda create -n uni python=3.10 -y
conda activate uni
```

All remaining steps in this guide should be executed inside:

```bash
conda activate uni
```

## Step 3: Install Isaac Sim 4.5.0

Install CUDA-enabled PyTorch first:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade pip
```

Install system packages required by Isaac Sim:

```bash
sudo apt-get update
sudo apt-get install gcc-11 g++-11
```

Install Isaac Sim:

```bash
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
```

Verify that Isaac Sim launches:

```bash
isaacsim --allow-root --headless
```

## Step 3.1: Fix Vulkan issues if Isaac Sim fails to start

On some machines, Isaac Sim fails because of Vulkan configuration issues. If that happens, install the Vulkan-related packages first:

```bash
sudo apt-get update
sudo apt-get install -y libvulkan1 vulkan-tools
sudo apt-get install -y libnvidia-gl-$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1) 2>/dev/null || true
sudo apt-get install -y nvidia-vulkan-icd 2>/dev/null || true
```

You can inspect possible NVIDIA Vulkan libraries with:

```bash
find /usr -name "*nvidia*vulkan*" 2>/dev/null
ls /usr/lib/x86_64-linux-gnu/libnvidia-glcore.so* 2>/dev/null
```

If needed, create an NVIDIA ICD file:

```bash
sudo mkdir -p /usr/share/vulkan/icd.d
echo '{"file_format_version": "1.0.0", "ICD": {"library_path": "/usr/lib/x86_64-linux-gnu/libnvidia-glcore.so.$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)", "api_version": "1.1.0"}}' | sudo tee /usr/share/vulkan/icd.d/nvidia_icd.json
```

Verify Vulkan:

```bash
# With display
vulkaninfo --summary

# Headless
DISPLAY= vulkaninfo --summary 2>&1 | head -30
```

## Step 4: Install Isaac Lab 2.1.1

Install additional system dependencies:

```bash
sudo apt install cmake build-essential
```

Clone and install Isaac Lab:

```bash
git clone https://github.com/isaac-sim/IsaacLab
cd IsaacLab
git fetch origin tag v2.1.1 && git checkout v2.1.1
./isaaclab.sh --install
```

Before editable installation, update the `flatdict` requirement:

- open `source/isaaclab/setup.py`
- replace `flatdict==4.0.1` with `flatdict==4.1.0`

Then install Isaac Lab into the current environment:

```bash
pip install --upgrade setuptools
pip install -e source/isaaclab
```

Verify Isaac Lab:

```bash
conda activate uni
cd /path/to/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py -- --task=Isaac-Ant-v0 --headless
```

## Step 5: Install FTP1 inference into the Isaac Sim environment

This historical same-environment procedure requires an FTP1 checkout compatible with Python 3.10. The current full policy repository requires Python `>=3.11`; do not lower that requirement to force this legacy workflow. Use the pinned Docker workflow above for the current combined simulator/policy stack.

Then from the `ftp1` repository root, run:

```bash
bash scripts/shell/install_ftp1_infer_into_isaacsim.sh
```

This is the recommended method for UniVTAC because it installs FTP1 inference without replacing Isaac Sim's PyTorch stack.

After installation, the same `uni` environment should support both:

- Isaac Lab / TacEx simulation
- `FTP1InferenceWrapper`-based inference

## Step 6: Install TacEx from the local modified source

UniVTAC requires the modified TacEx source shipped inside this repository:

```bash
cd /path/to/ftp1/UniVTAC/third_party/TacEx
```

Do **not** install TacEx from the public upstream repository for this project setup.

### Step 6.1: Install TacEx core

Make sure you are still inside the Isaac Sim / Isaac Lab environment:

```bash
conda activate uni
```

Install TacEx core packages:

```bash
./tacex.sh -i
```

Verify TacEx with a simple demo:

```bash
python ./scripts/demos/tactile_sim_approaches/check_taxim_sim.py --debug_vis
```

You can also test an RL example:

```bash
python ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Ball-Rolling-Tactile-RGB-v0 --num_envs 512 --enable_cameras
```

If the simulation runs correctly, you can inspect tactile outputs in the Isaac Lab UI under:

`Scene Debug Visualization > Observations > sensor_output`

### Step 6.2: Install TacEx UIPC support

The `tacex_uipc` package is responsible for UIPC-based tactile simulation in TacEx.

#### 6.2.1 Install Vcpkg

If Vcpkg is not installed yet:

```bash
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
sudo apt-get update
sudo apt-get install -y curl zip unzip tar
./bootstrap-vcpkg.sh -disableMetrics
```

Install `spdlog` through Vcpkg:

```bash
./vcpkg install spdlog:x64-linux
```

#### 6.2.2 Export `CMAKE_TOOLCHAIN_FILE`

Set the Vcpkg toolchain path so that CMake can find it:

```bash
export CMAKE_TOOLCHAIN_FILE="${YOUR_PATH}/Toolchain/vcpkg/scripts/buildsystems/vcpkg.cmake"
```

You may want to put the export into `~/.bashrc`.

#### 6.2.3 Install the build environment for libuipc

TacEx UIPC currently expects:

- CMake `3.26`
- GCC `11.4`
- CUDA `12.4`

Install the conda environment additions from TacEx:

```bash
cd /path/to/ftp1/UniVTAC/third_party/TacEx
conda activate uni
conda env update -n uni --file ./source/tacex_uipc/libuipc/conda/env.yaml
```

If CUDA 12.4 does not work on your machine, update your NVIDIA driver or adjust the TacEx environment file to use an older CUDA version such as 12.2.

#### 6.2.4 Install `tacex_uipc`

Make sure `CMAKE_TOOLCHAIN_FILE` is set, then install:

```bash
conda activate uni
rm -rf source/tacex_uipc/build
export CMAKE_TOOLCHAIN_FILE="${YOUR_PATH}/Toolchain/vcpkg/scripts/buildsystems/vcpkg.cmake"
pip install -e source/tacex_uipc -v
```

You may also install all TacEx packages through:

```bash
./tacex.sh -i all
```

#### 6.2.5 Verify `tacex_uipc`

Run:

```bash
python ./scripts/benchmarking/tactile_sim_performance/run_ball_rolling_experiment.py --num_envs 1 --debug_vis --env uipc
```

## Step 7: Install cuRobo

UniVTAC uses cuRobo for GPU-accelerated collision-aware motion planning.

Follow the official cuRobo guide:

- https://curobo.org/get_started/1_install_instructions.html

## Step 8: Fix `transformers_replace` inside the Conda environment

If FTP1 evaluation reports an error similar to:

> `transformers_replace is not installed correctly`

copy the custom transformer replacement files into the active Conda environment instead of a `.venv`.

For Python `3.10`, run from the `ftp1` repository root:

```bash
conda activate uni
cd /path/to/ftp1
cp -r ./src/openpi/models_pytorch/transformers_replace/* "$CONDA_PREFIX/lib/python3.10/site-packages/transformers/"
```

If your environment uses a different Python version, replace `python3.10` with the actual version path.

## Step 9: Install ffmpeg

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
ffmpeg -version
```

## Step 10: Fix `torch-scatter` if needed

If `torch_scatter` has binary compatibility issues in the current environment:

```bash
conda activate uni
pip uninstall torch_scatter
pip install torch_scatter --no-binary :all: --no-build-isolation
```

## Recommended verification checklist

Before using UniVTAC for collection or evaluation, verify the following in order:

1. `isaacsim --allow-root --headless` launches successfully.
2. Isaac Lab runs a basic training script.
3. `bash UniVTAC/scripts/shell/install_ftp1_infer_into_isaacsim.sh` completes without replacing `torch`.
4. FTP1 inference imports correctly in the `uni` environment.
5. TacEx core demo runs.
6. `tacex_uipc` example runs if UIPC simulation is needed.
7. `ffmpeg -version` works.

## Notes

- The current installation flow is designed around a single shared Conda environment called `uni`.
- For this repository version, the local TacEx source under `UniVTAC/third_party/TacEx` is the required source of truth.
- The FTP1 integration path used here is the script:
  `UniVTAC/scripts/shell/install_ftp1_infer_into_isaacsim.sh`
- This guide intentionally keeps the troubleshooting commands from the original notes because they are still useful on real deployment machines.
