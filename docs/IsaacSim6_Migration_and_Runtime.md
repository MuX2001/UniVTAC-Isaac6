# UniVTAC Isaac Sim 6 Migration and Runtime Guide

**Status:** validated through 2026-09-07. This is the operational reference for the migrated simulator and the companion document for a technical review. It records what was changed, what was deliberately not changed, how the runtime is used safely on a 16 GiB GPU, and what remains open.

## Executive result

The Docker-based UniVTAC migration now runs on the pinned Isaac Sim 6 stack without changing Isaac Sim, Isaac Lab, cuRobo, TacEx/UIPC mechanics, materials, friction, actuator limits, or task success criteria. Guarded `lift_can`, `pull_out_key`, and two of three fixed-seed `insert_tube` runs reach their unchanged success conditions. Head and wrist RGB observations now have the official policy shape, the wrist camera follows its authored mount, and real simulator MP4 recording works without a standalone FFmpeg executable. The corrected observation and pose paths also run normally for `lift_bottle`; that task currently exposes a genuine command/controller-boundary reproducibility difference rather than a simulator crash or a criterion to relax.

The existing September 3 portable image is intentionally a **simulator runtime snapshot**, not a cache or dataset archive. It contains the source current at export time, assets, task configuration, and TacEx package sources. It excludes local diagnostics, CUDA cache, reference HDF files, FTP1 checkpoints, and the parent `openpi` source tree. The September 7 camera and recorder fixes are delivered by the repositories and override the snapshot through the documented source mounts; this avoids another redundant 25.6 GB archive.

## Pinned runtime and scope

| Item | Verified value | Why it is pinned |
| --- | --- | --- |
| Development base image | `user10/univtac-isaac60-lab3-tacex:ftp1-pytorch-runtime-fem-tactile-curobo-sm120-uipc-sm120-resetcache` | The tested Isaac Sim 6 runtime with FEM tactile, cuRobo, and UIPC extensions built for this GPU generation. |
| Base image digest | `sha256:7d54bad40ec8421a6fbef6a6e42e3e705b1f47baf7c9b4dafcda8f2501e858c6` | Lets another machine verify it is starting from the same local base. |
| Core stack | Isaac Sim 6.0, Isaac Lab 6.1.17, Python 3.12.13, PyTorch 2.10.0+cu128 | Versions observed in the supplied image. Do not upgrade or downgrade them for this migration. |
| Portable image tag | `univtac-isaac60-lab3-tacex:ftp1-migrated-20260903-r2` | A child image that contains the curated migrated UniVTAC source. |
| Tested operating envelope | one headless environment on a 16 GiB GPU | The guard reserves desktop VRAM instead of assuming all device memory is usable. |

The legacy Isaac Sim 4.5 / Isaac Lab 2.1.1 instructions remain below the historical section of [Installation_FTP1.md](../Installation_FTP1.md). They are not a procedure for rebuilding this migrated runtime.

## What was found and fixed

The migration work addressed compatibility boundaries and observation correctness. It did not tune the task to make a result pass.

| Area | Finding | Resolution | Evidence / implementation |
| --- | --- | --- | --- |
| End-effector pose boundary | Isaac Lab 3 returns the end-effector quaternion in `xyzw`, while UniVTAC `Pose` and cuRobo expect `wxyz`. Passing the former through changed the requested wrist orientation. | Convert only this interface boundary from `xyzw` to `wxyz`. | [`envs/robot/robot.py`](../envs/robot/robot.py) |
| CuRobo target pose | An extra quaternion reorder existed at the planner target boundary. | Pass the already-correct `Pose` list directly to cuRobo. | [`envs/robot/curobo_planner.py`](../envs/robot/curobo_planner.py) |
| Collision world | A temporary Isaac Sim 6 fallback had effectively removed meaningful scene collision geometry. | Reconstruct the normal ground/actor collision world and record diagnostics; no collision margin or task geometry was changed. | [`envs/robot/curobo_planner.py`](../envs/robot/curobo_planner.py) |
| Camera initialization | Isaac Lab's Fabric pose reader rejected a valid authored `float3` wrist-camera scale. | Use its USD pose-reader fallback for that camera only. PhysX/Fabric simulation state is untouched. | [`envs/sensors/camera.py`](../envs/sensors/camera.py) |
| Wrist RGB extrinsic | A stale nested-camera world pose was incorrectly combined with the live `WristCamera` body pose, placing the view about 0.671 m from the hand. | Compose the authored local USD camera transform with the live mount pose every render. Runtime distance to `panda_hand` is now about 0.07596 m. | [`envs/sensors/camera.py`](../envs/sensors/camera.py) |
| Camera observation shape | Isaac Sim rendered the normal RGB cameras at a different size from the official policy tensors. | Keep a `640x360` raw render and expose `480x270` RGB/RGBA with antialiased resize; resize depth with nearest-neighbour sampling. | [`envs/sensors/camera.py`](../envs/sensors/camera.py) |
| Moving tactile marker frame | UIPC gel vertices move in world coordinates while the authored marker camera frame is fixed. The resulting marker grid compressed incorrectly. | Reconstruct the rigid gelpad motion from constrained vertices and express current points in the immutable initial camera frame. | Local TacEx package source and [`scripts/analyze_tactile_validation.py`](../scripts/analyze_tactile_validation.py) |
| Tactile RGB/depth frame | The nested tactile camera did not follow the moving gelpad; the visual gel meshes also contaminated readback. | Update camera pose from the same rigid transform, hide all gel visual meshes only during each tactile readback, then restore visibility. | Local TacEx package source; saved frame-consistency telemetry |
| UIPC attachment conversion | PhysX uses `xyzw`; USD `Gf.Quat` is `wxyz`. Mixing them rotated attachment offsets/targets during motion. | Convert authored USD orientation to `xyzw` before offsets and keep the runtime attachment path in `xyzw`. | [`third_party/TacEx/source/tacex_uipc`](../third_party/TacEx/source/tacex_uipc) |
| Action execution | Writing joint state for every waypoint teleported the robot and created large pad-target jumps. | Use the normal controller target path for motion; state writes remain reset-only. | [`envs/_base_task.py`](../envs/_base_task.py) and validation telemetry |
| Evidence quality | `Plan True` alone could hide a tactile or attachment failure. | Add opt-in capture of plans, contact, attachments, camera pose, and rigid-fit diagnostics. | [`scripts/collect_data.py`](../scripts/collect_data.py) |
| Video recording | The image has no standalone `ffmpeg`, so the old writer silently produced no usable demonstration. | Prefer FFmpeg/libx264 when present and otherwise use the verified OpenCV `mp4v` backend with correct RGB/BGR conversion. | [`envs/utils/data.py`](../envs/utils/data.py) |
| Video montage | Camera and tactile frames have different aspect ratios and were unsuitable for a meeting view. | Build an aspect-preserving `960x540` 2x2 montage of head, wrist, left tactile, and right tactile streams. | [`envs/_base_task.py`](../envs/_base_task.py) |

The following were deliberately **not** changed: Isaac Sim/Lab/cuRobo version, UIPC constraint strength, material or friction values, collision activation margin, actuator limits, random-number convention, or task success predicates.

## Verified behavior

| Validation | Result | What it establishes |
| --- | --- | --- |
| `lift_can`, seed 41 | Success after 799 task steps / 281 saved frames; all five planner moves succeeded; bilateral tactile contact in 222 frames. | Full task, planner, controller, observation, attachment, and unchanged check work together. |
| `lift_can` tactile and frame diagnostics | No-contact/release are exactly flat at 34 mm; bilateral contact reaches 27.63 mm (left) and 27.35 mm (right). 83 anchors per pad, maximum snapshot attachment error 0.00684 mm, rigid-fit RMS at most 0.771 micrometres, camera position error at most 55.9 nm. | The visible tactile result has the intended moving-frame semantics rather than merely a nonempty image. |
| `pull_out_key`, seed 41 | Success after 794 steps / 208 saved frames; all four planner queries succeeded; bilateral contact in every saved frame; zero attachment callback-generation lag. | An independent contact-rich task also succeeds with unchanged logic. |
| `insert_tube`, seeds 41/42/43 | Seeds 42 and 43 pass after 927/875 steps. Seed 41 misses only the unchanged 5 mm lateral bound by 0.288 mm. | The task is runnable and succeeds across more than one seed; the marginal seed-41 edge remains reported rather than hidden by threshold or physics changes. |
| Head/wrist camera acceptance | Raw frames are `640x360`; all official and local policy inputs decode as `480x270x3`. The corrected wrist camera remains about 0.07596 m from the hand with mount error below `4.5e-8 m`. | Camera shape and mount-following contracts pass; rendering texture/noise and exact trajectory are not pixel-identical to the dataset. |
| MP4 demonstration | Two simulator recordings decode as `960x540`; the labelled meeting video contains 277 frames. | The recorder and four-stream visualization work. Playback FPS is presentation timing, not the simulation clock. |
| Resource safety | Each acceptance container exited `0` with `OOMKilled=false`; observed guarded VRAM was about 6.9–7.6 GiB, below the 10,000 MiB stop threshold. | The current one-environment workflow is safe for the tested 16 GiB card when the guard is used. |

Full comparison, including the remaining `lift_bottle` difference, is in [IsaacSim6_Validation_Comparison.md](./IsaacSim6_Validation_Comparison.md).

## Safe use on another machine

### Prerequisites

- Linux, Docker, an NVIDIA driver compatible with the image (the image declares minimum driver `570.169`), and NVIDIA Container Toolkit.
- The verified configuration uses a 16 GiB GPU for one environment. Treat free VRAM as a preflight check on every host; do not assume a 16 GiB card is sufficient while other GPU workloads are present.
- More than 30 GiB free disk for the image archive plus Docker's unpacked layers; keep additional space for output episodes.
- Do not run more than one Isaac Sim environment or a second GPU-heavy process while validating.

The `--memory-swap=15g` setting means the container receives no additional swap beyond its 15 GiB RAM allocation. Host swap activity by itself does not prove insufficient simulator RAM; check the container exit code and `OOMKilled` field.

### Load and verify the portable image

Copy the exported archive and its checksum file to the target machine, then run:

```bash
sha256sum -c univtac-isaacsim6-migrated-20260903-r2.tar.sha256
docker load -i univtac-isaacsim6-migrated-20260903-r2.tar
docker image inspect univtac-isaac60-lab3-tacex:ftp1-migrated-20260903-r2 \
  --format 'id={{.Id}} size={{.Size}} user={{.Config.User}}'
```

The image's default command opens a shell; it never starts Isaac Sim implicitly. Always give a named container to preserve logs and diagnostics.

The `r2` archive is an immutable September 3 snapshot. Its binary Isaac runtime is still the pinned runtime, but its embedded checkout predates the September 7 wrist-camera, recorder, montage, and insertion-envelope work. For the current behavior, clone one of the updated repositories and use the three source mounts made by `scripts/shell/run_guarded_validation.sh`. Do not report an unmounted `r2` run as validation of the later source fixes.

### Historical embedded-image acceptance probe

Create an output directory on the host and run this exact guarded command. It writes generated data to the host mount at `./univtac-output` and leaves the stopped container available for inspection.

```bash
IMAGE='univtac-isaac60-lab3-tacex:ftp1-migrated-20260903-r2'
RUN_NAME='univtac-lift-can-portable-seed41'
mkdir -p ./univtac-output

docker run --name "$RUN_NAME" \
  --gpus all --memory=15g --memory-swap=15g --cpus=6 --pids-limit=512 \
  --ulimit core=0 --shm-size=64m --user 1234:1000 \
  -v "$(pwd)/univtac-output:/output" \
  -w /workspace/UniVTAC "$IMAGE" -lc '
set -u
timeout --signal=TERM --kill-after=15s 900s \
  /isaac-sim/python.sh scripts/collect_data.py lift_can task_config/portable_contact.yml \
  --episode_num 1 --start_seed 41 --max_seed 41 --headless --livestream 0 \
  --validation-dir /output/lift-can-seed41 &
collector_pid=$!
while kill -0 "$collector_pid" 2>/dev/null; do
  vram_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")
  printf "[VRAM_GUARD] %s used_mb=%s threshold_mb=10000\\n" "$(date -Iseconds)" "$vram_used"
  if [ "$vram_used" -ge 10000 ]; then
    printf "[VRAM_GUARD] stopping collector before desktop VRAM reserve is exhausted\\n"
    kill -TERM "$collector_pid"
    wait "$collector_pid"
    exit 75
  fi
  sleep 1
done
wait "$collector_pid"
'
```

Afterward, inspect rather than delete the container:

```bash
docker inspect "$RUN_NAME" --format 'exit={{.State.ExitCode}} oom={{.State.OOMKilled}}'
docker logs "$RUN_NAME" | tail -100
```

Exit `0` with `oom=false` is necessary but not alone sufficient. Review the host output's `contact_trace.json`, `plan_trace.json`, bilateral-contact states, and captured images. If the guard exits `75`, stop additional simulations and free GPU workload before retrying; do not raise the threshold casually.

To stop a still-running probe safely, use `docker stop -t 30 "$RUN_NAME"`. Do not use `--rm`; retained containers are useful failure evidence.

### Current source-mounted acceptance (recommended)

For simulator work, clone the standalone repository. For FTP1 policy evaluation, clone the full policy repository and enter its `UniVTAC` directory. You do not need both repositories on the target machine.

```bash
# Simulator-only checkout:
git clone git@github.com:MuX2001/UniVTAC-Isaac6.git
cd UniVTAC-Isaac6

# Or, for policy evaluation:
# git clone git@github.com:MuX2001/ftp1-policy.git
# cd ftp1-policy/UniVTAC
```

When using the existing offline archive, select its tag explicitly. The launcher mounts the current checkout and both current TacEx source packages over the image:

```bash
UNIVTAC_IMAGE='univtac-isaac60-lab3-tacex:ftp1-migrated-20260903-r2' \
  bash scripts/shell/run_guarded_validation.sh lift_can 41 episode target-r1
```

If the pinned development image is available from a registry, omit `UNIVTAC_IMAGE`; it is the launcher's default. Results are written under `runtime-output/<container-name>/`. The stopped named container is retained for `docker inspect` and `docker logs`.

### Development workflow with a source checkout

The existing portable image is suitable as the repeatable binary runtime. For current source or active development, use the pinned development base or the loaded `r2` runtime and mount all three paths; the two TacEx package mounts are required so the current local compatibility code is imported.

```bash
docker run --name univtac-dev-shell \
  --gpus all --memory=15g --memory-swap=15g --cpus=6 --pids-limit=512 \
  --ulimit core=0 --shm-size=64m --user 1234:1000 \
  -v "<repo>/UniVTAC:/workspace/UniVTAC" \
  -v "<repo>/UniVTAC/third_party/TacEx/source/tacex/tacex:/opt/tacex/source/tacex/tacex" \
  -v "<repo>/UniVTAC/third_party/TacEx/source/tacex_uipc/tacex_uipc:/opt/tacex/source/tacex_uipc/tacex_uipc" \
  -w /workspace/UniVTAC --entrypoint /bin/bash \
  user10/univtac-isaac60-lab3-tacex:ftp1-pytorch-runtime-fem-tactile-curobo-sm120-uipc-sm120-resetcache
```

From that shell, invoke `/isaac-sim/python.sh`, not bare `python`. Use one task and one seed while validating. The parent FTP1/openpi source and policy checkpoints are intentionally not embedded in the portable simulator image; mount them separately only for FTP1 evaluation.

## Portable-image contents and regeneration

The release is created by [`scripts/shell/export_migrated_docker_image.sh`](../scripts/shell/export_migrated_docker_image.sh). It stages a curated source tree, copies the two edited TacEx package roots into the image's installed source locations, changes the default command to a harmless shell, commits a new local image, then writes an archive, SHA-256 checksum, and manifest.

On a source machine with enough disk, invoke it explicitly with an output directory and a unique release ID (defaults are `/home/mu/Downloads` and `20260907-r3`):

```bash
bash scripts/shell/export_migrated_docker_image.sh /path/with-enough-free-space 20260907-r3
```

No new `r3` archive was produced for this delivery: the pinned binary image ID did not change, and the current source is smaller, reviewable, and supplied by Git. Export `r3` only when an immutable no-bind-mount package is required or the target has no repository access. The exporter refuses to overwrite an existing image, container, archive, checksum, or manifest.

It excludes the following deliberately:

- local `data/` and `validation_artifacts/`, including generated episodes, GPU cache, and all prior diagnostics;
- downloaded public HDF files and extracted comparison images;
- Python bytecode/build caches;
- FTP1 checkpoints and the parent repository's `openpi` source.

The exclusions keep the simulator transfer practical and avoid presenting local evidence as part of runtime code. Retain the original output directories separately when an audit trail is required.

## Target-machine adjustment checklist

| Target-dependent item | Required check or adjustment |
| --- | --- |
| Driver/runtime | NVIDIA driver must satisfy the image's declared minimum (`570.169`), Docker must work, and NVIDIA Container Toolkit must expose the GPU. Do not change Isaac Sim or Isaac Lab versions. |
| GPU headroom | Start with one headless environment and the 10,000 MiB total-used stop threshold. Run `nvidia-smi` first and close other GPU-heavy work. A 16 GiB model name alone is not proof of available VRAM. |
| Host RAM and swap | Keep the default `15g` container RAM cap with `--memory-swap` equal to it. Host swap occupancy alone is not failure; rely on exit code, `OOMKilled`, and system responsiveness. Lower the cap only after a bounded validation proves the host needs it. |
| Disk | Loading the archive needs space for both the 25.6 GB tar and unpacked Docker layers; reserve more than 30 GB plus episode outputs. The registry-plus-clone route avoids copying the tar but not Docker layer storage. |
| Checkout layout | Run the launcher from either standalone repo root or `ftp1-policy/UniVTAC`. It derives absolute mount paths itself. Always mount the main tree and both TacEx package roots. |
| Permissions | The validated container identity is UID 1234/GID 1000. The launcher makes each new output folder writable. If site policy forbids that mode, pre-create `UNIVTAC_OUTPUT_ROOT` with an appropriate ACL and set `UNIVTAC_CONTAINER_USER` deliberately. |
| Policy assets | The simulator image deliberately excludes FTP1/openpi code and checkpoints. For policy evaluation, use the full policy repository and separately restore/mount the intended checkpoint and normalization assets. |
| Display/network | Use `--headless --livestream 0` for bounded validation. Enable livestream or a GUI only after the headless acceptance passes and with a new resource envelope. |
| Acceptance | First run `collect 3992 tactile-sanity`, then `lift_can 41 episode`; inspect `camera_validation.json`, `tactile_sanity.json`, plans, exit code, OOM state, and saved frames before policy evaluation. |

## Known limits and honest status

- `lift_bottle` plans and tactile sensing run normally, but its current free-object trajectory takes a different post-rotation branch and ends `Plan True, Check False` under the unchanged predicate. A guarded seed-44 replay of the official joint stream through rotate four tracks within 0.00536 rad and keeps the bottle within 3.945 mm of the reference, so the pinned controller/UIPC path is capable when it receives the official commands. The remaining bounded discrepancy is the normal CuRobo-generated command stream (first rotate: 0.00416 cadence-normalized joint-L2 RMS versus the official two-tick stream; the error compounds under contact). It is not a license to change physics, RNG, collision geometry, or success rules.
- The reported `grasp_classify` attempt could not construct its scene because the validation workspace lacked `assets/objects/RoughPrism.usd` and `assets/objects/PlainPrism.usd`. The standalone simulator repository now contains those assets, but this task has not yet received a guarded acceptance run there. A clean Isaac shutdown after a Python `FileNotFoundError` is not task success, and substitute assets must not be invented.
- The validated GPU guard is a conservative policy for the tested workstation, not a promise that every host has enough VRAM. Check `nvidia-smi` before each run and retain the 10,000 MiB stop threshold unless the hardware envelope is revalidated.
- Dataset comparison is valuable for behavior review, but published HDF file names do not by themselves guarantee the same reset state/RNG stream in this runtime. Use the comparison document before claiming bit-identical reproduction.
