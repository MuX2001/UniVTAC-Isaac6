#!/usr/bin/env bash
# Run one bounded UniVTAC validation episode without assuming the GPU or host
# has spare memory. The stopped container and output directory are retained.
set -Eeuo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  printf 'usage: %s <task> <seed> <episode|tactile-sanity> [run-id]\n' "$0" >&2
  exit 2
fi

task="$1"
seed="$2"
mode="$3"
run_id="${4:-r1}"

case "$task" in
  collect|lift_can|lift_bottle|pull_out_key|insert_tube) ;;
  *) printf 'unsupported validation task: %s\n' "$task" >&2; exit 2 ;;
esac
case "$seed" in
  ''|*[!0-9]*) printf 'seed must be a non-negative integer\n' >&2; exit 2 ;;
esac
case "$mode" in
  episode|tactile-sanity) ;;
  *) printf 'mode must be episode or tactile-sanity\n' >&2; exit 2 ;;
esac
case "$run_id" in
  ''|*[!A-Za-z0-9._-]*) printf 'run-id contains unsupported characters\n' >&2; exit 2 ;;
esac

for required in docker nvidia-smi; do
  command -v "$required" >/dev/null || {
    printf 'required command is unavailable: %s\n' "$required" >&2
    exit 127
  }
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
univtac_dir="$(cd -- "$script_dir/../.." && pwd)"
image="${UNIVTAC_IMAGE:-user10/univtac-isaac60-lab3-tacex:ftp1-pytorch-runtime-fem-tactile-curobo-sm120-uipc-sm120-resetcache}"
output_root="${UNIVTAC_OUTPUT_ROOT:-$univtac_dir/runtime-output}"
memory_limit="${UNIVTAC_MEMORY_LIMIT:-15g}"
cpu_limit="${UNIVTAC_CPU_LIMIT:-6}"
vram_stop_mib="${UNIVTAC_VRAM_STOP_MIB:-10000}"
timeout_seconds="${UNIVTAC_TIMEOUT_SECONDS:-900}"
container_user="${UNIVTAC_CONTAINER_USER:-1234:1000}"

case "$vram_stop_mib" in
  ''|*[!0-9]*) printf 'UNIVTAC_VRAM_STOP_MIB must be an integer\n' >&2; exit 2 ;;
esac
case "$timeout_seconds" in
  ''|*[!0-9]*) printf 'UNIVTAC_TIMEOUT_SECONDS must be an integer\n' >&2; exit 2 ;;
esac

docker image inspect "$image" >/dev/null || {
  printf 'Docker image is not available locally: %s\n' "$image" >&2
  exit 1
}

task_slug="${task//_/-}"
mode_slug="${mode//_/-}"
container="univtac-${task_slug}-seed${seed}-${mode_slug}-${run_id}"
run_output="$output_root/$container"

if docker container inspect "$container" >/dev/null 2>&1; then
  printf 'refusing to replace existing container: %s\n' "$container" >&2
  exit 3
fi
if [[ -e "$run_output" ]]; then
  printf 'refusing to replace existing output: %s\n' "$run_output" >&2
  exit 3
fi

host_gpu_used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '1p' | tr -d ' ')"
case "$host_gpu_used_mib" in
  ''|*[!0-9]*) printf 'could not read current GPU memory use\n' >&2; exit 1 ;;
esac
if (( host_gpu_used_mib >= vram_stop_mib )); then
  printf 'refusing to start: GPU already uses %s MiB (stop threshold %s MiB)\n' \
    "$host_gpu_used_mib" "$vram_stop_mib" >&2
  exit 75
fi

mkdir -p "$run_output"
# The validated container UID/GID is 1234:1000. A run-specific output folder
# is made writable so differing host UID maps do not strand generated data.
chmod 0777 "$run_output"

printf 'image=%s\ncontainer=%s\noutput=%s\ninitial_gpu_used_mib=%s\n' \
  "$image" "$container" "$run_output" "$host_gpu_used_mib"

exec docker run --name "$container" \
  --no-healthcheck \
  --gpus all --memory="$memory_limit" --memory-swap="$memory_limit" \
  --cpus="$cpu_limit" --pids-limit=512 --ulimit core=0 --shm-size=64m \
  --user "$container_user" \
  -e UNIVTAC_TASK="$task" \
  -e UNIVTAC_SEED="$seed" \
  -e UNIVTAC_MODE="$mode" \
  -e UNIVTAC_VRAM_STOP_MIB="$vram_stop_mib" \
  -e UNIVTAC_TIMEOUT_SECONDS="$timeout_seconds" \
  -v "$univtac_dir:/workspace/UniVTAC" \
  -v "$univtac_dir/third_party/TacEx/source/tacex/tacex:/opt/tacex/source/tacex/tacex" \
  -v "$univtac_dir/third_party/TacEx/source/tacex_uipc/tacex_uipc:/opt/tacex/source/tacex_uipc/tacex_uipc" \
  -v "$run_output:/output" \
  -w /workspace/UniVTAC \
  "$image" -lc '
set -Eeuo pipefail
validation_args=()
if [[ "$UNIVTAC_MODE" == tactile-sanity ]]; then
  validation_args+=(--validation-tactile-sanity)
fi
timeout --signal=TERM --kill-after=15s "${UNIVTAC_TIMEOUT_SECONDS}s" \
  /isaac-sim/python.sh scripts/collect_data.py "$UNIVTAC_TASK" task_config/portable_contact.yml \
  --episode_num 1 --start_seed "$UNIVTAC_SEED" --max_seed "$UNIVTAC_SEED" \
  --headless --livestream 0 --validation-dir /output/evidence "${validation_args[@]}" &
collector_pid=$!
guard_max=0
while kill -0 "$collector_pid" 2>/dev/null; do
  vram_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "1p" | tr -d " ")
  if (( vram_used > guard_max )); then guard_max=$vram_used; fi
  printf "[VRAM_GUARD] %s used_mb=%s max_mb=%s threshold_mb=%s\n" \
    "$(date -Iseconds)" "$vram_used" "$guard_max" "$UNIVTAC_VRAM_STOP_MIB"
  if (( vram_used >= UNIVTAC_VRAM_STOP_MIB )); then
    printf "[VRAM_GUARD] stopping collector before desktop VRAM reserve is exhausted\n"
    kill -TERM "$collector_pid"
    wait "$collector_pid" || true
    exit 75
  fi
  sleep 1
done
wait "$collector_pid"
printf "VRAM_GUARD_MAX_MIB=%s\n" "$guard_max"
'
