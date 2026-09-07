#!/usr/bin/env bash
# Package the validated UniVTAC Isaac Sim 6 runtime without validation output,
# caches, public HDF files, or parent FTP1/openpi source. Run from a host that
# has Docker access; this script never starts Isaac Sim.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
univtac_dir="$(cd -- "$script_dir/../.." && pwd)"
archive_dir="${1:-/home/mu/Downloads}"
release_id="${2:-20260907-r3}"
case "$release_id" in
  ''|*[!A-Za-z0-9._-]*) printf 'release-id contains unsupported characters\n' >&2; exit 2 ;;
esac

base_image='user10/univtac-isaac60-lab3-tacex:ftp1-pytorch-runtime-fem-tactile-curobo-sm120-uipc-sm120-resetcache'
portable_image="univtac-isaac60-lab3-tacex:ftp1-migrated-${release_id}"
archive_name="univtac-isaacsim6-migrated-${release_id}.tar"
container_name="univtac-package-isaacsim6-migrated-${release_id}"
archive_path="$archive_dir/$archive_name"
manifest_path="$archive_dir/univtac-isaacsim6-migrated-${release_id}.manifest.txt"

if [[ $# -gt 2 ]]; then
  printf 'usage: %s [archive-directory] [release-id]\n' "$0" >&2
  exit 2
fi

for required in docker rsync sha256sum; do
  command -v "$required" >/dev/null || {
    printf 'required command is unavailable: %s\n' "$required" >&2
    exit 127
  }
done

docker image inspect "$base_image" >/dev/null || {
  printf 'base image is not available locally: %s\n' "$base_image" >&2
  exit 1
}

if docker image inspect "$portable_image" >/dev/null 2>&1; then
  printf 'portable image already exists; refusing to overwrite: %s\n' "$portable_image" >&2
  exit 1
fi
if docker container inspect "$container_name" >/dev/null 2>&1; then
  printf 'packaging container already exists; refusing to overwrite: %s\n' "$container_name" >&2
  exit 1
fi
if [[ -e "$archive_path" || -e "$archive_path.sha256" || -e "$manifest_path" ]]; then
  printf 'export artifact already exists in %s; refusing to overwrite\n' "$archive_dir" >&2
  exit 1
fi

mkdir -p "$archive_dir"
stage_dir="$(mktemp -d /tmp/univtac-portable-stage.XXXXXX)"
cleanup_stage() {
  rm -rf -- "$stage_dir"
}
trap cleanup_stage EXIT

printf 'Staging curated UniVTAC source from %s\n' "$univtac_dir"
rsync -a \
  --exclude='.git/' \
  --exclude='data/' \
  --exclude='validation_artifacts/' \
  --exclude='__pycache__/' \
  --exclude='*.py[cod]' \
  --exclude='.pytest_cache/' \
  --exclude='*.egg-info/' \
  --exclude='third_party/TacEx/source/tacex_uipc/build*/' \
  --exclude='third_party/TacEx/source/tacex_uipc/libuipc/build*/' \
  "$univtac_dir/" "$stage_dir/UniVTAC/"

# The container is a dedicated packaging object. It remains stopped after the
# export so its image ancestry and assembly can be inspected later.
docker create --name "$container_name" --user 0 --entrypoint /bin/sleep "$base_image" infinity >/dev/null
docker start "$container_name" >/dev/null

docker exec "$container_name" /bin/bash -lc '
set -Eeuo pipefail
rm -rf /workspace/UniVTAC
mkdir -p /workspace/UniVTAC /output
'
docker cp "$stage_dir/UniVTAC/." "$container_name:/workspace/UniVTAC"
docker exec "$container_name" /bin/bash -lc '
set -Eeuo pipefail
rm -rf /opt/tacex/source/tacex/tacex /opt/tacex/source/tacex_uipc/tacex_uipc
cp -a /workspace/UniVTAC/third_party/TacEx/source/tacex/tacex /opt/tacex/source/tacex/tacex
cp -a /workspace/UniVTAC/third_party/TacEx/source/tacex_uipc/tacex_uipc /opt/tacex/source/tacex_uipc/tacex_uipc
chmod -R a+rX /workspace/UniVTAC
chown 1234:1000 /output
test -f /workspace/UniVTAC/envs/robot/robot.py
test -f /opt/tacex/source/tacex/tacex/gelsight_sensor.py
test -f /opt/tacex/source/tacex_uipc/tacex_uipc/sim/uipc_attachments.py
'
docker commit \
  --change 'WORKDIR /workspace/UniVTAC' \
  --change 'USER 1234:1000' \
  --change 'ENTRYPOINT ["/bin/bash"]' \
  --change 'CMD ["-lc", "cd /workspace/UniVTAC && exec /bin/bash"]' \
  --change 'LABEL org.opencontainers.image.title="UniVTAC Isaac Sim 6 migrated runtime"' \
  --change 'LABEL org.opencontainers.image.description="Pinned Isaac Sim 6 runtime with curated UniVTAC and TacEx migration sources"' \
  "$container_name" "$portable_image" >/dev/null

# Commit while the harmless idle process is alive. This avoids coupling the
# image snapshot to the Docker daemon's process-stop timeout. The stopped
# container is deliberately retained after a successful export.
docker stop -t 30 "$container_name" >/dev/null || true
if [[ "$(docker inspect "$container_name" --format '{{.State.Running}}')" != 'false' ]]; then
  printf 'packaging container is still running after stop request: %s\n' "$container_name" >&2
  exit 1
fi

docker image inspect "$portable_image" --format 'id={{.Id}} size={{.Size}} created={{.Created}}'
docker save --output "$archive_path" "$portable_image"
sha256sum "$archive_path" > "$archive_path.sha256"
{
  printf 'portable_image=%s\n' "$portable_image"
  printf 'base_image=%s\n' "$base_image"
  printf 'source_directory=%s\n' "$univtac_dir"
  printf 'archive=%s\n' "$archive_path"
  printf 'image_id=%s\n' "$(docker image inspect "$portable_image" --format '{{.Id}}')"
  printf 'created=%s\n' "$(docker image inspect "$portable_image" --format '{{.Created}}')"
  printf 'size_bytes=%s\n' "$(docker image inspect "$portable_image" --format '{{.Size}}')"
  sha256sum "$archive_path"
  printf 'excluded=local data and validation artifacts, python/build caches, TacEx build trees, parent FTP1/openpi source, checkpoints\n'
  printf 'packaging_container=%s (stopped and retained)\n' "$container_name"
} > "$manifest_path"

printf 'Created portable image: %s\nArchive: %s\nChecksum: %s\nManifest: %s\n' \
  "$portable_image" "$archive_path" "$archive_path.sha256" "$manifest_path"
