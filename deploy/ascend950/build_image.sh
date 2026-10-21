#!/usr/bin/env bash
# CPU-only build, resumable with the same command. No NPU devices are passed.
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
test -f "$ROOT/deploy.env" || { echo 'Run step 1 of the operating manual first (deploy.env missing).'; exit 1; }
source "$ROOT/deploy.env"
cd "$ROOT"
BUILD=deepinfer-950-build
if [ "${1:-}" != --force ] && python3 - "$ROOT" "$DEEPINFER_IMAGE" <<'PY'
import json, pathlib, subprocess, sys
r = pathlib.Path(sys.argv[1])
try:
    assert 'BUILD_COMPLETE_DEVICE_TEST_PENDING' in (r/'build-state.txt').read_text()
    # venv interpreters point into the container filesystem, not the host.
    assert all(p.is_file() or p.is_symlink() for p in [r/'engine-venv/bin/python', r/'recipe-venv/bin/python'])
    lock = json.loads((r/'image-lock.json').read_text())
    image = json.loads(subprocess.check_output(['docker','image','inspect',sys.argv[2]],stderr=subprocess.DEVNULL))[0]
    assert image['Id'] == lock['image_id']
    subprocess.run(['python3',str(r/'apply_patches.py'),'--check-only'],check=True,stdout=subprocess.DEVNULL)
except (OSError, ValueError, AssertionError, subprocess.SubprocessError):
    raise SystemExit(1)
PY
then
  echo 'BUILD_IMAGE_READY (reusing completed build and locked image)'
  exit 0
fi
for name in deepinfer-950-engine deepinfer-950-api; do
  if [ "$(docker inspect --format '{{.State.Running}}' "$name" 2>/dev/null || true)" = true ]; then
    echo "Stop this deployment's $name before changing its build. See the rebuild commands in the manual."
    exit 1
  fi
done
if docker container inspect "$BUILD" >/dev/null 2>&1; then
  python3 - "$BUILD" "$ROOT" "$BASE_IMAGE" <<'PY'
import json, subprocess, sys
d = json.loads(subprocess.check_output(['docker','inspect',sys.argv[1]]))[0]
assert any(m['Source'] == sys.argv[2] and m['Destination'] == '/work' for m in d['Mounts']), 'Build container workspace mismatch'
base = json.loads(subprocess.check_output(['docker','image','inspect',sys.argv[3]]))[0]['Id']
assert d['Image'] == base, 'Build container base image mismatch'
assert d['Config']['Cmd'] == ['/work/bootstrap.sh'], 'Unexpected build command'
PY
  if [ "$(docker inspect --format '{{.State.Running}}' "$BUILD")" != true ]; then
    docker start "$BUILD"
  fi
else
  docker run -d --name "$BUILD" --label inference-foundry.scope=950-deploy \
    --network host --cpus 16 --memory 64g -v "$ROOT:/work" \
    --entrypoint /bin/bash "$BASE_IMAGE" /work/bootstrap.sh
fi
docker logs --follow --tail 100 "$BUILD"
code=$(docker wait "$BUILD")
if [ "$code" != 0 ]; then
  echo "BUILD_FAILED exit=$code; fix the reported prerequisite and rerun: bash $ROOT/build_image.sh"
  exit 1
fi
grep -q BUILD_COMPLETE_DEVICE_TEST_PENDING "$ROOT/build-state.txt"
docker commit "$BUILD" "$DEEPINFER_IMAGE"
python3 - "$ROOT" "$DEEPINFER_IMAGE" <<'PY'
import json, pathlib, subprocess, sys
d = json.loads(subprocess.check_output(['docker','image','inspect',sys.argv[2]]))[0]
p = pathlib.Path(sys.argv[1])/'image-lock.json'
p.write_text(json.dumps({'image':sys.argv[2],'image_id':d['Id']},indent=2)+'\n')
print('BUILD_IMAGE_READY', d['Id'])
PY
