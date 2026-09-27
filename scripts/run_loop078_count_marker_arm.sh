#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wio/Inference_Foundry
ARM=${1:?A0 B or A1}
case "${ARM}" in A0) RUN=413;; B) RUN=410;; A1) RUN=411;; *) exit 2;; esac
OUT=${ROOT}/evidence/20260927_loop078_bound/run${RUN}
DATASET=/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl
CONTAINER=vllm-ascend26-dsv4f-w4a8
RUN_TS=LOOP078-RUN${RUN}-${ARM}
SOURCES=(
"${ROOT}/runtime/extreme_decode.py"
"${ROOT}/runtime/fixed_decode.py"
"${ROOT}/runtime/fixed_serving.py"
"${ROOT}/bootstrap/vllm_dspark_handoff.py"
/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/spec_decode/utils.py
/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/attention/dsa_v1.py
)
mkdir -p "${OUT}/runtime"
if [[ "${ARM}" == B ]]; then mkdir -p "${OUT}/capture"; fi
if curl -fsS --max-time 2 http://127.0.0.1:8080/health >/dev/null 2>&1; then
  echo "service already healthy" >&2; exit 1
fi
if pgrep -af "[p]ython3 scripts/bench.py" | grep -v "<defunct>" >"${OUT}/orphan_bench.txt"; then
  echo "orphan bench process" >&2; exit 1
fi
RUN_STARTED=0
verify_stopped() {
  docker exec "${CONTAINER}" python3 -c '
import pathlib, sys
live = []
for p in pathlib.Path("/proc").iterdir():
    if not p.name.isdigit():
        continue
    try:
        comm = (p / "comm").read_text().strip()
        status = (p / "status").read_text()
    except (FileNotFoundError, ProcessLookupError):
        continue
    state = next((line.split()[1] for line in status.splitlines() if line.startswith("State:")), None)
    if state is None:
        raise RuntimeError("cannot establish process state for " + p.name)
    if state == "Z":
        continue
    if comm.startswith("VLLM::") or comm == "vllm":
        live.append((p.name, comm))
print(live)
sys.exit(bool(live))
' >"${OUT}/stop_process_probe.txt" 2>&1 || return $?
  npu-smi info >"${OUT}/stop_npu_probe.txt" 2>&1 || return $?
  python3 - "${OUT}/stop_npu_probe.txt" <<'CHECK_STOP'
import pathlib, re, sys
text = pathlib.Path(sys.argv[1]).read_text()
used = [int(x) for x in re.findall(r"(\d+)\s*/\s*65536", text)]
print({"used_mb": used, "VLLM_rows_present": "VLLM" in text})
assert len(used) == 8 and max(used) < 6144 and "VLLM" not in text, "stop not established"
CHECK_STOP
}
cleanup() {
  local run_rc=$?
  trap - EXIT
  set +e
  local stop_rc=0 verify_rc=0 restore_rc=not_needed sha_rc=0 compare_rc=not_checked
  local final_rc=${run_rc}
  if [[ "${RUN_STARTED}" == 1 ]]; then
    bash /data/wio/vllm_ascend_26/scripts/99_stop_service.sh >"${OUT}/stop.log" 2>&1
    stop_rc=$?
    if [[ "${stop_rc}" == 0 ]]; then
      verify_stopped >>"${OUT}/stop.log" 2>&1
      verify_rc=$?
    else
      verify_rc=not_attempted
    fi
  fi
  if [[ "${ARM}" == B && -f "${OUT}/patch_state/manifest.json" ]]; then
    if [[ "${stop_rc}" == 0 && "${verify_rc}" == 0 ]]; then
      python3 "${ROOT}/scripts/loop078_count_markers_patch.py" restore --state-dir "${OUT}/patch_state" --record "${OUT}/restore.json" >"${OUT}/restore.log" 2>&1
      restore_rc=$?
    else
      restore_rc=skipped_service_stop_unproven
      echo "Refusing source restore: service stop failed or is unproven; preserve patch and backups." >&2
    fi
  fi
  sha256sum "${SOURCES[@]}" >"${OUT}/source_after.sha256"
  sha_rc=$?
  if [[ "${sha_rc}" == 0 ]]; then
    cmp -s "${OUT}/source_before.sha256" "${OUT}/source_after.sha256"
    compare_rc=$?
  fi
  if [[ "${stop_rc}" != 0 || "${verify_rc}" != 0 || "${sha_rc}" != 0 || "${compare_rc}" != 0 || ( "${restore_rc}" != not_needed && "${restore_rc}" != 0 ) ]]; then
    if [[ "${final_rc}" == 0 ]]; then final_rc=1; fi
  fi
  printf 'run_exit=%s\nstop_exit=%s\nstop_verify_exit=%s\nrestore_exit=%s\nsha_exit=%s\nsha_compare_exit=%s\nfinal_exit=%s\n' \
    "${run_rc}" "${stop_rc}" "${verify_rc}" "${restore_rc}" "${sha_rc}" "${compare_rc}" "${final_rc}" \
    | tee "${OUT}/cleanup_status.txt" >&2
  exit "${final_rc}"
}
trap cleanup EXIT
sha256sum "${SOURCES[@]}" >"${OUT}/source_before.sha256"
if [[ "${ARM}" == B ]]; then
  python3 "${ROOT}/scripts/loop078_count_markers_patch.py" install \
    --expected-check "${ROOT}/evidence/20260927_loop078_bound/run401/source_check_sol.json" \
    --state-dir "${OUT}/patch_state" --record "${OUT}/install.json"
fi
RUN_STARTED=1
ENV_ARGS=(
  -e MAX_MODEL_LEN=1048576
  -e RUN_TS="${RUN_TS}"
  -e EXTREME_RUNTIME_RUN_DIR="${OUT}/runtime"
  -e EXTREME_RUNTIME_SERVE=1
  -e EXTREME_RUNTIME_RESERVE_TOKENS=1088
  -e EXTREME_NATIVE_TARGET_METADATA=1
  -e EXTREME_RUNTIME_TARGET_GRAPH=1
  -e EXTREME_DSPARK_SLOT_REFRESH=1
  -e EXTREME_TARGET_METADATA_STATIC_KV_MAX=1
)
if [[ "${ARM}" == B ]]; then
  ENV_ARGS+=(-e EXTREME_COUNT_MARKER_DIR="${OUT}/capture" -e EXTREME_COUNT_MARKER_RUN_ID="${RUN_TS}")
fi
docker exec "${ENV_ARGS[@]}" "${CONTAINER}" bash -lc "cd ${ROOT} && bash scripts/serve.sh > ${OUT}/launcher.log 2>&1"
for i in $(seq 1 180); do
  if curl -fsS --max-time 3 http://127.0.0.1:8080/health >/dev/null 2>&1; then break; fi
  sleep 10
done
curl -fsS --max-time 5 http://127.0.0.1:8080/health >/dev/null
docker exec "${CONTAINER}" bash -lc "cd ${ROOT} && python3 scripts/bench.py --dataset ${DATASET} --out ${OUT}/warmup48.json --limit 48 --concurrency 12 --max-tokens 1024 >${OUT}/warmup48.log 2>&1"
docker exec "${CONTAINER}" bash -lc "cd ${ROOT} && python3 scripts/bench.py --dataset ${DATASET} --out ${OUT}/bench.json --limit 12 --concurrency 12 --max-tokens 1024 >${OUT}/bench.log 2>&1"
LOG="${ROOT}/logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_${RUN_TS}.log"
POSTS=$(grep -c "POST /v1/chat/completions" "${LOG}")
printf "%s\n" "${POSTS}" >"${OUT}/server_post_count.txt"
test "${POSTS}" -eq 60
