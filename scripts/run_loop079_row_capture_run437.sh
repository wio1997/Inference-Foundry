#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wio/Inference_Foundry
OUT=${ROOT}/evidence/20260927_loop079_identity/run437
CONTAINER=vllm-ascend26-dsv4f-w4a8
RUN_TS=LOOP079-RUN437
DATASET=/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl
FRAME=/data/wio/vllm_ascend_26/framework
SOURCES=(
    "${FRAME}/vllm-ascend/vllm_ascend/compilation/acl_graph.py"
    "${FRAME}/vllm-ascend/vllm_ascend/quantization/methods/w4a8.py"
    "${FRAME}/vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py"
    "${FRAME}/vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py"
    "${ROOT}/runtime/extreme_decode.py"
    "${ROOT}/runtime/fixed_serving.py"
)
LOG=${ROOT}/logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_${RUN_TS}.log
mkdir -p "${OUT}/runtime" "${OUT}/capture"
if [[ -e "${OUT}/patch_state" || -e "${OUT}/bench.json" || -e "${OUT}/warmup48.json" ]] || \
   find "${OUT}/runtime" "${OUT}/capture" -mindepth 1 -print -quit | grep -q .; then
    echo "Run437 output directory is not a fresh acquisition" >&2; exit 1
fi

if curl -fsS --max-time 2 http://127.0.0.1:8080/health >/dev/null 2>&1; then
    echo "service already healthy" >&2; exit 1
fi
if npu-smi info | grep -q "VLLM::"; then
    echo "old VLLM worker still on NPU" >&2; exit 1
fi
docker exec "${CONTAINER}" python3 -c '
import pathlib
live=[]
for p in pathlib.Path("/proc").iterdir():
    if not p.name.isdigit(): continue
    try: comm=(p/"comm").read_text().strip(); status=(p/"status").read_text()
    except (FileNotFoundError,ProcessLookupError): continue
    state=next((x.split()[1] for x in status.splitlines() if x.startswith("State:")),None)
    if state!="Z" and (comm.startswith("VLLM::") or comm=="vllm"):
        live.append((p.name,comm))
print(live)
assert not live
' >"${OUT}/preflight_service_process.txt"
if pgrep -af "[p]ython3 scripts/bench.py" | grep -v "<defunct>" >"${OUT}/orphan_bench.txt"; then
    echo "orphan benchmark process" >&2; exit 1
fi
docker exec "${CONTAINER}" python3 -c '
import pathlib
active=[]
for p in pathlib.Path("/proc").iterdir():
    if not p.name.isdigit(): continue
    try: status=(p/"status").read_text()
    except (FileNotFoundError,ProcessLookupError): continue
    state=next((x.split()[1] for x in status.splitlines() if x.startswith("State:")),None)
    if state is None: raise RuntimeError("unknown process state")
    if state!="Z": active.append(p.name)
print("active container processes",len(active))
assert len(active)<=10
' >"${OUT}/preflight_process.txt"
python3 - <<'PY' >"${OUT}/preflight_memory.txt"
from pathlib import Path
s=Path("/proc/meminfo").read_text()
available=int(next(x.split()[1] for x in s.splitlines() if x.startswith("MemAvailable:")))
print("MemAvailable_kB",available)
assert available>256*1024*1024
PY
npu-smi info >"${OUT}/preflight_npu.txt"
python3 - "${OUT}/preflight_npu.txt" <<'PY'
import pathlib,re,sys
s=pathlib.Path(sys.argv[1]).read_text()
used=[int(x) for x in re.findall(r"(\d+)\s*/\s*65536",s)]
assert len(used)==8 and max(used)<6144 and "VLLM" not in s
print("preflight idle HBM",used)
PY

RUN_STARTED=0
verify_stopped() {
    if curl -fsS --max-time 2 http://127.0.0.1:8080/health >/dev/null 2>&1; then
        echo "service remains healthy after stop" >&2; return 1
    fi
    docker exec "${CONTAINER}" python3 -c '
import pathlib,sys
live=[]
for p in pathlib.Path("/proc").iterdir():
    if not p.name.isdigit(): continue
    try:
        comm=(p/"comm").read_text().strip()
        status=(p/"status").read_text()
    except (FileNotFoundError,ProcessLookupError): continue
    state=next((x.split()[1] for x in status.splitlines() if x.startswith("State:")),None)
    if state is None: raise RuntimeError("unknown process state")
    if state=="Z": continue
    if comm.startswith("VLLM::") or comm=="vllm": live.append((p.name,comm))
print(live)
sys.exit(bool(live))
' >"${OUT}/stop_process_probe.txt" || return $?
    npu-smi info >"${OUT}/stop_npu_probe.txt" || return $?
    python3 - "${OUT}/stop_npu_probe.txt" <<'PY'
import pathlib,re,sys
s=pathlib.Path(sys.argv[1]).read_text()
used=[int(x) for x in re.findall(r"(\d+)\s*/\s*65536",s)]
assert len(used)==8 and max(used)<6144 and "VLLM" not in s
print("stop idle HBM",used)
PY
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
    if [[ -f "${OUT}/patch_state/manifest.json" ]]; then
        if [[ "${stop_rc}" == 0 && "${verify_rc}" == 0 ]]; then
            python3 "${ROOT}/scripts/loop079_row_capture_patch.py" restore \
                --state-dir "${OUT}/patch_state" --record "${OUT}/restore.json" \
                >"${OUT}/restore.log" 2>&1
            restore_rc=$?
        else
            restore_rc=skipped_unproven_stop
            echo "Refusing source restore while service stop is unproved" >&2
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
trap 'exit 130' INT
trap 'exit 143' TERM
docker exec "${CONTAINER}" bash -lc 'cd /data/wio/Inference_Foundry && python3 -c "from scripts import loop079_row_capture; print(loop079_row_capture.__file__)"' >"${OUT}/module_import.txt"
python3 "${ROOT}/scripts/loop079_row_capture_patch.py" check --record "${OUT}/source_check.json"
sha256sum "${SOURCES[@]}" >"${OUT}/source_before.sha256"
python3 "${ROOT}/scripts/loop079_row_capture_patch.py" install \
    --state-dir "${OUT}/patch_state" --record "${OUT}/install.json"
RUN_STARTED=1
docker exec \
    -e MAX_MODEL_LEN=1048576 -e RUN_TS="${RUN_TS}" \
    -e EXTREME_RUNTIME_RUN_DIR="${OUT}/runtime" \
    -e EXTREME_RUNTIME_SERVE=1 -e EXTREME_RUNTIME_RESERVE_TOKENS=1088 \
    -e EXTREME_NATIVE_TARGET_METADATA=1 -e EXTREME_RUNTIME_TARGET_GRAPH=1 \
    -e EXTREME_DSPARK_SLOT_REFRESH=1 -e EXTREME_TARGET_METADATA_STATIC_KV_MAX=1 \
    -e EXTREME_BOUND_ROW_CAPTURE_DIR="${OUT}/capture" \
    "${CONTAINER}" bash -lc "cd ${ROOT} && bash scripts/serve.sh > ${OUT}/launcher.log 2>&1"
for i in $(seq 1 180); do
    if curl -fsS --max-time 3 http://127.0.0.1:8080/health >/dev/null 2>&1; then break; fi
    if [[ -f "${LOG}" ]] && grep -q "rtsMallocHost execution failed" "${LOG}"; then
        echo "service startup pinned Host OOM" >&2; exit 1
    fi
    sleep 10
done
curl -fsS --max-time 5 http://127.0.0.1:8080/health >/dev/null
docker exec "${CONTAINER}" bash -lc "cd ${ROOT} && python3 scripts/bench.py --dataset ${DATASET} --out ${OUT}/warmup48.json --limit 48 --concurrency 12 --max-tokens 1024 >${OUT}/warmup48.log 2>&1"
docker exec "${CONTAINER}" bash -lc "cd ${ROOT} && python3 scripts/bench.py --dataset ${DATASET} --out ${OUT}/bench.json --limit 12 --concurrency 12 --max-tokens 1024 >${OUT}/bench.log 2>&1"
POSTS=$(grep -c "POST /v1/chat/completions" "${LOG}")
printf '%s\n' "${POSTS}" >"${OUT}/server_post_count.txt"
test "${POSTS}" -eq 60
CAPTURE_FILES=$(find "${OUT}/capture" -name 'rank*_cohort*.json' -type f | wc -l)
printf '%s\n' "${CAPTURE_FILES}" >"${OUT}/capture_file_count.txt"
test "${CAPTURE_FILES}" -eq 40
python3 - "${OUT}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
for name,count in (("warmup48.json",48),("bench.json",12)):
 d=json.loads((root/name).read_text())
 assert len(d["requests"])==count
 assert all(r["error"] is None and r["output_tokens"]==1024 for r in d["requests"])
assert len(list((root/"runtime").glob("rank*_cohort*.json")))==40
PY
python3 "${ROOT}/scripts/loop079_row_capture_validate.py" "${OUT}/capture" \
    --output "${OUT}/validation.json"
