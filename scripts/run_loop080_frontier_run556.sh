#!/usr/bin/env bash
# Guarded selected-frontier diagnostic: warmup48 + measured48, never formal TPS.
set -euo pipefail

ROOT=/data/wio/Inference_Foundry
OUT=${ROOT}/evidence/20260927_loop080_bound/run556
CONTAINER=vllm-ascend26-dsv4f-w4a8
RUN_TS=LOOP080-RUN556
DATASET=/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl
LOG=${ROOT}/logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_${RUN_TS}.log
FRAME=/data/wio/vllm_ascend_26/framework
SOURCES=(
    "${FRAME}/vllm-ascend/vllm_ascend/patch/platform/patch_kv_delivery_preemption.py"
    "${FRAME}/vllm-ascend/vllm_ascend/worker/model_runner_v1.py"
    "${ROOT}/runtime/fixed_serving.py"
    "${ROOT}/runtime/extreme_decode.py"
    "${FRAME}/vllm/vllm/v1/engine/output_processor.py"
    "${FRAME}/vllm/vllm/entrypoints/openai/chat_completion/serving.py"
)
PATCH=${ROOT}/scripts/loop079_formal_ledger_patch.py
FRONTIER_PATCH=${ROOT}/scripts/loop080_frontier_patch.py
SCRIPTS=(
    "${ROOT}/scripts/loop079_formal_ledger_server.py"
    "${ROOT}/scripts/loop079_formal_ledger_patch.py"
    "${ROOT}/scripts/loop080_frontier_server.py"
    "${ROOT}/scripts/loop080_frontier_patch.py"
    "${ROOT}/scripts/loop080_frontier_client.py"
    "${ROOT}/scripts/loop080_frontier_client_validate.py"
    "${ROOT}/scripts/loop080_frontier_final_admit.py"
    "${ROOT}/scripts/run_loop080_frontier_run556.sh"
    "${ROOT}/scripts/loop079_formal_ledger_phase.py"
    "${ROOT}/scripts/loop079_formal_ledger_phase_barrier.py"
    "${ROOT}/scripts/loop079_formal_ledger_client_validate.py"
    "${ROOT}/scripts/loop079_formal_ledger_server_validate.py"
    "${ROOT}/scripts/loop079_formal_ledger_final_admit.py"
)
MARKER=${OUT}/phase.json
TRANSITIONS=${OUT}/phase_transitions.jsonl
LEDGER=${OUT}/ledger
FRONTIER=${OUT}/frontier
RUN_STARTED=0

exec 9>/tmp/loop080_frontier.lock
if ! flock -n 9; then
    echo 'another Loop080 acquisition holds the lock' >&2; exit 75
fi
if [[ -e "${OUT}" ]]; then
    echo "run-specific output exists: ${OUT}" >&2; exit 75
fi
mkdir -m 700 "${OUT}"
mkdir "${OUT}/runtime" "${LEDGER}" "${FRONTIER}"

verify_stopped() {
    if curl -fsS --max-time 2 http://127.0.0.1:8080/health >/dev/null 2>&1; then
        echo 'service still healthy' >&2; return 1
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
' >"${OUT}/stop_process_probe.txt" || return 1
    npu-smi info >"${OUT}/stop_npu_probe.txt" || return 1
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
    local stop_rc=0 verify_rc=0 restore_rc=not_needed frontier_restore_rc=not_needed sha_rc=0 compare_rc=not_checked script_sha_rc=0 script_compare_rc=not_checked final_rc=${run_rc}
    if [[ "${RUN_STARTED}" == 1 ]]; then
        bash /data/wio/vllm_ascend_26/scripts/99_stop_service.sh >"${OUT}/stop.log" 2>&1
        stop_rc=$?
    fi
    if [[ "${stop_rc}" == 0 ]]; then
        verify_stopped >"${OUT}/stop_verify.log" 2>&1
        verify_rc=$?
    else
        verify_rc=not_attempted
    fi
    if [[ -f "${OUT}/frontier_patch_state/manifest.json" ]]; then
        if [[ "${stop_rc}" == 0 && "${verify_rc}" == 0 ]]; then
            python3 "${FRONTIER_PATCH}" restore --state-dir "${OUT}/frontier_patch_state" \
                --record "${OUT}/frontier_restore.json" --offline-confirmed >"${OUT}/frontier_restore.log" 2>&1
            frontier_restore_rc=$?
        else
            frontier_restore_rc=skipped_unproven_stop
        fi
    fi
    if [[ -f "${OUT}/patch_state/manifest.json" ]]; then
        if [[ "${stop_rc}" == 0 && "${verify_rc}" == 0 ]]; then
            python3 "${PATCH}" restore --state-dir "${OUT}/patch_state" \
                --record "${OUT}/restore.json" --offline-confirmed >"${OUT}/restore.log" 2>&1
            restore_rc=$?
        else
            restore_rc=skipped_unproven_stop
        fi
    fi
    sha256sum "${SOURCES[@]}" >"${OUT}/source_after.sha256"
    sha_rc=$?
    if [[ "${sha_rc}" == 0 && -f "${OUT}/source_before.sha256" ]]; then
        cmp -s "${OUT}/source_before.sha256" "${OUT}/source_after.sha256"
        compare_rc=$?
    fi
    sha256sum "${SCRIPTS[@]}" >"${OUT}/scripts_after.sha256"
    script_sha_rc=$?
    if [[ "${script_sha_rc}" == 0 && -f "${OUT}/scripts_before.sha256" ]]; then
        cmp -s "${OUT}/scripts_before.sha256" "${OUT}/scripts_after.sha256"
        script_compare_rc=$?
    fi
    if [[ "${stop_rc}" != 0 || "${verify_rc}" != 0 || "${sha_rc}" != 0 || "${compare_rc}" != 0 || "${script_sha_rc}" != 0 || "${script_compare_rc}" != 0 || ( "${restore_rc}" != not_needed && "${restore_rc}" != 0 ) || ( "${frontier_restore_rc}" != not_needed && "${frontier_restore_rc}" != 0 ) ]]; then
        if [[ "${final_rc}" == 0 ]]; then final_rc=1; fi
    fi
    printf 'run_exit=%s\nstop_exit=%s\nstop_verify_exit=%s\nrestore_exit=%s\nfrontier_restore_exit=%s\nsha_exit=%s\nsha_compare_exit=%s\nscript_sha_exit=%s\nscript_compare_exit=%s\npre_admission_exit=%s\n' \
        "${run_rc}" "${stop_rc}" "${verify_rc}" "${restore_rc}" "${frontier_restore_rc}" "${sha_rc}" "${compare_rc}" "${script_sha_rc}" "${script_compare_rc}" "${final_rc}" \
        | tee "${OUT}/cleanup_status.txt" >&2
    python3 "${ROOT}/scripts/loop080_frontier_final_admit.py" \
        --run-dir "${OUT}" --output "${OUT}/final_admission.json" \
        >"${OUT}/final_validate.log" 2>&1
    local admit_rc=$?
    if [[ "${admit_rc}" != 0 && "${final_rc}" == 0 ]]; then final_rc=1; fi
    printf 'admission_exit=%s\nfinal_exit=%s\n' "${admit_rc}" "${final_rc}" \
        | tee -a "${OUT}/cleanup_status.txt" >&2
    exit "${final_rc}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if curl -fsS --max-time 2 http://127.0.0.1:8080/health >/dev/null 2>&1; then
    echo 'service already healthy' >&2; exit 75
fi
verify_stopped >"${OUT}/preflight_stopped.log" 2>&1
if pgrep -af '[p]ython3 scripts/loop080_frontier_client.py' | grep -v '<defunct>' >"${OUT}/orphan_bench.txt"; then
    echo 'orphan benchmark process' >&2; exit 75
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
printf '%s  %s\n' \
    '01e33fea82c60cc301f34a4dc1fe5651511f319e3ac808d7a076b24ea92dd7f6' "${ROOT}/scripts/loop080_frontier_server.py" \
    '386817d863a9984b8fbd448a346c0b7c0691787975a4d67be30441bad41b4467' "${FRONTIER_PATCH}" \
    | sha256sum -c - >"${OUT}/reviewed_frontier_sha.log"
sha256sum "${SOURCES[@]}" >"${OUT}/source_before.sha256"
sha256sum "${SCRIPTS[@]}" >"${OUT}/scripts_before.sha256"
python3 "${PATCH}" check --record "${OUT}/patch_check.json"
python3 "${FRONTIER_PATCH}" check --record "${OUT}/frontier_patch_check.json"
python3 "${ROOT}/scripts/loop079_formal_ledger_phase.py" warmup \
    --run-id "${RUN_TS}" --marker "${MARKER}" --transitions "${TRANSITIONS}" \
    >"${OUT}/phase_warmup.json"
python3 "${PATCH}" install --state-dir "${OUT}/patch_state" \
    --record "${OUT}/install.json" --offline-confirmed >"${OUT}/install.log"
python3 "${FRONTIER_PATCH}" install --state-dir "${OUT}/frontier_patch_state" \
    --record "${OUT}/frontier_install.json" --offline-confirmed >"${OUT}/frontier_install.log"

RUN_STARTED=1
docker exec \
    -e MAX_MODEL_LEN=1048576 -e RUN_TS="${RUN_TS}" \
    -e EXTREME_RUNTIME_RUN_DIR="${OUT}/runtime" \
    -e EXTREME_RUNTIME_SERVE=1 -e EXTREME_RUNTIME_RESERVE_TOKENS=1088 \
    -e EXTREME_NATIVE_TARGET_METADATA=1 -e EXTREME_RUNTIME_TARGET_GRAPH=1 \
    -e EXTREME_DSPARK_SLOT_REFRESH=1 -e EXTREME_TARGET_METADATA_STATIC_KV_MAX=1 \
    -e EXTREME_FORMAL_LEDGER_DIR="${LEDGER}" \
    -e EXTREME_FORMAL_LEDGER_RUN_ID="${RUN_TS}" \
    -e EXTREME_FORMAL_LEDGER_PHASE_FILE="${MARKER}" \
    -e EXTREME_FRONTIER_DIR="${FRONTIER}" \
    "${CONTAINER}" bash -lc "cd ${ROOT} && bash scripts/serve.sh > ${OUT}/launcher.log 2>&1"
for i in $(seq 1 180); do
    if curl -fsS --max-time 3 http://127.0.0.1:8080/health >/dev/null 2>&1; then break; fi
    if [[ -f "${LOG}" ]] && grep -q 'rtsMallocHost execution failed' "${LOG}"; then
        echo 'service startup pinned Host OOM' >&2; exit 1
    fi
    sleep 10
done
curl -fsS --max-time 5 http://127.0.0.1:8080/health >/dev/null

docker exec "${CONTAINER}" bash -lc "cd ${ROOT} && python3 scripts/loop080_frontier_client.py --dataset ${DATASET} --out-dir ${OUT}/warmup_client --phase warmup --limit 48 --concurrency 12 --max-tokens 1024 >${OUT}/warmup.log 2>&1"
docker exec "${CONTAINER}" bash -lc "cd ${ROOT} && python3 scripts/loop079_formal_ledger_client_validate.py --dataset ${DATASET} --warmup-dir ${OUT}/warmup_client --warmup-only --output ${OUT}/warmup_client_admission.json >${OUT}/warmup_validate.log 2>&1"
python3 "${ROOT}/scripts/loop079_formal_ledger_phase_barrier.py" \
    --ledger-dir "${LEDGER}" --client-report "${OUT}/warmup_client_admission.json" \
    --phase-marker "${MARKER}" --run-id "${RUN_TS}" \
    --output "${OUT}/warmup_phase_barrier.json" >"${OUT}/warmup_barrier.log" 2>&1
python3 "${ROOT}/scripts/loop079_formal_ledger_phase.py" measured \
    --run-id "${RUN_TS}" --marker "${MARKER}" --transitions "${TRANSITIONS}" \
    --completed-client-report "${OUT}/warmup_client_admission.json" >"${OUT}/phase_measured.json"

docker exec "${CONTAINER}" bash -lc "cd ${ROOT} && python3 scripts/loop080_frontier_client.py --dataset ${DATASET} --out-dir ${OUT}/measured_client --phase measured --limit 48 --concurrency 12 --max-tokens 1024 >${OUT}/measured.log 2>&1"
docker exec "${CONTAINER}" bash -lc "cd ${ROOT} && python3 scripts/loop079_formal_ledger_client_validate.py --dataset ${DATASET} --warmup-dir ${OUT}/warmup_client --measured-dir ${OUT}/measured_client --output ${OUT}/client_admission.json >${OUT}/client_validate.log 2>&1"
python3 "${ROOT}/scripts/loop080_frontier_client_validate.py" --phase-dir "${OUT}/measured_client" \
    --output "${OUT}/frontier_client_admission.json" >"${OUT}/frontier_client_validate.log" 2>&1

POSTS=$(grep -c 'POST /v1/chat/completions' "${LOG}")
printf '%s\n' "${POSTS}" >"${OUT}/server_post_count.txt"
test "${POSTS}" -eq 96
python3 "${ROOT}/scripts/loop079_formal_ledger_server_validate.py" \
    --ledger-dir "${LEDGER}" --client-report "${OUT}/client_admission.json" \
    --warmup-report "${OUT}/warmup_client_admission.json" \
    --warmup-dir "${OUT}/warmup_client" \
    --measured-dir "${OUT}/measured_client" \
    --phase-transitions "${TRANSITIONS}" --run-id "${RUN_TS}" \
    --output "${OUT}/server_admission.json" >"${OUT}/server_validate.log" 2>&1
