#!/usr/bin/env bash
# Metadata Graph OFF/ON/OFF with admitted input-token and initial-hash caches ON.
set -euo pipefail

ROOT=/data/wio/Inference_Foundry
case ${1:-} in
    --preflight-only) OUT=${ROOT}/evidence/20260929_loop081_bound/run675/preflight ;;
    '') OUT=${ROOT}/evidence/20260929_loop081_bound/run675/live ;;
    *) echo 'usage: run_loop081_stack_metadata_run675.sh [--preflight-only]' >&2; exit 64 ;;
esac
CONTAINER=vllm-ascend26-dsv4f-w4a8
DATASET=/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl
PATCH=${ROOT}/scripts/install_input_token_cache.py
HASH_PATCH=${ROOT}/scripts/loop081_prefix_hash_install.py
META_PATCH=${ROOT}/scripts/loop081_stack_metadata_install_run675.py
SOURCES=(
    ${ROOT}/serving/cohort_mode.py
    ${ROOT}/experiments/prefix_hash_cache/router.py
    ${ROOT}/experiments/prefix_hash_cache/cache.py
    /data/wio/vllm_ascend_26/framework/vllm/vllm/v1/core/kv_cache_utils.py
    ${ROOT}/serving/__init__.py
    ${ROOT}/serving/input_token_cache.py
    /data/wio/vllm_ascend_26/framework/vllm/vllm/renderers/base.py
    /data/wio/vllm_ascend_26/framework/vllm/vllm/renderers/deepseek_v4.py
    /data/wio/vllm_ascend_26/framework/vllm/vllm/config/model.py
    ${ROOT}/runtime/target_metadata.py
    /data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py
)
SCRIPTS=(
    ${META_PATCH}
    ${ROOT}/scripts/loop081_metadata_graph_patch_run666.py
    ${ROOT}/scripts/loop081_stack_metadata_prepare_run675.py
    ${ROOT}/scripts/loop081_runtime_observer_install_run638.py
    ${ROOT}/scripts/loop081_stack_metadata_mode_gate_run675.py
    ${ROOT}/evidence/20260929_loop081_bound/run675/candidate_manifest.json
    ${ROOT}/evidence/20260929_loop081_bound/run675/candidate/metadata.py
    ${ROOT}/evidence/20260929_loop081_bound/run675/candidate/runner.py
    ${HASH_PATCH}
    ${ROOT}/scripts/loop081_prefix_hash_integration_run674.py
    ${PATCH}
    ${ROOT}/scripts/loop081_token_cache_integration_run673.py
    ${ROOT}/scripts/bench.py
    ${ROOT}/scripts/analyze_loop034_e2e.py
    ${ROOT}/scripts/serve.sh
    ${ROOT}/scripts/env.sh
    ${DATASET}
    ${ROOT}/scripts/run_loop081_stack_metadata_run675.sh
)
RUN_STARTED=0
CURRENT_RUN_TS=

exec 9>/tmp/loop080_frontier.lock
if ! flock -n 9; then
    echo 'another bound acquisition holds the lock' >&2; exit 75
fi
if [[ -e ${OUT} ]]; then
    echo "run-specific output exists: ${OUT}" >&2; exit 75
fi
mkdir -p "$(dirname "${OUT}")"
mkdir -m 700 "${OUT}"

verify_stopped() {
    if curl -fsS --max-time 2 http://127.0.0.1:8080/health >/dev/null 2>&1; then
        echo 'service is still healthy' >&2; return 1
    fi
    timeout 15s docker exec "${CONTAINER}" python3 -c '
import pathlib,sys
live=[]
for p in pathlib.Path("/proc").iterdir():
    if not p.name.isdigit(): continue
    try:
        comm=(p/"comm").read_text().strip()
        raw=(p/"cmdline").read_bytes()
        status=(p/"status").read_text()
    except (FileNotFoundError,ProcessLookupError): continue
    state=next((x.split()[1] for x in status.splitlines() if x.startswith("State:")),None)
    if state is None: raise RuntimeError("unknown process state")
    if state=="Z": continue
    argv=raw.split(b"\0")
    vllm_serve=(any(b"vllm" in arg for arg in argv)
                and any(arg==b"serve" for arg in argv))
    if comm.startswith("VLLM::") or comm=="vllm" or vllm_serve:
        live.append((p.name,comm))
print(live)
sys.exit(bool(live))
' || return 1
    local client_guard
    client_guard=$(timeout 15s docker exec -i "${CONTAINER}" python3 - "${OUT}" "${CURRENT_RUN_TS}" <<'PY'
import pathlib,sys,os
root=sys.argv[1].encode()
tag=('RUN_TS='+sys.argv[2]).encode() if sys.argv[2] else None
live=[]
for p in pathlib.Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name)==os.getpid(): continue
    try:
        raw=(p/'cmdline').read_bytes()
        status=(p/'status').read_text()
        env=(p/'environ').read_bytes()
    except (FileNotFoundError,ProcessLookupError): continue
    if not raw or 'State:\tZ' in status: continue
    argv=raw.split(b'\0')
    if any(b'loop080_frontier_client.py' in arg or arg.endswith(b'/bench.py') for arg in argv) or (tag and tag in env.split(b'\0')):
        live.append((p.name,raw[:160]))
print('owned clients',live)
print('RUN568_CLIENT_GUARD_EXECUTED')
sys.exit(bool(live))
PY
    ) || return 1
    [[ ${client_guard} == *RUN568_CLIENT_GUARD_EXECUTED* ]] || return 1
    # NPU process tables and HBM release can lag process exit. Keep the
    # restore gate bounded, but do not mistake a transient release for a leak.
    local idle=0
    for i in $(seq 1 30); do
        if timeout 15s npu-smi info >"${OUT}/stop_npu_probe.txt" && python3 - "${OUT}/stop_npu_probe.txt" <<'PY'
import pathlib,re,sys
s=pathlib.Path(sys.argv[1]).read_text()
used=[int(x) for x in re.findall(r"(\d+)\s*/\s*65536",s)]
assert len(used)==8 and max(used)<6144 and "VLLM" not in s
assert s.count("No running processes found in NPU") == 8
print("idle HBM",used)
PY
        then
            idle=1; break
        fi
        sleep 3
    done
    [[ ${idle} == 1 ]] || { echo 'NPU resources did not release after 30 bounded probes' >&2; return 1; }
}

stop_owned() {
    # Preflight proves this container had no VLLM or task client before launch.
    # Signal only matching processes inside this container; never use the
    # shared host's all-NPU PID kill fallback.
    local stop_output tag=${CURRENT_RUN_TS:?missing_run_tag}
    stop_output=$(timeout 90s docker exec -i "${CONTAINER}" python3 - "${OUT}" "${tag}" <<'PY'
import os,pathlib,signal,sys,time
root=sys.argv[1].encode()
tag=('RUN_TS='+sys.argv[2]).encode()
def targets():
    found=[]
    for p in pathlib.Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid(): continue
        try:
            comm=(p/'comm').read_text().strip()
            raw=(p/'cmdline').read_bytes()
            status=(p/'status').read_text()
            env=(p/'environ').read_bytes()
        except (FileNotFoundError,ProcessLookupError): continue
        if not raw or 'State:\tZ' in status: continue
        argv=raw.split(b'\0')
        owned_client=(any(b'loop080_frontier_client.py' in arg or arg.endswith(b'/bench.py') for arg in argv)
                      and any(root in arg for arg in argv))
        owned_vllm=(tag in env.split(b'\0') and
                    (comm.startswith('VLLM::') or comm=='vllm'
                     or (any(b'vllm' in arg for arg in argv)
                         and any(arg==b'serve' for arg in argv))))
        owned_helper=(tag in env.split(b'\0') and
                      b'multiprocessing.' in raw and b'python3' in raw)
        if owned_vllm or owned_helper or owned_client:
            found.append((int(p.name),comm,owned_client))
    return found
for sig in (signal.SIGTERM,signal.SIGKILL):
    current=targets()
    print('owned stop',sig,current,flush=True)
    for pid,_,_ in current:
        try: os.kill(pid,sig)
        except ProcessLookupError: pass
    for _ in range(30):
        if not targets(): break
        time.sleep(1)
if targets(): raise RuntimeError('owned service/client processes survived stop')
print('RUN568_OWNED_STOP_EXECUTED')
PY
    ) || return 1
    printf '%s\n' "${stop_output}"
    [[ ${stop_output} == *RUN568_OWNED_STOP_EXECUTED* ]]
}

verify_prestart() {
    verify_stopped
    timeout 15s docker exec "${CONTAINER}" python3 -c '
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
'
    python3 - <<'PY'
from pathlib import Path
s=Path('/proc/meminfo').read_text()
available=int(next(x.split()[1] for x in s.splitlines() if x.startswith('MemAvailable:')))
print('MemAvailable_kB',available)
assert available>256*1024*1024
PY
}

cleanup() {
    local run_rc=$?
    trap - EXIT
    set +e
    local final_rc=${run_rc} stop_rc=0 verify_rc=0 restore_rc=0 sha_rc=0 compare_rc=not_checked script_rc=0 script_compare_rc=not_checked
    if [[ ${RUN_STARTED} == 1 ]]; then
        stop_owned >"${OUT}/emergency_stop.log" 2>&1
        stop_rc=$?
    fi
    if [[ ${stop_rc} == 0 ]]; then
        verify_stopped >"${OUT}/final_stop_verify.log" 2>&1
        verify_rc=$?
    else
        verify_rc=not_attempted
    fi
    if [[ -f ${OUT}/meta_patch_state/manifest.json ]]; then
        if [[ ${stop_rc} == 0 && ${verify_rc} == 0 ]]; then
            python3 "${META_PATCH}" restore --state-dir "${OUT}/meta_patch_state" --record "${OUT}/meta_restore.json" --offline-confirmed >"${OUT}/meta_restore.log" 2>&1
            restore_rc=$?
        else
            restore_rc=skipped_unproven_stop
        fi
    fi
    if [[ -f ${OUT}/hash_patch_state/manifest.json ]]; then
        if [[ ${stop_rc} == 0 && ${verify_rc} == 0 ]]; then
            python3 "${HASH_PATCH}" restore --state-dir "${OUT}/hash_patch_state" --record "${OUT}/hash_restore.json" --offline-confirmed >"${OUT}/hash_restore.log" 2>&1
            local hash_restore_rc=$?
            if [[ ${hash_restore_rc} != 0 ]]; then restore_rc=${hash_restore_rc}; fi
        else
            restore_rc=skipped_unproven_stop
        fi
    fi
    if [[ -f ${OUT}/patch_state/manifest.json ]]; then
        if [[ ${stop_rc} == 0 && ${verify_rc} == 0 ]]; then
            python3 "${PATCH}" restore --state-dir "${OUT}/patch_state" --record "${OUT}/restore.json" --offline-confirmed >"${OUT}/restore.log" 2>&1
            local input_restore_rc=$?
            if [[ ${input_restore_rc} != 0 ]]; then restore_rc=${input_restore_rc}; fi
        else
            restore_rc=skipped_unproven_stop
        fi
    fi
    sha256sum "${SOURCES[@]}" >"${OUT}/source_after.sha256"
    sha_rc=$?
    if [[ ${sha_rc} == 0 && -f ${OUT}/source_before.sha256 ]]; then
        cmp -s "${OUT}/source_before.sha256" "${OUT}/source_after.sha256"
        compare_rc=$?
    fi
    sha256sum "${SCRIPTS[@]}" >"${OUT}/scripts_after.sha256"
    script_rc=$?
    if [[ ${script_rc} == 0 && -f ${OUT}/scripts_before.sha256 ]]; then
        cmp -s "${OUT}/scripts_before.sha256" "${OUT}/scripts_after.sha256"
        script_compare_rc=$?
    fi
    if [[ ${stop_rc} != 0 || ${verify_rc} != 0 || ${restore_rc} != 0 || ${sha_rc} != 0 || ${compare_rc} != 0 || ${script_rc} != 0 || ${script_compare_rc} != 0 ]]; then
        if [[ ${final_rc} == 0 ]]; then final_rc=1; fi
    fi
    printf 'run_exit=%s\nstop_exit=%s\nstop_verify_exit=%s\nrestore_exit=%s\nsource_sha_exit=%s\nsource_compare_exit=%s\nscript_sha_exit=%s\nscript_compare_exit=%s\nfinal_exit=%s\n' \
        "${run_rc}" "${stop_rc}" "${verify_rc}" "${restore_rc}" "${sha_rc}" "${compare_rc}" "${script_rc}" "${script_compare_rc}" "${final_rc}" \
        | tee "${OUT}/cleanup_status.txt" >&2
    exit "${final_rc}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

verify_prestart >"${OUT}/preflight_stopped.log" 2>&1
if pgrep -af '[p]ython3 scripts/bench.py' | grep -v '<defunct>' >"${OUT}/orphan_client.txt"; then
    echo 'orphan client process' >&2; exit 75
fi
# No profiler, competing diagnosis, page audit, DAG export, or stale observer env.
docker exec -i "${CONTAINER}" python3 - <<'PYENV' >"${OUT}/container_env_guard.json"
import json,os
banned=('EXTREME_RUNTIME_PROFILE_SCOPES','EXTREME_RUNTIME_PROFILE_DAG',
        'EXTREME_RUNTIME_DIAGNOSE','EXTREME_RUNTIME_CYCLE_PROFILE_DIR',
        'EXTREME_RUNTIME_DAG_DIR','EXTREME_SCHEDULE_ALIAS_DIR',
        'EXTREME_BOUND_LEDGER_DIR','EXTREME_BOUND_PACKET_DIR',
        'PROFILER_DIR','VLLM_CUSTOM_SCOPES_FOR_PROFILING',
        'EXTREME_METADATA_MODE_FILE','EXTREME_TARGET_METADATA_GRAPH','EXTREME_TARGET_METADATA_VERIFY_EVERY','EXTREME_TEXT_TOKEN_CACHE','EXTREME_INITIAL_PREFIX_HASH_CACHE','EXTREME_PREFIX_HASH_MODE_FILE')
present={key:os.environ[key] for key in banned if key in os.environ}
print(json.dumps({'banned_present':list(present)}))
assert not present, 'conflicting container environment'
PYENV
docker inspect "${CONTAINER}" --format '{"container_id":"{{.Id}}","image_id":"{{.Image}}","image_name":"{{.Config.Image}}","working_dir":"{{.Config.WorkingDir}}"}' >"${OUT}/container_identity.json"
docker exec -i -w "${ROOT}" "${CONTAINER}" python3 - <<'PYAPI' >"${OUT}/runtime_api_pin.json"
import json, pathlib, torch_npu
root=pathlib.Path('/usr/local/Ascend/ascend-toolkit/latest')
print(json.dumps({'torch_npu_version':torch_npu.__version__,
                  'torch_npu_file':torch_npu.__file__,
                  'cann_latest':str(root.resolve())}))
assert torch_npu.__version__=='2.10.0.post4'
assert root.resolve().name=='cann-9.1.0'
PYAPI
sha256sum "${SOURCES[@]}" >"${OUT}/source_before.sha256"
sha256sum "${SCRIPTS[@]}" >"${OUT}/scripts_before.sha256"
python3 "${PATCH}" check --record "${OUT}/patch_check.json" >"${OUT}/patch_check.log"
python3 "${PATCH}" install --state-dir "${OUT}/patch_state" --record "${OUT}/install.json" --offline-confirmed >"${OUT}/install.log"
python3 "${HASH_PATCH}" install --state-dir "${OUT}/hash_patch_state" --record "${OUT}/hash_install.json" --offline-confirmed >"${OUT}/hash_install.log" 2>&1
docker exec "${CONTAINER}" python3 -m py_compile /data/wio/vllm_ascend_26/framework/vllm/vllm/renderers/base.py >"${OUT}/installed_compile.log" 2>&1
python3 "${META_PATCH}" install --state-dir "${OUT}/meta_patch_state" --record "${OUT}/meta_install.json" --offline-confirmed >"${OUT}/meta_install.log" 2>&1
if [[ ${1:-} == --preflight-only ]]; then
    python3 "${ROOT}/scripts/loop081_stack_metadata_mode_gate_run675.py" >"${OUT}/mode_gate.json"
    echo 'Run675 selector/source roundtrip complete; reused Run674 exact cache semantics and Run666 Graph arithmetic.'
    exit 0
fi

# One resident service; all modes share the same routing overhead and source.
set_mode() {
    python3 - "${OUT}/metadata_mode.json" "$1" "$2" "$3" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);tmp=p.with_suffix('.tmp')
tmp.write_text(json.dumps({'phase':sys.argv[2],'enabled':sys.argv[3]=='1','version':int(sys.argv[4])}))
tmp.replace(p)
PY
}
CURRENT_RUN_TS=LOOP081-RUN675-SHARED
log=${ROOT}/logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_${CURRENT_RUN_TS}.log
mkdir "${OUT}/runtime"
set_mode off_a 0 1
printf '%s\n' '{"phase":"on","enabled":true,"version":1}' >"${OUT}/hash_mode.json"
RUN_STARTED=1
docker exec -e MAX_MODEL_LEN=1048576 -e RUN_TS="${CURRENT_RUN_TS}" \
    -e EXTREME_RUNTIME_RUN_DIR="${OUT}/runtime" \
    -e EXTREME_RUNTIME_SERVE=1 -e EXTREME_RUNTIME_RESERVE_TOKENS=1088 \
    -e EXTREME_NATIVE_TARGET_METADATA=1 -e EXTREME_RUNTIME_TARGET_GRAPH=1 \
    -e EXTREME_DSPARK_SLOT_REFRESH=1 -e EXTREME_TARGET_METADATA_STATIC_KV_MAX=1 \
    -e 'EXTRA_SERVE_ARGS=--renderer-num-workers 4' -e EXTREME_TEXT_TOKEN_CACHE=1 \
    -e EXTREME_INITIAL_PREFIX_HASH_CACHE=1 -e EXTREME_PREFIX_HASH_MODE_FILE="${OUT}/hash_mode.json" \
    -e EXTREME_METADATA_MODE_FILE="${OUT}/metadata_mode.json" \
    "${CONTAINER}" bash -lc "cd ${ROOT} && bash scripts/serve.sh > ${OUT}/launcher.log 2>&1"
ready=0
for i in $(seq 1 180); do
    if curl -fsS --max-time 3 http://127.0.0.1:8080/health >/dev/null 2>&1; then ready=1; break; fi
    if [[ -f ${log} ]] && grep -q 'rtsMallocHost execution failed' "${log}"; then exit 1; fi
    sleep 10
done
[[ ${ready} == 1 ]]
grep -Eq 'Foundry initial prefix hash cache active block_size=2([^0-9]|$)' "${log}"
docker exec -i "${CONTAINER}" python3 - "${CURRENT_RUN_TS}" <<'PY' >"${OUT}/loaded_renderer.json"
import pathlib,sys,json
rows=[];tag=('RUN_TS='+sys.argv[1]).encode()
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x];env=(p/'environ').read_bytes().split(b'\0')
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 if tag in env and 'serve' in argv:
  assert argv.count('--renderer-num-workers')==1 and argv[argv.index('--renderer-num-workers')+1]=='4'
  rows.append({'pid':int(p.name),'argv':argv})
assert len(rows)==1
print(json.dumps({'expected_workers':4,'processes':rows},indent=2))
PY
run_phase() {
    local phase=$1 on=$2 version=$3 offset=$4
    local arm_dir=${OUT}/${phase}
    mkdir "${arm_dir}" "${arm_dir}/runtime"
    set_mode "${phase}" "${on}" "${version}"
    cp "${OUT}/loaded_renderer.json" "${arm_dir}/loaded_renderer.json"
    for label in warmup bench48_1 bench48_2 bench48_3; do
        timeout --signal=TERM --kill-after=30s 960s docker exec -e RUN_TS="${CURRENT_RUN_TS}" "${CONTAINER}" bash -lc \
            "cd ${ROOT} && timeout --signal=TERM --kill-after=30s 900s python3 scripts/bench.py --dataset ${DATASET} --out ${arm_dir}/${label}.json --limit 48 --concurrency 12 --max-tokens 1024 >${arm_dir}/${label}.log 2>&1"
    done
    grep -q 'Foundry bounded input-token cache active for DeepseekV4Renderer' "${log}"
    python3 - "${OUT}" "${phase}" "${on}" "${version}" "${offset}" <<'PY'
import json,sys,shutil,statistics
from pathlib import Path
root=Path(sys.argv[1]);phase=sys.argv[2];on=sys.argv[3]=='1';version=int(sys.argv[4]);offset=int(sys.argv[5]);p=root/phase
summaries=[]
for name in ('warmup','bench48_1','bench48_2','bench48_3'):
 b=json.loads((p/(name+'.json')).read_text());s=b['summary']
 assert s['n']==s['success']==len(b['requests'])==48 and s['fail']==0 and s['concurrency']==12 and s['max_tokens']==1024
 assert all(r['output_tokens']==1024 and r['error'] is None for r in b['requests'])
 if name!='warmup':summaries.append(s)
for c in range(offset+1,offset+17):
 rows=[]
 for rank in range(8):
  f=root/'runtime'/f'rank{rank}_cohort{c}.json';r=json.loads(f.read_text());rows.append(r)
  assert r['rank']==rank and r['cohort']==c and r['pass'] and r['host_mirror_exact']
  assert r['metadata_graph_mode']=={'phase':phase,'enabled':on,'version':version}
  assert r['metadata_graph_capture']==on
  assert r['metadata_graph_replays']==(r['cycles'] if on else 0)
  assert r['generated_output_counts']==[1024]*12 and r['oracle_target_calls_after_handoff']==0 and 'FULL' in r['target_graph_mode']
  shutil.copyfile(f,p/'runtime'/f.name)
 for field in ('cycles','staged_output_counts','overshoot_tokens','acceptance_window_means'):
  assert all(r[field]==rows[0][field] for r in rows)
route=[json.loads(line) for line in (root/'hash_routing.jsonl').read_text().splitlines()]
assert len(route)==4*version and [r['initial_requests'] for r in route]==list(range(48,192*version+1,48))
assert all(r['enabled'] and r['phase']=='on' and r['version']==1 for r in route)
assert route[-1]['memo_misses']==route[0]['memo_misses']==48
assert route[-1]['memo_hits']==192*version-48
route=route[-4:]
result={'pass':True,'offset':offset,'runs':summaries,'median_tps':statistics.median(s['output_tps'] for s in summaries),'routing':route}
(p/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(phase,'FORMAL_4X48_ALL8_ADMITTED',result['median_tps'],flush=True)
PY
    posts=$(grep -c 'POST /v1/chat/completions' "${log}")
    [[ ${posts} -eq $((192*version)) ]]
    printf '%s\n' "${posts}" >"${arm_dir}/cumulative_post_count.txt"
}
run_phase off_a 0 1 0
run_phase on 1 2 16
run_phase off_b 0 3 32
stop_owned >"${OUT}/stop.log" 2>&1
verify_stopped >"${OUT}/stop_verify.log" 2>&1
[[ $(grep -c 'POST /v1/chat/completions' "${log}") -eq 576 ]]
RUN_STARTED=0
