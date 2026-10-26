#!/usr/bin/env bash
# All8 BF16 AG Graph fresh-output completion preflight. No TPS or Bound claim.
set -euo pipefail
ROOT=/data/wio/Inference_Foundry
OUT=${ROOT}/evidence/20260928_loop080_bound/run579/event_gate
CONTAINER=vllm-ascend26-dsv4f-w4a8
SCRIPT=${ROOT}/scripts/loop080_mixed_event_gate_run579.py
TAG=RUN579-EVENT-GATE-A
exec 9>/tmp/loop080_frontier.lock
flock -n 9 || { echo 'bound lock busy' >&2; exit 75; }
[[ ! -e ${OUT} ]] || { echo 'run579 output exists' >&2; exit 75; }
mkdir -p "$(dirname "${OUT}")"
mkdir -m 700 "${OUT}"
verify_idle() {
    ! curl -fsS --max-time 2 http://127.0.0.1:8080/health >/dev/null 2>&1
    npu-smi info >"${OUT}/npu_probe.txt"
    python3 - "${OUT}/npu_probe.txt" <<'PY'
import pathlib,re,sys
s=pathlib.Path(sys.argv[1]).read_text()
used=[int(x) for x in re.findall(r'(\d+)\s*/\s*65536',s)]
assert len(used)==8 and max(used)<6144
assert s.count('No running processes found in NPU')==8
PY
}
stop_owned() {
    docker exec -i "${CONTAINER}" python3 - "${TAG}" <<'PY'
import os,pathlib,signal,sys,time
tag=('EXTREME_RUN579_TAG='+sys.argv[1]).encode()
def targets():
    hits=[]
    for p in pathlib.Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:
            env=(p/'environ').read_bytes().split(b'\0')
            status=(p/'status').read_text()
        except (FileNotFoundError,ProcessLookupError):continue
        if tag in env and 'State:\tZ' not in status:hits.append(int(p.name))
    return hits
for sig in (signal.SIGTERM,signal.SIGKILL):
    for pid in targets():
        try:os.kill(pid,sig)
        except ProcessLookupError:pass
    for _ in range(20):
        if not targets():break
        time.sleep(0.5)
assert not targets(), 'tagged processes survived'
print('tagged processes stopped')
PY
}
cleanup() {
    local run_rc=$? stop_rc=0 idle_rc=0 source_rc=0 final_rc
    trap - EXIT
    set +e
    stop_owned >"${OUT}/stop.log" 2>&1; stop_rc=$?
    verify_idle >"${OUT}/final_idle.log" 2>&1; idle_rc=$?
    sha256sum "${SCRIPT}" >"${OUT}/script_after.sha256"
    cmp -s "${OUT}/script_before.sha256" "${OUT}/script_after.sha256"; source_rc=$?
    final_rc=${run_rc}
    if [[ ${stop_rc} != 0 || ${idle_rc} != 0 || ${source_rc} != 0 ]]; then final_rc=1; fi
    printf 'run_exit=%s\nstop_exit=%s\nidle_exit=%s\nscript_compare_exit=%s\nfinal_exit=%s\n' \
        "${run_rc}" "${stop_rc}" "${idle_rc}" "${source_rc}" "${final_rc}" \
        >"${OUT}/cleanup_status.txt"
    exit "${final_rc}"
}
trap cleanup EXIT
verify_idle >"${OUT}/preflight_idle.log" 2>&1
sha256sum "${SCRIPT}" >"${OUT}/script_before.sha256"
timeout --signal=TERM --kill-after=20s 260s docker exec -e EXTREME_RUN579_TAG="${TAG}" \
    "${CONTAINER}" bash -lc "cd ${ROOT} && timeout --signal=TERM --kill-after=10s 240s torchrun --standalone --nproc_per_node=8 scripts/loop080_mixed_event_gate_run579.py" \
    >"${OUT}/torchrun.log" 2>&1
python3 - "${OUT}" "${TAG}" <<'PY'
import json,pathlib,sys
out=pathlib.Path(sys.argv[1]);tag=sys.argv[2]
paths=sorted(out.glob('rank*.json'))
assert len(paths)==8
for rank in range(8):
    row=json.loads((out/f'rank{rank}.json').read_text())
    assert row['status']=='fresh_hccl_graph_join_pass' and row['run_tag']==tag
    assert row['rank']==rank and row['world']==8 and row['not_timed'] is True
    assert [x['generation'] for x in row['generations']]==[11,21,31]
    assert all(x['mismatches']==0 and x['sum']==x['expected'] for x in row['generations'])
print('all8 fresh-output join PASS')
PY
