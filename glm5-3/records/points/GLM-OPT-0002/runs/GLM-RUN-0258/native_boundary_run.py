"""Exactly one original-stock PD request under the unique controller."""
from pathlib import Path
import importlib.util
import json
import selectors
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / "functional_plan.json").read_text())
ROLES = json.loads((ROOT / "resident_roles.json").read_text())
old = ROOT.parent / "GLM-RUN-0251/pd_gather_compare.py"
spec = importlib.util.spec_from_file_location("original_PD_client", old)
x = importlib.util.module_from_spec(spec)
spec.loader.exec_module(x)
x.ROOT = x.m.ROOT = ROOT
x.PLAN = PLAN


def guard(host):
    oldguard = ROOT.parent / "GLM-RUN-0256/pd_mc2_compare.py"
    cmd = ["/usr/bin/python3", str(oldguard), "guardfresh" if host == "167" else "guard", host]
    if host == "167":
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "root@172.16.10.167", shlex.join(cmd)]
    a = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    assert a.returncode == 0, a.stderr[-2500:]
    row = json.loads(a.stdout)
    assert all(row["root"][k] == ROLES[host]["root"][k] for k in ("pid", "boot_id", "start_ticks"))
    assert row["device_owners"] == ROLES[host]["device_owners"]
    assert row["worker_start_ticks"] == ROLES[host]["worker_start_ticks"]
    return row


def read_ack(observer, timeout=30):
    sel = selectors.DefaultSelector()
    sel.register(observer.stdout, selectors.EVENT_READ)
    try:
        assert sel.select(timeout), 'observer IPC timeout'
        return json.loads(observer.stdout.readline())
    finally:
        sel.close()


def native_witness():
    command = ['/usr/bin/python3', str(ROOT.parent / 'GLM-RUN-0256/pd_mc2_compare.py'), 'stockwitness', '167']
    a = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                        'root@172.16.10.167', shlex.join(command)], capture_output=True, text=True, timeout=60)
    assert a.returncode == 0, a.stderr[-2500:]
    return json.loads(a.stdout)


def workflow():
    owner = json.loads(Path("/data/tiankuan/wio/glm52-pd/controller-owner.json").read_text())
    assert owner["run_id"] == "GLM-RUN-0258" and owner["status"] == "running" and x.m.live(owner["owner"])
    x.m.write("guards_before.json", {h: guard(h) for h in ("166", "167")})
    x.m.write('native_before.json', native_witness())
    err = (ROOT / "observer.stderr.log").open("xb")
    observer = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "root@172.16.10.167",
        shlex.join(["/usr/bin/python3", str(ROOT / "sample_native_boundaries.py"), str(ROOT)])],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True)
    started = False
    try:
        ready = read_ack(observer)
        assert ready["ready"]
        x.m.write("observer_ready.json", ready)
        original_fetch = x.fetch

        def marked_fetch(url, *args, **kwargs):
            nonlocal started
            response = original_fetch(url, *args, **kwargs)
            if url.endswith(":9081/v1/chat/completions"):
                status, raw = response
                (ROOT / 'P_export_before_observer.raw').write_bytes(raw)
                value = json.loads(raw)
                assert status == 200 and value['usage']['completion_tokens'] == 1
                kv = value['kv_transfer_params']
                assert kv['do_remote_prefill'] and kv['remote_host'] == '172.16.10.166'
                assert kv['remote_engine_id'].startswith('glm53-P-run249-') and kv['remote_port'] == 36000
                assert kv['remote_dcp_size'] == 16 and kv['remote_pcp_size'] == 1 and all(kv['remote_block_ids'])
                x.m.write('P_export_validated_before_observer.json', value)
                observer.stdin.write("go\n")
                observer.stdin.flush()
                ack = read_ack(observer)
                assert ack["go"]
                x.m.write("observer_go.json", ack)
                started = True
            return response

        x.fetch = marked_fetch
        result = x.request("native_boundaries")
        assert started
        observer.stdin.write("done\n")
        observer.stdin.flush()
        done = read_ack(observer)
        assert done["done"] and done["boundary_task_clock"]
        x.m.write("observer_done.json", done)
        assert observer.wait(timeout=12) == 0
    finally:
        errors = []
        try:
            if observer.poll() is None:
                observer.stdin.close()
                try:
                    observer.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    observer.terminate()
                    try:
                        observer.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        observer.kill()
                        observer.wait(timeout=5)
        except Exception as error:
            errors.append(dict(stage='owned_SSH_cleanup', error=repr(error)))
        after = {}
        for host in ('166', '167'):
            try:
                after[host] = guard(host)
            except Exception as error:
                errors.append(dict(stage='guard_'+host, error=repr(error)))
        x.m.write('guards_after.json', after)
        try:
            x.m.write('native_after.json', native_witness())
        except Exception as error:
            errors.append(dict(stage='native_after', error=repr(error)))
        x.m.write('cleanup_errors.json', errors)
        assert not errors, errors
    x.m.write("final_status.json", dict(status="completed", request=result,
        native_boundary_observation=True, performance_claim=False, NPU_profiler_active=False,
        observer_setup_before_P=True, observer_setup_ns=done["registration_end_ns"]-done["started_ns"], observer_go_overhead_ns=done["enabled_ns"]-done["go_received_ns"],
        step_wall_decomposition=False))


if __name__ == "__main__":
    assert sys.argv[1] == "workflow"
    workflow()
