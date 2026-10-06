"""One direct PD probe of already ready Run246 roles; never restarts models."""
import json
import subprocess
import sys
import time
from pathlib import Path
import pd_functional_run as m

ROOT = Path(__file__).resolve().parent


def guard(host, plan):
    root = plan['reuse_roles'][host]
    assert m.live(root), 'resident role identity changed'
    owners, raw = m.device_owners()
    members = m.tree(root['pid'], m.catalogue())
    assert len(owners) == 16 and owners.issubset({p['pid'] for p in members})
    if host == '166':
        owner = json.loads(Path('/data/tiankuan/wio/glm52-pd/controller-owner.json').read_text())
        assert owner['status'] == 'running' and owner['run_id'] == plan['run_id']
        assert m.live(owner['owner'])
    spec = json.loads((ROOT / 'controller_spec.json').read_text())
    for source in spec['stages'][0]['sources']:
        assert m.hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256']
    port = plan['ports'][host]
    status, _ = m.fetch('http://127.0.0.1:%d/health' % port)
    status2, models = m.fetch('http://127.0.0.1:%d/v1/models' % port)
    assert status == status2 == 200
    assert any(x['id'] == 'glm-53' for x in json.loads(models)['data'])
    m.assert_idle(port)
    m.write('guard_%s.json' % host, dict(root=root, members=members, npu_owners=sorted(owners),
                                      npu_raw=raw, idle=True, served_model='glm-53'))
    return dict(host=host, live_identity=True, npu_owners_count=len(owners), idle=True)


def call(host, plan):
    argv = ['/usr/bin/python3', str(Path(__file__).resolve()), 'guard', host]
    if host == '167':
        argv = ['ssh', '-o', 'BatchMode=yes', 'root@172.16.10.167'] + argv
    p = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr[-5000:]
    return json.loads(p.stdout)


def workflow(plan):
    started = time.monotonic_ns()
    try:
        checks = {h: call(h, plan) for h in ['166', '167']}
        m.write('before_probe.json', checks)
        result = m.probe(plan)
        m.write('final_status.json', dict(status='completed', result=result,
            borrowed_roles='Run246 P166/D167 retained', model_restarts=0,
            public8000='no replacement;full API pending', performance_gain=None,
            wall_s=(time.monotonic_ns()-started)/1e9))
        print(json.dumps(result), flush=True)
    except BaseException as error:
        m.write('final_status.json', dict(status='failed', error=type(error).__name__+': '+str(error),
            borrowed_roles='Run246 roles retained;read health/native logs before next action',
            model_restarts=0, performance_gain=None, Current=None))
        raise


if __name__ == '__main__':
    plan = json.loads((ROOT / 'functional_plan.json').read_text())
    if sys.argv[1] == 'workflow':
        workflow(plan)
    else:
        print(json.dumps(guard(sys.argv[2], plan)))
