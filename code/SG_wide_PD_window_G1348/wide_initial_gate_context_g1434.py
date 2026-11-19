"""Bind a wide first-warmup gate to an exact staged plan and live receipts."""

from datetime import datetime, timedelta
from hashlib import sha256
import json

from plan_SG_wide_campaign_leases_g1424 import windows


PLAN_BY_WIDTH = {
    65536: (1347, '57029ac9c98017263b8d1955c61e97ebd5bea2835fbee4c78263b0bbec5ca14a'),
    131072: (14331, '02da14c6679538d8d90f4f712a1e02c0e6e5bfc65b63597635999fcd626d56a6'),
    262144: (14332, 'f3bff00f136f9383625d9d54313a5bf947766de0d10a34dd1e74776fb8ff1f79'),
}


def load_plan(path, width):
    epoch, expected = PLAN_BY_WIDTH[width]
    raw = path.read_bytes()
    assert sha256(raw).hexdigest() == expected
    plan = json.loads(raw)
    assert plan['epoch'] == epoch and plan['derived_from_actual_epoch'] == 1317
    assert plan['workload']['width_tokens'] == width
    return plan, expected


def binding(now, plan, plan_sha, run_id, p, d, pd, cpu, *, required_seconds=None):
    epoch = plan['epoch']
    workload = plan['workload']
    width = workload['width_tokens']
    assert PLAN_BY_WIDTH[width] == (epoch, plan_sha)
    lease = windows(workload)
    assert run_id == f'sg_pd_g{epoch}_aisbench_{width // 1024}k'
    router = pd['handle']['router']
    p_guard = p['actual_workload_guard']
    assert p['actual_router'] == router and p['guard_FD_verified']
    assert p_guard == pd['fresh_live_guard'] and p_guard['phase'] == 'holding'
    assert p_guard['epoch'] == epoch and p_guard['input_tokens'] == width
    assert p_guard['preparation_manifest_sha256'] == plan_sha
    assert p_guard['campaign_id'] == run_id and p_guard['pair'] == 'P2203/D2204'
    assert p_guard['bound_s'] == lease['pair_guard_lease_seconds']
    assert p['guard_PID'] == d['handle']['guard_PID'] == router['guard_PID'] == p_guard['pid']
    expiry = datetime.fromisoformat(p_guard['at']) + timedelta(seconds=p_guard['bound_s'])
    assert expiry == datetime.fromisoformat(p_guard['deadline_UTC'])
    timeout = workload['requested_router_request_timeout_seconds']
    wait = workload['requested_D_wait_seconds']
    assert router['configured_request_deadline_seconds'] == timeout
    assert d['KV_waiting_timeout_actual_by_rank'] == {str(rank): wait for rank in range(8)}

    deadlines = {'pair_workload_lock': expiry, 'CPU_client': datetime.fromisoformat(cpu['conservative_deadline_UTC'])}
    assert cpu['lease'] == 'wide_cpu_only' and cpu['container_ID'] and cpu['CPU_controller_PID'] > 1
    assert -60 <= (now-datetime.fromisoformat(cpu['reported_at_UTC'])).total_seconds() < 600
    for role, tag, proof, cid in (('prefill', 'P', p, p['container']['Id']),
                                  ('decode', 'D', d, d['container_ID'])):
        handle = proof['handle']
        state = proof['state']
        assert proof['ready'] == 200 and proof['all8_ownership_and_FDs_verified']
        assert handle['epoch'] == state['generation'] == epoch
        assert router[f'{tag}_controller_PID'] == handle['pid'] == state['pid']
        assert router[f'{tag}_container_ID'] == cid == state['container_id']
        assert state['lifetime_seconds'] == lease['minimum_model_role_lifetime_seconds']
        assert proof['idle'] and all(values and all(v == 0 for v in values) for values in proof['idle'].values())
        deadlines[role] = datetime.fromisoformat(handle['at']) + timedelta(seconds=state['lifetime_seconds'])

    remaining = {key: (stamp-now).total_seconds() for key, stamp in deadlines.items()}
    if required_seconds is None:
        required_seconds = lease['minimum_same_client_lease_from_first_warmup_seconds']
    assert min(remaining.values()) > required_seconds, remaining
    return {
        'phase_seconds': workload['warmup_phase_seconds'],
        'campaign_seconds': workload['warmup_campaign_seconds'],
        'decode_waiting_seconds_by_rank': d['KV_waiting_timeout_actual_by_rank'],
        'router_request_timeout_seconds': timeout,
        'resource_deadlines_UTC': {role: expiry.isoformat() for role, expiry in deadlines.items()},
        'CPU_client_container_id': cpu['container_ID'],
        'wide_preparation_sha256': plan_sha,
    }
