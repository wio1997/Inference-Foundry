"""Once-only diagnostic capture after the preserved startup observer timeout.

No model start, inference, SDK init, retry, cleanup or fallback. This independent
capture preserves all audit rows so the timeout can be diagnosed; capture
completion alone never qualifies startup or authorizes model requests.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT/'scripts/glm53_C32_L17_native_capture_v1.py'
CORE_SHA = '229893e45c8f0778699889446cd5f7d9097f7c8d1b6873612cc90eb679c793f2'
REVIEW = ROOT/'results/C32-L17-native-startup-capture-review-v1.json'
FAILED = ROOT/'runs/20261019T-L17-D-models-and-observe-v1/receipt.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan():
    assert sha(CORE) == CORE_SHA
    spec = importlib.util.spec_from_file_location('l17_timeout_capture', CORE)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    return core, core.plan_hashes(core.contract.STARTUP_CAPTURE, 'startup')


def main():
    core, hashes = plan()
    review = json.loads(REVIEW.read_text())
    assert review['launcher_sha256'] == sha(Path(__file__))
    assert review['core_sha256'] == CORE_SHA
    assert review['failed_observer_receipt_sha256'] == sha(FAILED)
    assert review['decision'] == 'APPROVED_DIAGNOSTIC_CAPTURE_AFTER_PRESERVED_OBSERVER_TIMEOUT'
    assert review['host_codes_sha256'] == hashes
    assert review['model_requests'] == 0 and review['SDK_init_calls'] == 0
    assert review['startup_qualified'] is False and review['runtime_model_dispatch_authorized'] is False
    failed = json.loads(FAILED.read_text())
    assert failed['pass'] is False and failed['phase'] == 'FAILED_OR_PARTIAL_STATE_PRESERVED'
    assert failed['starts'] == {'start-36':0, 'start-41':0}
    assert failed['observe'] == {'observe-36':1, 'observe-41':1}
    assert failed['external_model_requests'] == 0 and failed['cleanup_or_rollback_executed'] is False
    assert failed['capacity_sha256'] == core.contract.D_CAPACITY_SHA
    assert failed['runtime_CIDs'] == {n:r['container_id'] for n,r in core.contract.combined_providers().items()}
    for node in (36,41):
        wire_path = FAILED.parent/('observe-'+str(node)+'.json')
        assert review['observer_wire_sha256'][str(node)] == sha(wire_path)
        wire = json.loads(wire_path.read_text())
        assert type(wire['rc']) is int and wire['rc'] == 1
        assert 'AssertionError: D startup observation bounded 1800s exceeded' in wire['stderr']
        progress = [json.loads(line.split('=',1)[1]) for line in wire['stdout'].splitlines() if line.startswith('GLM53_L17_D_OBSERVE_PROGRESS=')]
        last = progress[-1]
        assert last['node'] == node and last['health'] == 200 and last['idle'] is True
        assert last['registered'] == last['graph_ranks'] == last['native_ops_workers'] == 16
        assert last['eager_status'] == 'COMPLETE_ALL16' and last['eager_bs2_draft_decode_pids'] == 16
        assert last['native_audit_errors'] > 0 and last['done'] is False
    result = core.execute(core.contract.STARTUP_CAPTURE, core.contract.STARTUP_CAPTURE_REVIEW, 'startup')
    result.update(startup_qualified=False, model_dispatch_authorized=False,
                  reason='Independent diagnostic capture only; original timeout and audit errors require separate acceptance')
    print(json.dumps(result))
    return 0 if result.get('pass') is True else 1


if __name__ == '__main__':
    raise SystemExit(main())
