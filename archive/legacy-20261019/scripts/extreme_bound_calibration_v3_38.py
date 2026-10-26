#!/usr/bin/env python3
"""V3.38 fixed-algorithm Bound calibration from Run611 salvaged profile scope."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
BASE = ROOT / 'evidence/20260928_loop081_bound/run608/bound_calibration_v3_37.json'
R = ROOT / 'evidence/20260928_loop081_bound/run611'
RUN610 = ROOT / 'evidence/20260928_loop081_bound/run610/live'
FILES = {
    'base': BASE,
    'recovery': R / 'raw_recovery_admission.json',
    'parse': R / 'offline_parse/manifest.json',
    'scope': R / 'scope_ledger.json',
    'astra': R / 'astra_recovery_review.md',
    'clock': R / 'astra_host_clock_witness.json',
    'cleanup': RUN610 / 'cleanup_status.txt',
}
EXPECTED = {
    'base': '62eb0954909151f532a424cf21805c1dc6df8d1fa76d1da1a336a99a8648a659',
    'recovery': '9903001902801471e005601bf331028a4d02a8b04cb8689f11a5d81c645bd802',
    'parse': 'b51eac01dcdc9f183544d51d20040f371f5ffb23a686ded7372b2cb366c84e83',
    'scope': '3c9bb23a28674fae048daad9c32a2803324d5509aa418c41d66c129fe27fd2ff',
    'astra': 'd9ef7aae0db0cd5eaa4585755ad5b20472dfbf32efe09f141807a26d4455356b',
    'clock': 'fabf5ca8968316529864dc28cff7eef9e7c39851db3c675bd06083772a14b42d',
    'cleanup': 'c116121d1db8ed3ad730ad0222105e8071ae0a203f08b97fcc5b81f9a7ecd006',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok, why):
    if not ok:
        raise ValueError(why)


for key, path in FILES.items():
    need(sha(path) == EXPECTED[key], f'input SHA drift: {key}')
base = json.loads(BASE.read_text())
recovery = json.loads(FILES['recovery'].read_text())
parse = json.loads(FILES['parse'].read_text())
scope = json.loads(FILES['scope'].read_text())
need(base['bound_semantics_v3_37']['current_formal_tps'] == 571.681 and
     recovery['status'] == 'all8_cohort5_raw_profile_salvaged_after_invalid_validator_clock_gate' and
     parse['status'] == 'all8_salvaged_raw_copy_offline_parse_pass' and
     scope['status'] == 'all8_instrumented_host_scope_ledger_pass' and
     len(scope['rows']) == 24, 'identity/status')
need(parse['raw_recovery_admission_sha256'] == EXPECTED['recovery'] and
     scope['manifest_sha256'] == EXPECTED['parse'], 'evidence SHA chain')
cleanup = dict(line.split('=', 1) for line in FILES['cleanup'].read_text().splitlines())
need(cleanup['run_exit'] == cleanup['final_exit'] == '1' and
     all(v == '0' for k, v in cleanup.items() if k not in ('run_exit', 'final_exit')),
     'Run610 INVALID/cleanup')
need(len(recovery['raw_sha256']) == 604 and len(parse['parsed']) == 8 and
     len({r['rank'] for r in scope['rows']}) == 8, 'all8 raw/parsed/scope')
target = [r['stages']['target']['duration_us'] for r in scope['rows']]
proposer = [r['stages']['proposer']['duration_us'] for r in scope['rows']]
base['model_revision'] = 'v3.38_run611_instrumented_scope_only'
base['bound_semantics_v3_38'] = {
    'active_objective': 'minimum Product E2E Runtime for unchanged DSpark7 algorithm, acceptance trajectory, output semantics and model work',
    'current_formal_tps': 571.681,
    'run610': 'INVALID controller; client and three upstream admissions passed, but original validator compared RAW and MONOTONIC clocks. Diagnostic 555.1125 tok/s observer-perturbed, never formal.',
    'run611': {
        'scope': 'salvaged all8 cohort5 Level0/no-counter raw, offline parsed; 604 files; 24 profiler CPU stage-scope rows',
        'raw_integrity': 'independent Astra scoped PASS; 282 producer .done lengths independently verified; script itself checks 210 relevant markers',
        'clock': 'installed profiler start_info CLOCK_MONOTONIC_RAW; end_info Python CLOCK_MONOTONIC. Wall mapping to Host is conditional on 1ms margin, not a strict calibration bound.',
        'host_target_scope_us_range': [min(target), max(target)],
        'host_proposer_scope_us_range': [min(proposer), max(proposer)],
        'admission': 'instrumented Current Host intervals only; no native device completion or unperturbed stage cost'
    },
    'strict_resource_hardware_floor_s': None,
    'strict_scheduling_execution_floor_s': None,
    'strict_product_e2e_tps_ceiling': None,
    'numeric_current_to_credible_limit_gap': None,
    'missing_for_resource': ['fixed-W0 compulsory fresh work and traffic including prefill/seed/KV/state/HCCL',
                             'exact-board attainable cumulative compute/HBM/HCCL C-plus/B and freshness/loaded image'],
    'missing_for_scheduling': ['typed actual per-layer Draft KV writer/read generations across cycle64/65',
                               'all8 native producer/consumer and Host→device readiness/completion',
                               'legal alternative schedule under shared-resource contention and observer OFF/ON transfer'],
    'next_measurement': 'offline native task correlation from Run611 copies with same-W0 Basis/Product/dispatch; preserve cycle63 tail and cycle66 stop-fence ambiguity, then targeted typed KV and HCCL dependencies. Resource certificate work continues in parallel.',
    'independent_review': 'Run611 Astra recovery review scoped PASS for raw and conditional Host containment; not a Bound review'
}
need(all(base['bound_semantics_v3_38'][key] is None for key in
         ('strict_resource_hardware_floor_s', 'strict_scheduling_execution_floor_s',
          'strict_product_e2e_tps_ceiling', 'numeric_current_to_credible_limit_gap')),
     'fail-closed strict endpoints')
out = R / 'bound_calibration_v3_38.json'
out.write_text(json.dumps(base, indent=2) + '\n')
print(json.dumps({'status': 'v3_38_instrumented_scope_only',
                  'strict_endpoints': 'null', 'rows': 24}))
