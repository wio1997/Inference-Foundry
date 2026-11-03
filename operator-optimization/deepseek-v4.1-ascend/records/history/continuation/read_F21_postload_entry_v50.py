"""Bounded EPLB adaptor entry read; local dependency bytes are not a new remote attestation."""
from datetime import datetime, timezone
from pathlib import Path
import ast
import hashlib

from finalize_origin_coverage_v9 import N, O, read, write

ROOT = Path('/Users/wio/work/910c-deepinfer-control/installed-vllm-source')


def main():
    assert not (O / 'REVIEW-CHECKPOINT-v50.json').exists()
    old = read(O / 'EP-mapping-and-weight-contract-v19.json')
    caller = ROOT / 'vllm_ascend/ops/fused_moe/routed_experts.py'
    caller_remote = '/vllm-workspace/vllm-ascend/vllm_ascend/ops/fused_moe/routed_experts.py'
    assert hashlib.sha256(caller.read_bytes()).hexdigest() == old['source_files'][caller_remote]['sha256']
    path = ROOT / 'vllm_ascend/eplb/adaptor/vllm_adaptor.py'
    expected = 'e372c2e515cee4711f919e65e14bc2a81666b10c3074aca181ba519e2d186b99'
    names = ['register_layer', '__init__', 'init_buffer_tensor', 'init_expert_param_per_layer',
             'do_update_expert_map', 'do_update_expert_weight', 'do_update_log2phy_map', 'get_global_expert_map']
    gate = dict(status='PROFILE_CANNOT_PROVE_POSTLOAD_MAP_WEIGHT_PAIRING;POINT_SOURCE_TRACE',
        previous_turn_classification='PROGRESS:v49M17fallback_contracts',
        evidence=['EP-mapping-and-weight-contract-v19.json', 'F21-interface-branch-ledger-v19.json',
                  'F21-replica-source-granularity-contract-v19.json'],
        question='Where are checkpoint-loaded expert slots reconciled with Ascend custom/static execution maps,and which weights/scales/generations are transferred before publication?',
        allowed_source=[dict(path=str(path), sha256=expected, class_name='VllmEplbAdaptor', methods=names)],
        version_scope='Local installed-source dependency snapshot with a caller matching historical166SHA. Adaptor bytes have no historical166pin yet;any result about them is conditional on this dependency identity.',
        no_model_or_torch_import_or_device_service_health_process_work=True, no_remote_work=True,
        no_broad_repository_or_SDK_audit=True, whole_module_closed=False, overall_complete=False)
    write(O / 'PROFILE-CODE-QUESTION-GATE-v50.json', gate)
    raw = path.read_bytes();assert hashlib.sha256(raw).hexdigest() == expected
    lines = raw.decode().splitlines();tree = ast.parse(raw)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VllmEplbAdaptor')
    out = []
    for name in names:
        f = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
        out.append(dict(name='VllmEplbAdaptor.' + name, start=f.lineno, end=f.end_lineno,
            source=[str(i + 1) + ': ' + lines[i] for i in range(f.lineno - 1, f.end_lineno)],
            calls=sorted({ast.unparse(n.func) for n in ast.walk(f) if isinstance(n, ast.Call)})))
    write(O / 'F21-postload-adaptor-entry-source-v50.json', dict(
        generated_utc=datetime.now(timezone.utc).isoformat(), path=str(path), sha256=expected,
        caller_path=str(caller), caller_historical166sha=old['source_files'][caller_remote]['sha256'],
        dependency_historical166hash_attested=False, functions=out,
        scope=gate['version_scope'], no_target_implementation_changed=True,
        whole_module_closed=False, overall_complete=False))
    for f in out:print('\n'.join(f['source']))


if __name__ == '__main__':
    main()
