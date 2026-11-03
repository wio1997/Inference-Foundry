"""Read the checkpoint-name -> expert-ID -> local-slot path under historical source hashes."""
from datetime import datetime, timezone
from pathlib import Path
import ast
import hashlib

from finalize_origin_coverage_v9 import N, O, W, read, write


def main():
    assert not (O / 'REVIEW-CHECKPOINT-v51.json').exists()
    old = read(O / 'EP-mapping-and-weight-contract-v19.json')
    core = W / 'frozen-source-snapshot/vllm/vllm/model_executor/layers/fused_moe'
    specs = [
        (core / 'routed_experts.py', '/vllm-workspace/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py',
         'RoutedExperts', ['expert_map', '_map_global_expert_id_to_local_expert_id',
                           'load_weights', 'get_expert_mapping', 'make_expert_params_mapping', 'build_expert_params_mapping']),
        (core / 'expert_map_manager.py', '/vllm-workspace/vllm/vllm/model_executor/layers/fused_moe/expert_map_manager.py',
         'ExpertMapManager', ['map_global_to_local']),
        (Path('/Users/wio/work/910c-deepinfer-control/installed-vllm-source/vllm_ascend/models/deepseek_v41/model.py'),
         '/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py',
         'AscendDeepseekV41LLMForCausalLM', ['_is_milestone_weight', 'load_weights',
                                          'set_moe_parameters', 'get_expert_mapping', '_load_model_weights']),
    ]
    gate = dict(status='PROFILE_CANNOT_PROVE_CHECKPOINT_TO_EXECUTION_SLOT_COMMUTATION;POINT_SOURCE_READ',
        previous_turn_classification='PROGRESS:v50static_dynamic_publication_boundary',
        profile_evidence=['F21-placement-and-hotspot-variants-v17.json', 'EP-mapping-and-weight-contract-v19.json',
                          'F21-static-dynamic-loading-contract-v50.json'],
        question='Which IDs and maps does the specified model use to route checkpoint expert weights into local physical slots,and is Ascend staticplacement explicitly connected to that path?',
        allowed_source=[dict(path=str(p), sha256=old['source_files'][remote]['sha256'],
                             class_name=cls, methods=names, historical166_hash_matched=True)
                        for p, remote, cls, names in specs],
        reused_edge='Base weight_loader575-608 andAscendinit_eplb493-578alreadycachedv19;reuse their exact hash-pinned selection rather thanrereadingallquantloaderbranches.',
        no_model_torch_import_or_device_service_health_process_work=True, no_remote_work=True,
        no_broad_repository_or_SDK_audit=True, whole_module_closed=False, overall_complete=False)
    write(O / 'PROFILE-CODE-QUESTION-GATE-v51.json', gate)
    files = []
    for path, remote, cls_name, names in specs:
        raw = path.read_bytes();sha = hashlib.sha256(raw).hexdigest()
        assert sha == old['source_files'][remote]['sha256']
        frozen = N / 'F21-checkpoint-source-v51' / ('vllm' if '/vllm/vllm/' in remote else 'vllm-ascend') / Path(remote).relative_to('/vllm-workspace/' + ('vllm' if '/vllm/vllm/' in remote else 'vllm-ascend'))
        frozen.parent.mkdir(parents=True, exist_ok=True)
        if frozen.exists():assert hashlib.sha256(frozen.read_bytes()).hexdigest() == sha
        else:frozen.write_bytes(raw)
        tree = ast.parse(raw);lines = raw.decode().splitlines()
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
        parts = []
        for name in names:
            f = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
            parts.append(dict(name=cls_name + '.' + name, start=f.lineno, end=f.end_lineno,
                source=[str(i + 1) + ': ' + lines[i] for i in range(f.lineno - 1, f.end_lineno)],
                direct_calls=sorted({ast.unparse(n.func) for n in ast.walk(f) if isinstance(n, ast.Call)})))
        if cls_name == 'AscendDeepseekV41LLMForCausalLM':
            for node in cls.body:
                targets = getattr(node, 'targets', [])
                if isinstance(node, ast.AnnAssign):targets = [node.target]
                if any(isinstance(a, ast.Name) and a.id in ['_DEFERRED_WEIGHT_PREFIXES', '_DEFERRED_WEIGHT_MARKERS'] for a in targets):
                    parts.append(dict(name=cls_name + '/deferred_filter_constant', start=node.lineno, end=node.end_lineno,
                        source=[str(i + 1) + ': ' + lines[i] for i in range(node.lineno - 1, node.end_lineno)],
                        direct_calls=[]))
        files.append(dict(path=str(path), frozen_path=str(frozen), remote_path=remote, sha256=sha,
                          historical166_hash_matched=True, parts=parts))
    write(O / 'F21-checkpoint-routing-source-v51.json', dict(
        generated_utc=datetime.now(timezone.utc).isoformat(), files=files,
        scope='Only the named checkpoint mapping/load entry methods. This doesnotattest actual maps/weights,unseen hooks/factory overrides,nativeprecisionorallloaderfamilies.',
        no_target_implementation_changed=True, no_model_import_remote_or_device_work=True,
        whole_module_closed=False, overall_complete=False))
    for f in files:
        if f['remote_path'].endswith('/model.py'):
            for p in f['parts']:
                if p['name'].endswith(('._is_milestone_weight', '.set_moe_parameters')):print('\n'.join(p['source']))


if __name__ == '__main__':
    main()
