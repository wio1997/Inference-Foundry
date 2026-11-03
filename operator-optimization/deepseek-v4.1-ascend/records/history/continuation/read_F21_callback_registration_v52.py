"""Read only the concrete callback-registration route,with explicit source identity limits."""
from datetime import datetime, timezone
from pathlib import Path
import ast
import hashlib

from finalize_origin_coverage_v9 import N, O, W, read, write


def main():
    assert not (O / 'REVIEW-CHECKPOINT-v52.json').exists()
    local = Path('/Users/wio/work/910c-deepinfer-control/installed-vllm-source/vllm_ascend')
    specs = [
        (W / 'frozen-source-snapshot/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py',
         '68c48e5df0ea4a7cc1f04436b7b9153eb53802ecd9bb55d38db8738508daac6d', 'CORE_INIT_CALLBACK', True),
        (N / 'F21-source-snapshot-v50/vllm_ascend/ops/fused_moe/routed_experts.py',
         'f9db74836ae210a541432b923118696c7a15be9a10d5776834e1d3b64ce11d77', 'ASCEND_INHERITANCE', True),
        (N / 'F21-source-snapshot-v50/vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py',
         'a67601ee596d9aa4ae72ea3638f67ff7236cd8df884564bcd19f33653a1e0910', 'W8A8_SCHEME_LAYOUT', True),
        (local / 'quantization/method_adapters.py',
         '341d93cae5774414a810444a50eb752cd425330156f404d1b9a10381e6bc0138', 'MOE_PARAM_ADAPTER', False),
        (local / 'quantization/configs/modelslim_config.py',
         '92c924abc0af1ae37d7614ee01cbae0b3be1da86ef6abc885b2208560f3dcbb1', 'MODELSLIM_MOE_SELECTION', False),
    ]
    gate = dict(status='PROFILE_LOCATES_QUANT_PATH_BUT_NOT_PARAMETER_CALLBACK;POINT_REGISTRATION_READ',
        previous_turn_classification='PROGRESS:v51checkpoint_entry_namespace',
        profile_evidence=['F21-checkpoint-map-commutation-contract-v51.json', 'EP-mapping-and-weight-contract-v19.json'],
        supporting_receipt=str(N / 'generic-callers166-v18.json'),
        supporting_receipt_limit='Savedgenericchainsincludemethod_adapters.py224linearapply;thislocatesthepackagebutdoesnotattestMoEconstructorcallbackorModelSlimselection.',
        question='For a concrete ModelSlim/W8A8 MoE route,where is the loader callback supplied,which parameters receive it,and does Ascend override the inherited mapper?',
        allowed_source=[dict(path=str(p), sha256=sha, role=role, historical166_hash_matched=pin)
                        for p, sha, role, pin in specs],
        scope='Core/Ascend/scheme files matchhistorical166hashes;adapterandconfigarelocalreferenceonly. Do not infer actualglobalfactory/callbackobject identity.',
        no_model_torch_import_or_device_service_health_process_work=True, no_remote_work=True,
        no_broad_repository_or_SDK_audit=True, whole_module_closed=False, overall_complete=False)
    write(O / 'PROFILE-CODE-QUESTION-GATE-v52.json', gate)
    files = []
    for ordinal, (p, sha, role, historical) in enumerate(specs):
        raw = p.read_bytes();assert hashlib.sha256(raw).hexdigest() == sha
        tree = ast.parse(raw);lines = raw.decode().splitlines();selected = [];metadata = {}
        if role == 'CORE_INIT_CALLBACK':
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RoutedExperts')
            init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
            for n in ast.walk(init):
                if isinstance(n, ast.stmt) and not isinstance(n, (ast.If, ast.For, ast.With, ast.FunctionDef)):
                    text = ast.unparse(n)
                    if ((('weight_loader' in text or '_get_quant_method' in text) and isinstance(n, (ast.Assign, ast.AnnAssign))) or
                        (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and
                         ast.unparse(n.value.func) == 'self.quant_method.create_weights')):
                        selected.append(n)
            assert selected
        elif role == 'ASCEND_INHERITANCE':
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AscendRoutedExperts')
            metadata = dict(class_name=cls.name, bases=[ast.unparse(n) for n in cls.bases],
                directly_defined_methods=[n.name for n in cls.body if isinstance(n, ast.FunctionDef)],
                own_weight_loader_defined=any(isinstance(n, ast.FunctionDef) and n.name == 'weight_loader' for n in cls.body),
                own_global_to_local_override=any(isinstance(n, ast.FunctionDef) and n.name == '_map_global_expert_id_to_local_expert_id' for n in cls.body))
            selected = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_get_quant_method']
        elif role == 'W8A8_SCHEME_LAYOUT':
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AscendW8A8DynamicFusedMoEMethod')
            selected = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ['get_weight', 'get_dynamic_quant_param']]
            assert len(selected) == 2
        elif role == 'MOE_PARAM_ADAPTER':
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AscendFusedMoEMethod')
            selected = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'create_weights']
            assert len(selected) == 1
        else:
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AscendModelSlimConfig')
            fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_quant_method')
            candidates = [n for n in ast.walk(fn) if isinstance(n, ast.If) and ast.unparse(n.test) == 'is_fused_moe_layer(layer)']
            assert len(candidates) == 1
            selected = candidates
            mapped_loader = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_make_modelslim_moe_weight_loader')
            selected.append(mapped_loader)
            for n in tree.body:
                targets = getattr(n, 'targets', [])
                if isinstance(n, ast.AnnAssign):targets = [n.target]
                if any(isinstance(a, ast.Name) and a.id == '_MOE_WEIGHT_LOADER_NAME_MAP' for a in targets):selected.append(n)
            assert len(selected) == 3
            gate.setdefault('direct_dependency_extension', [])
            extension = dict(path=str(p), sha256=sha, function='_make_modelslim_moe_weight_loader',
                reason='The selected MoE factory installs this wrapper before param registration;inspect exact forwarding semantics and its literal name map.')
            if extension not in gate['direct_dependency_extension']:gate['direct_dependency_extension'].append(extension)
            write(O / 'PROFILE-CODE-QUESTION-GATE-v52.json', gate)
            metadata['ancestor_arms'] = [dict(line=n.lineno, condition=ast.unparse(n.test),
                arm='BODY' if any(x.lineno <= candidates[0].lineno <= x.end_lineno for x in n.body) else 'ORELSE')
                for n in ast.walk(fn) if isinstance(n, ast.If) and n is not candidates[0]
                and n.lineno <= candidates[0].lineno <= n.end_lineno]
        parts = [dict(start=n.lineno, end=max(x.end_lineno for x in n.body) if isinstance(n, ast.If) else n.end_lineno,
                      source=[str(i + 1) + ': ' + lines[i] for i in range(n.lineno - 1, max(x.end_lineno for x in n.body) if isinstance(n, ast.If) else n.end_lineno)])
                 for n in selected]
        frozen = N / 'F21-callback-source-v52' / ('%02d-' % ordinal + p.name)
        frozen.parent.mkdir(parents=True, exist_ok=True)
        if frozen.exists():assert hashlib.sha256(frozen.read_bytes()).hexdigest() == sha
        else:frozen.write_bytes(raw)
        files.append(dict(path=str(p), frozen_path=str(frozen), sha256=sha,
                          role=role, historical166_hash_matched=historical, metadata=metadata, parts=parts))
    write(O / 'F21-callback-registration-source-v52.json', dict(
        generated_utc=datetime.now(timezone.utc).isoformat(), files=files,
        scope=gate['scope'], no_model_import_remote_or_device_work=True,
        implementation_changed=False, whole_module_closed=False, overall_complete=False))
    for f in files:
        if f['role'] != 'MODELSLIM_MOE_SELECTION':continue
        print('FILE_ROLE',f['role'])
        if f['metadata']:print('CLASS_METADATA',f['metadata'])
        for p in f['parts']:print('\n'.join(p['source']))


if __name__ == '__main__':
    main()
