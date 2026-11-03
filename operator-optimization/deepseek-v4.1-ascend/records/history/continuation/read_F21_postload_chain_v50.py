"""Follow only updater/transfer callees exposed by the exact adaptor entry."""
from datetime import datetime, timezone
import ast
import hashlib

from finalize_origin_coverage_v9 import O, read, write
from read_F21_postload_entry_v50 import ROOT


def main():
    assert not (O / 'REVIEW-CHECKPOINT-v50.json').exists()
    specs = [
        ('vllm_ascend/eplb/adaptor/vllm_adaptor.py', 'e372c2e515cee4711f919e65e14bc2a81666b10c3074aca181ba519e2d186b99',
         None, ['EPLB_EXPERT_WEIGHT_NAMES']),
        ('vllm_ascend/eplb/eplb_updator.py', 'a734e8e12a69526c185cca305bc6bed7124bc30447da4a9f1e9e4aec48c53915',
         'EplbUpdator', ['__init__', 'set_adaptor', 'init_eplb', 'update_iteration',
                        'get_update_info_flag', 'wakeup_eplb_worker_flag', 'update_expert_weight_flag',
                        'forward_before', 'forward_end', 'warm_up_eplb']),
        ('vllm_ascend/eplb/core/eplb_device_transfer_loader.py', 'db1023bd25fbae05eddd3e6650e3d7680c4b522048997cda1d6f5549b3539130',
         'D2DExpertWeightLoader', ['__init__', 'generate_expert_d2d_transfer_task', 'asyn_expert_weight_transfer', 'update_expert_map_and_weight']),
        ('vllm_ascend/ops/fused_moe/routed_experts.py', 'f9db74836ae210a541432b923118696c7a15be9a10d5776834e1d3b64ce11d77',
         'AscendRoutedExperts', ['get_eplb_parameter']),
        ('vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py', 'a67601ee596d9aa4ae72ea3638f67ff7236cd8df884564bcd19f33653a1e0910',
         'AscendW8A8DynamicFusedMoEMethod', ['process_weights_after_loading']),
    ]
    verified = {'vllm_ascend/ops/fused_moe/routed_experts.py',
                'vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py'}
    old = read(O / 'EP-mapping-and-weight-contract-v19.json')
    for rel, sha, _, _ in specs:
        if rel in verified:
            assert old['source_files']['/vllm-workspace/vllm-ascend/' + rel]['sha256'] == sha
    gate = read(O / 'PROFILE-CODE-QUESTION-GATE-v50.json')
    additions = [dict(path=str(ROOT / rel), sha256=sha, class_name=cls, symbols=names,
                      historical166dependency_hash_attested=(rel in verified)) for rel, sha, cls, names in specs]
    additions.append(dict(path=str(ROOT / 'vllm_ascend/worker/model_runner_v1.py'),
        sha256='568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c',
        symbols=['eplb_warmup', 'load_model/if self.eplb_enable', '__init__/dynamic_eplb_and_eplb_enable_assignments'], historical166dependency_hash_attested=True))
    for item in additions:
        if item not in gate['allowed_source']:gate['allowed_source'].append(item)
    gate['dependency_extension'] = 'Exact symbol search finds updater.set_adaptor/init_eplb and D2D update callers;follow only their selected initialization/publication bodies and the expert-plane registry. Runner4357caller checked separately under historical hash.'
    write(O / 'PROFILE-CODE-QUESTION-GATE-v50.json', gate)
    out = []
    for rel, sha, cls, names in specs:
        p = ROOT / rel;raw = p.read_bytes();assert hashlib.sha256(raw).hexdigest() == sha
        lines = raw.decode().splitlines();tree = ast.parse(raw)
        container = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls) if cls else tree
        parts = []
        for name in names:
            if cls:
                f = next(n for n in container.body if isinstance(n, ast.FunctionDef) and n.name == name)
            else:
                f = next(n for n in container.body if isinstance(n, ast.Assign) and
                         any(isinstance(a, ast.Name) and a.id == name for a in n.targets))
            parts.append(dict(name=(cls + '.' if cls else '') + name, start=f.lineno, end=f.end_lineno,
                source=[str(i + 1) + ': ' + lines[i] for i in range(f.lineno - 1, f.end_lineno)]))
        out.append(dict(path=str(p), sha256=sha, historical166dependency_hash_attested=(rel in verified), parts=parts))
    runner = ROOT / 'vllm_ascend/worker/model_runner_v1.py'
    raw = runner.read_bytes();sha = hashlib.sha256(raw).hexdigest()
    expected = next(f['sha256'] for f in read(O / 'M17-sampling-source-and-liveness-v28.json')['source_files']
                    if f['path'].endswith('/worker/model_runner_v1.py'))
    assert sha == expected
    lines = raw.decode().splitlines();tree = ast.parse(raw)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'load_model')
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'VllmEplbAdaptor')
    enclosing = min((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.lineno <= call.lineno <= n.end_lineno),
                    key=lambda n: n.end_lineno - n.lineno)
    assert enclosing.name == 'eplb_warmup'
    selected = [dict(name='eplb_warmup/EPLB_entry', start=enclosing.lineno, end=enclosing.end_lineno,
        source=[str(i + 1) + ': ' + lines[i] for i in range(enclosing.lineno - 1, enclosing.end_lineno)])]
    static_entry = next(n for n in ast.walk(fn) if isinstance(n, ast.If) and ast.unparse(n.test) == 'self.eplb_enable')
    selected.append(dict(name='load_model/if self.eplb_enable', start=static_entry.lineno, end=static_entry.end_lineno,
        source=[str(i + 1) + ': ' + lines[i] for i in range(static_entry.lineno - 1, static_entry.end_lineno)]))
    runner_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NPUModelRunner')
    init = next(n for n in runner_cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    for n in ast.walk(init):
        if isinstance(n, ast.Assign) and any(ast.unparse(a) in ['self.dynamic_eplb', 'self.eplb_enable'] for a in n.targets):
            selected.append(dict(name='__init__/EPLB_eligibility', start=n.lineno, end=n.end_lineno,
                source=[str(i + 1) + ': ' + lines[i] for i in range(n.lineno - 1, n.end_lineno)]))
    order = [dict(line=n.lineno, function=ast.unparse(n.func)) for n in ast.walk(fn) if isinstance(n, ast.Call)
             and any(s in ast.unparse(n.func) for s in ['load_model', 'weight', 'eplb', 'get_model', 'VllmEplbAdaptor', 'set_adaptor'])]
    out.append(dict(path=str(runner), sha256=sha, historical166dependency_hash_attested=True,
                    parts=selected, load_model_selected_call_locations=order))
    write(O / 'F21-postload-updater-transfer-source-v50.json', dict(
        generated_utc=datetime.now(timezone.utc).isoformat(), files=out,
        no_model_or_torch_import=True, no_remote_or_device_work=True,
        scope='Only selected direct updater/transfer/registry bodies and exact runner entry. Most dependency bytes are local-reference-only until original hash can be checked.',
        whole_module_closed=False, overall_complete=False))
    for file in out:
        print('FILE', file['path'])
        for part in file['parts']:
            if part['name'] in ['EplbUpdator.update_iteration', 'EplbUpdator.get_update_info_flag',
                                'EplbUpdator.wakeup_eplb_worker_flag', 'EplbUpdator.update_expert_weight_flag',
                                'AscendRoutedExperts.get_eplb_parameter', 'AscendW8A8DynamicFusedMoEMethod.process_weights_after_loading',
                                '__init__/EPLB_eligibility']:
                print('\n'.join(part['source']))
        if 'load_model_selected_call_locations' in file:print('LOAD_CALL_LOCATIONS', file['load_model_selected_call_locations'])


if __name__ == '__main__':
    main()
