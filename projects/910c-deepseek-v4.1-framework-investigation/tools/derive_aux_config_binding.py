#!/usr/bin/env python3
"""Execute only the two pinned aux selector methods, without vLLM/Torch imports."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

OUT=Path('audit/execution-search')
def read(p):return json.loads(Path(p).read_bytes())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def method(source,name):
    return copy.deepcopy(next(n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n,ast.FunctionDef) and n.name==name))
def main():
    pins=read(OUT/'staging-execution-identity.json')['sources']
    by={r['runtime_path']:r for r in pins}
    rp=by['/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py']
    bp=by['/vllm-workspace/vllm/vllm/v1/worker/gpu_model_runner.py']
    for p in (rp,bp):assert digest(p['local_path'])==p['runtime_sha256']
    name='_get_eagle3_aux_layers_from_config'
    b=ast.ClassDef(name='Base',bases=[],keywords=[],body=[method(Path(bp['local_path']),name)],decorator_list=[])
    n=ast.ClassDef(name='NPU',bases=[ast.Name(id='Base',ctx=ast.Load())],keywords=[],body=[method(Path(rp['local_path']),name)],decorator_list=[])
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    ns={};exec(compile(ast.fix_missing_locations(ast.Module(body=[future,b,n],type_ignores=[])),'pinned aux selector methods','exec'),ns)
    old=read(OUT/'dspark-execution-identity.json');config_path=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/evidence/model-config.json')
    assert digest(config_path)==old['config_pin']['sha256']
    raw=read(config_path);text=raw['text_config']
    for level in (raw,text):
        assert not any(level.get(k) for k in ('eagle_aux_hidden_state_layer_ids','dflash_config','eagle_config'))
    hf=SimpleNamespace(**{k:v for k,v in raw.items() if k!='text_config'},text_config=SimpleNamespace(**text))
    spec=SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=hf),use_dspark=lambda:True)
    base=ns['Base']();base.speculative_config=spec
    npu=ns['NPU']();npu.speculative_config=spec
    assert base._get_eagle3_aux_layers_from_config() is None
    selected=npu._get_eagle3_aux_layers_from_config();entries=[i for i in range(text['num_hidden_layers']) if i+1 in selected]
    assert list(selected)==[38,39,40] and entries==[37,38,39]
    setter=read(OUT/'aux-mixin-source-pin.json')
    assert 'self.aux_hidden_state_layers = tuple(sorted(set(layers)))' in setter['functions'][0]['source']
    d=dict(status='SOURCE_DERIVED_FULL_AUX_CONFIG_CALL_CHAIN',checkpoint_ids=text['dspark_target_layer_ids'],
        upstream_selector_result=None,NPU_selector_result=list(selected),model_entry_layers_zero_based=entries,
        after_last_aux_body_layers=[39],
        source_methods=[dict(path=p['runtime_path'],sha256=p['runtime_sha256'],method=name) for p in (bp,rp)],
        setter=setter,config_sha256=digest(config_path),reader_sha256=digest(__file__),
        path='GPU setup virtual _get -> NPU fallback adds1 -> root set_aux -> EagleModelMixin stores sorted ids unchanged -> model tests layer_idx+1',
        qualification='Pure selector execution from frozen config fields and pinned source, not a live object dump. Draft-normalization source preserves DSpark ids and no upstream eagle/dflash override is present in frozen config. Current Docker logs contain no matching info message; missing logs are not evidence of another selector result.',
        correction='Earlier model-only derivation omitted NPU runner +1 and incorrectly claimed entries36/37/38. Retract unconditional body38 pruning: its full output mean is aux at entry39. The old raw topology flag37/38/39 was accidentally correct but is now derived rather than hardcoded.',
        new_NPU_work=0,service_requests=0,remote_writes=[])
    (OUT/'aux-config-call-chain.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:d[k] for k in ('status','checkpoint_ids','NPU_selector_result','model_entry_layers_zero_based','after_last_aux_body_layers')}))
if __name__=='__main__':main()
