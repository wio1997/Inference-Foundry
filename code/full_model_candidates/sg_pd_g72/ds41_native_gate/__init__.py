"""Native model registry with auditable load coverage and INT8 Engram storage."""
from pathlib import Path
import functools
import hashlib
import json
import os
import sys

import adapter

ROOT=Path(__file__).resolve().parents[1]
MODEL='/models/ds41'
adapter.install(MODEL)
from pd_gate_bridge import install as install_pd_gate
install_pd_gate(ROOT)
from draft_quant_bridge import install as install_draft_quant
install_draft_quant(ROOT)

from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
from sglang.srt.disaggregation.ascend.transfer_engine import AscendTransferEngine,TransferEngine,import_error

if import_error is not None:raise import_error
if not hasattr(TransferEngine.TransDataOpType,'DEVICE_RDMA'):
    raise RuntimeError('Fixed SDK has no DEVICE_RDMA entry point; do not fall back to SDMA')
EXPECTED=json.loads((ROOT/'native_source_sha256.json').read_text())
for cls in [DeepseekV4ForCausalLM,DeepseekV4ForCausalLMDSpark]:
    p=Path(sys.modules[cls.__module__].__file__)
    if hashlib.sha256(p.read_bytes()).hexdigest()!=EXPECTED[p.name]:
        raise RuntimeError('Native model source changed: '+str(p))

TARGET_PARAMS={}


def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def instrument(cls,stage):
    original=cls.load_weights
    @functools.wraps(original)
    def wrapped(self,*args,**kwargs):
        captured={};prior=sys.getprofile();path=ROOT/'weight_ledgers'/f'{stage}.pid{os.getpid()}.json'
        def trace(frame,event,arg):
            if frame.f_code is original.__code__ and event=='return':
                for key in ['params_dict','loaded_params','unloaded_params','skipped_checking_patterns']:
                    if key in frame.f_locals:captured[key]=frame.f_locals[key]
            if prior is not None:prior(frame,event,arg)
        sys.setprofile(trace)
        try:result=original(self,*args,**kwargs)
        except BaseException as exc:
            atomic(path,{'stage':stage,'status':'failed','error':repr(exc),'class':cls.__name__});raise
        finally:sys.setprofile(prior)
        if 'params_dict' not in captured or 'loaded_params' not in captured:
            raise RuntimeError('Native loader coverage capture missing; no fabricated readiness')
        params=captured['params_dict'];loaded=set(captured['loaded_params'])
        if 'unloaded_params' in captured:
            missing=set(captured['unloaded_params'])
        else:missing=set(params)-loaded
        reused={}
        for name in list(missing):
            if id(params[name]) in TARGET_PARAMS:
                reused[name]=TARGET_PARAMS[id(params[name])];missing.remove(name)
        report={'stage':stage,'status':'passed' if not missing else 'failed','class':cls.__name__,
                'parameter_count':len(params),'loaded_names':sorted(loaded),'missing':sorted(missing),
                'reused_target_parameter_identity':reused,
                'native_excluded_from_missing':sorted(set(params)-loaded-set(reused)-set(captured.get('unloaded_params',set(params)-loaded))),
                'native_skip_patterns':captured.get('skipped_checking_patterns',[]),
                'scope':'Native loader resolved-name coverage after asynchronous futures completed; not per-expert physical-byte or quality proof'}
        atomic(path,report)
        if missing:raise RuntimeError(f'{stage} native load coverage incomplete: {sorted(missing)[:20]}')
        if stage=='target':TARGET_PARAMS.update({id(p):name for name,p in params.items() if name in loaded})
        return result
    cls.load_weights=wrapped


instrument(DeepseekV4ForCausalLM,'target')
instrument(DeepseekV4ForCausalLMDSpark,'draft')


def register(self,ptrs,lengths):
    if len(ptrs)!=len(lengths):raise ValueError('PD memory descriptor lengths disagree')
    result=self.engine.batch_register_memory(ptrs,lengths)
    report={'ret':int(result),'entries':len(ptrs),'bytes':int(sum(lengths)),
            'zero_entries':sum(not p or not n for p,n in zip(ptrs,lengths)),
            'protocol':self._get_transfer_protocol(),'npu_id':self.npu_id}
    with (ROOT/'weight_ledgers'/f'PD_registration.pid{os.getpid()}.jsonl').open('a') as f:
        f.write(json.dumps(report)+'\n');f.flush()
    if result!=0:raise RuntimeError(f'PD memory registration failed: {result}')


AscendTransferEngine.batch_register=register
EntryClass=[DeepseekV4ForCausalLM,DeepseekV4ForCausalLMDSpark]
