"""Scoped native Ascend PD candidate; actual state-copy gates remain mandatory."""
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path


def install(root):
    import sglang.srt.arg_groups.deepseek_v4_hook as hooks
    if getattr(hooks,'_ds41_ascend_candidate_installed',False):return
    path=Path(hooks.__file__)
    pinned=json.loads((root/'pd_gate_source_sha256.json').read_text())[path.name]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=pinned:
        raise RuntimeError('Unexpected native feature-gate source')
    original=hooks.validate_deepseek_v41_features
    node=ast.parse(inspect.getsource(original)).body[0]
    changed=0
    for part in ast.walk(node):
        if isinstance(part,ast.Tuple) and [x.value if isinstance(x,ast.Constant) else None for x in part.elts]==['mooncake','nixl','mori']:
            part.elts.append(ast.Constant(value='ascend'));changed+=1
    if changed!=1:raise RuntimeError('Expected exactly one transport capability tuple')
    check=ast.parse('_ds41_ascend_scope_check(cfg)').body[0]
    cfg_assignments=[i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cfg' for t in n.targets)]
    if len(cfg_assignments)!=1:raise RuntimeError('Unexpected native resolving-view binding')
    at=cfg_assignments[0]+1
    node.body.insert(at,check)

    def scoped_check(cfg):
        if cfg.disaggregation_mode=='null' or cfg.speculative_algorithm is None or cfg.disaggregation_transfer_backend!='ascend':return
        from sglang.srt.utils import is_npu
        from sglang.srt.disaggregation.ascend.conn import AscendKVManager,MooncakeKVManager
        from sglang.srt.disaggregation.ascend.transfer_engine import TransferEngine
        from sglang.srt.speculative.ragged_verify import read_ragged_verify_mode,RaggedVerifyMode
        if not is_npu() or not issubclass(AscendKVManager,MooncakeKVManager):
            raise ValueError('Candidate requires the installed native NPU/Mooncake inheritance')
        if (cfg.tp_size,cfg.ep_size,cfg.pp_size,cfg.attn_cp_size,cfg.dcp_size)!=(8,8,1,1,1):
            raise ValueError('Ascend DSpark PD candidate is scoped to TP8 EP8 PP1 CP1 DCP1')
        if read_ragged_verify_mode() is not RaggedVerifyMode.STATIC:
            raise ValueError('Native static verify requirement must remain enabled')
        if os.environ.get('ASCEND_MF_TRANSFER_PROTOCOL')!='device_rdma' or not hasattr(TransferEngine.TransDataOpType,'DEVICE_RDMA'):
            raise ValueError('Candidate requires the actual DEVICE_RDMA SDK entry point')

    tree=ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[]))
    source=ast.unparse(tree)+'\n';overlay=root/f'native_v41_PD_gate.pid{os.getpid()}.py';overlay.write_text(source)
    context=dict(hooks.__dict__);context['_ds41_ascend_scope_check']=scoped_check
    exec(compile(tree,str(overlay),'exec'),context)
    updated=context[original.__name__]
    if original.__code__.co_freevars or updated.__code__.co_freevars:raise RuntimeError('Unexpected validator closure')
    hooks.__dict__['_ds41_ascend_scope_check']=scoped_check
    # Keep the function object so aliases already imported by model_hook agree.
    original.__code__=updated.__code__
    hooks._ds41_ascend_candidate_installed=True
    receipt={'native_source_sha256':pinned,'candidate_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
             'changed_backend_capability_tuples':changed,'native_static_verify_and_other_feature_guards_preserved':True,
             'scope':'TP8EP8PP1CP1DCP1 static verify DEVICE_RDMA candidate; native Ascend KV manager inherits Mooncake API',
             'actual_complete_model_PD_state_copy_verified':False}
    (root/f'PD_gate_candidate.pid{os.getpid()}.json').write_text(json.dumps(receipt,indent=2))
