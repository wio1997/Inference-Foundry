"""Actual staging/consumer AST with real CPU Torch tensors; no model/device load."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import json
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'llm_base_proposer.py'
TREE=ast.parse(SOURCE.read_text())
SFASOURCE=HERE.parent/'vllm-ascend__vllm_ascend__attention__context_parallel__sfa_cp.py'
MODES=NS(FULL='FULL',NONE='NONE')


def main():
    env=dict(torch=torch,CUDAGraphMode=MODES)
    helper=next(n for n in ast.walk(TREE) if isinstance(n,ast.FunctionDef) and n.name=='_glm_k1_stage_block_table')
    exec(compile(ast.Module(body=[helper],type_ignores=[]),str(SOURCE),'exec'),env)
    sfatree=ast.parse(SFASOURCE.read_text())
    local=next(n for n in ast.walk(sfatree) if isinstance(n,ast.FunctionDef) and n.name=='_get_dcp_local_block_table')
    exec(compile(ast.Module(body=[local],type_ignores=[]),str(SFASOURCE),'exec'),env)
    proposer=NS(block_table_tensor_clone=torch.zeros((512,71),dtype=torch.int32))
    builder=NS(max_local_block_table_cols=71)
    stage=lambda common:env['_glm_k1_stage_block_table'](proposer,common)
    retained=[];values=[]
    dummy=NS(block_table_tensor=torch.arange(71,dtype=torch.int32).reshape(1,71),num_reqs=1)
    stage(dummy)
    captured=env['_get_dcp_local_block_table'](builder,dummy.block_table_tensor,dummy.num_reqs)
    capture_ptr=captured.data_ptr()
    for step in range(5):
        # Actual runtime _adjust_tensor pads one request to two query rows.
        original=(torch.arange(71,dtype=torch.int32)+1000*(step+1)).reshape(1,71)
        padded=F.pad(original,(0,0,0,1),mode='constant',value=0)
        retained.append(padded)
        assert padded.data_ptr()!=capture_ptr
        common=NS(block_table_tensor=padded,num_reqs=1)
        stage(common)
        view=env['_get_dcp_local_block_table'](builder,common.block_table_tensor,common.num_reqs)
        assert view.shape==(1,71) and view.data_ptr()==capture_ptr
        assert torch.equal(captured,original) and torch.equal(view,original)
        assert common.block_table_tensor.shape==(2,71)
        assert not common.block_table_tensor[1].any()
        assert torch.equal(padded[0],original[0])
        values.append(view[0,0].item())
    assert len({x.data_ptr() for x in retained})==5
    assert values==[1000,2000,3000,4000,5000]
    unchanged=dummy.block_table_tensor.clone()
    stage(dummy)
    assert torch.equal(dummy.block_table_tensor,unchanged)
    assert dummy.block_table_tensor.shape==(1,71)
    unpadded=NS(block_table_tensor=torch.full((1,71),9000,dtype=torch.int32),num_reqs=1)
    original_unpadded=unpadded.block_table_tensor
    stage(unpadded)
    assert unpadded.block_table_tensor.data_ptr()==capture_ptr
    assert torch.equal(unpadded.block_table_tensor,original_unpadded)
    assert original_unpadded[0,0].item()==9000
    # The old late rebinding leaves a stale consumer view.
    stale=retained[0][:1];common=NS(block_table_tensor=retained[0])
    stage(common)
    assert stale.data_ptr()!=common.block_table_tensor.data_ptr()
    assert stale.data_ptr()!=capture_ptr
    dummyfn=next(n for n in ast.walk(TREE) if isinstance(n,ast.FunctionDef) and n.name=='dummy_run')
    propose=next(n for n in ast.walk(TREE) if isinstance(n,ast.FunctionDef) and n.name=='_propose')
    call=lambda n,name:next(x for x in ast.walk(n) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr==name)
    dummy_stage=call(dummyfn,'_glm_k1_stage_block_table');dummy_build=call(dummyfn,'build_for_graph_capture')
    real_stage=call(propose,'_glm_k1_stage_block_table');real_build=call(propose,'build_draft_attn_metadata')
    assert dummy_stage.lineno<dummy_build.lineno and real_stage.lineno<real_build.lineno
    dummybranch=next(n for n in ast.walk(dummyfn) if isinstance(n,ast.If) and dummy_stage in list(ast.walk(n)) and isinstance(n.test,ast.BoolOp) and any(isinstance(x,ast.Attribute) and x.attr=='_glm_k1_graph' for x in ast.walk(n.test)))
    realbranch=next(n for n in ast.walk(propose) if isinstance(n,ast.If) and real_stage in list(ast.walk(n)) and isinstance(n.test,ast.BoolOp) and any(isinstance(x,ast.Attribute) and x.attr=='_glm_k1_graph' for x in ast.walk(n.test)))
    for flag in (False,True):
        for index in (0,1):
            assert eval(compile(ast.Expression(dummybranch.test),'dummy_scope','eval'),dict(self=NS(_glm_k1_graph=flag),draft_index=index))==(flag and index==0)
        for mode in ('NONE','FULL'):
            assert eval(compile(ast.Expression(realbranch.test),'real_scope','eval'),dict(self=NS(_glm_k1_graph=flag),aclgraph_runtime_mode=mode,CUDAGraphMode=MODES))==(flag and mode=='FULL')
    rejected=0
    for tensor in (torch.zeros(513,71,dtype=torch.int32),torch.zeros(2,70,dtype=torch.int32),torch.zeros(71,dtype=torch.int32)):
        common=NS(block_table_tensor=tensor)
        try:stage(common)
        except AssertionError:rejected+=1
        else:raise AssertionError('invalid shape admitted')
        assert common.block_table_tensor is tensor
    row=dict(passed=True,CPU_only=True,NPU_initialized=False,model_requests=0,real_Torch_CPU=True,
        transient_steps=5,same_capture_runtime_pointer=True,live_consumer_refresh=True,padded_zero_tail=True,
        original_late_binding_failure_reproduced=True,shape_rejections=rejected,alias_self_copy_skipped=True,unpadded_path=True,
        actual_dummy_and_runtime_call_order=True,scope_checks=8,
        proposer_sha256=__import__('hashlib').sha256(SOURCE.read_bytes()).hexdigest(),
        limitations='No device graph replay, stream concurrency or numerical model proof.')
    (HERE/'block_table_CPU_result.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row))


if __name__=='__main__':main()
