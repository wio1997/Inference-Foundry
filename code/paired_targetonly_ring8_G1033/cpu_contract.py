"""Run only in the fixed remote SG image; metadata contract, no NPU init."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,os
import torch,torch_npu
assert not torch.npu.is_initialized()
import sglang.srt.mem_cache.deepseek_v4_memory_pool as pools
import sglang.srt.model_executor.pool_configurator as planner
from sglang.srt.disaggregation.ascend.conn import AscendKVManager
from sglang.srt.disaggregation.base.conn import StateType
import targetonly_ring8
root=Path(os.environ['RING8_CONTRACT_ROOT'])
fn=pools.get_compress_state_ring_size
def outcome(fn,ratio,drafts):
    try:return ('value',fn(ratio,drafts))
    except Exception as exc:return ('native_exception',type(exc).__name__,str(exc))
before={(ratio,drafts):outcome(fn,ratio,drafts) for ratio in [2,4,128] for drafts in [0,6]}
targetonly_ring8.install(root)
assert fn is pools.get_compress_state_ring_size
assert fn(2,0)==8
for key,value in before.items():
    if key!=(2,0):assert outcome(fn,*key)==value
assert planner.resolve_compress_state_ring_size is pools.resolve_compress_state_ring_size
assert planner.resolve_compress_state_ring_size.__globals__['get_compress_state_ring_size'] is fn

# Original native state-stride check, synthetic descriptor addresses only.
manager=object.__new__(AscendKVManager)
manager.is_mla_backend=True;manager.is_hybrid_mla_backend=False;manager.pp_size=1
manager.kv_args=SimpleNamespace(mla_compression_ratios=[0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]+[1]*20,
                              prefill_start_layer=0,prefill_end_layer=40)
args=dict(src_data_ptrs=[1000,2000,3000],dst_data_ptrs=[4000,5000,6000],item_lens=[32768]*3,
          state_type=StateType.DSV4_REQUEST_STATE,src_layer_ids=[2,8,14],dst_layer_ids=[2,8,14])
mismatch=manager._state_stride_mismatch(**args,dst_item_lens=[8192]*3)
assert mismatch is not None
assert manager._state_stride_mismatch(**args,dst_item_lens=[32768]*3) is None

# Native fixed-byte formula with synthetic request counts, no tensor allocation.
prices=[]
for slots in [1,16,129]:
    obj=SimpleNamespace(ring_sizes={2:8},attn_head_dim=512,num_layers=lambda ratio:3,
                        _get_num_req_slots=lambda count:slots)
    price=planner.DSV4PoolConfigurator._get_c2_state_fixed_bytes(obj,16)
    expected=((slots*8+8+1+1)//2*2)*1024*4*3
    assert price==expected
    prices.append(dict(req_slots=slots,actual_native_price=price,expected_physical_bytes=expected))
assert not torch.npu.is_initialized()
result=dict(status='CPU_CONTRACT_PASSED',NPU_initialized=False,native_stride_mismatch_8_vs2=mismatch,
            native_stride_8_vs8_accepted=True,resolver_identity=True,other_ratios_and_SP_unchanged=True,
            native_pricing_checks=prices,
            scope='CPU nativefunction/metadata only, syntheticpointers/counts. NotactualNPUallocation/PD/qualityproof.')
(root/'CPU_contract.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
