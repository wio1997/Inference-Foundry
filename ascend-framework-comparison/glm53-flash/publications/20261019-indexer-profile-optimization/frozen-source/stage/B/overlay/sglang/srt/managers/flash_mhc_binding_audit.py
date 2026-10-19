# SPDX-License-Identifier: Apache-2.0
"""Record actual Flash fixed mHC dispatch once per layer/phase after warmup."""
import hashlib,inspect,pathlib
_SEEN=set()
_SOURCES=None

def record(owner,phase,x,result):
    global _SOURCES
    from sglang.srt.managers.flash_pd_request_context import forward_tags
    tags=forward_tags()
    if not tags or not x.shape[0]:
        return
    layer=int(owner.layer_id)
    key=(layer,phase)
    if key in _SEEN:
        return
    from sglang.srt.hardware_backend.npu.flash_mhc_fixed_rows import FIXED_ROWS,flash_hc_pre_fixed,mhc
    from sglang.srt.disaggregation.ascend.flash_b103_pd_events import append
    assert FIXED_ROWS==32 and x.device.type=='npu' and 0<=layer<45
    assert phase in ('attn_pre','ffn_pre','post')
    if _SOURCES is None:
        _SOURCES={}
        for name,obj in [('model',type(owner)),('helper',flash_hc_pre_fixed),('audit',record),('native_mhc',mhc.hc_pre)]:
            p=pathlib.Path(inspect.getfile(obj))
            _SOURCES[name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    if phase!='post':
        assert result[3] is False and all(t.shape[0]==x.shape[0] for t in result[:3])
        dtypes=[str(t.dtype) for t in result[:3]]
        assert dtypes==['torch.bfloat16','torch.float32','torch.float32']
    else:
        assert result.shape[0]==x.shape[0] and str(result.dtype)=='torch.bfloat16'
        dtypes=[str(result.dtype)]
    append('fixed_mhc_binding',device=x.device.index,layer=layer,phase=phase,
           actual_rows=x.shape[0],internal_rows=FIXED_ROWS,output_dtypes=dtypes,
           actual_rids=sorted(t['rid'] for t in tags.values()),
           sources=_SOURCES,
           helper='sglang.srt.hardware_backend.npu.flash_mhc_fixed_rows',
           no_state_or_model_token_padding=True)
    _SEEN.add(key)
